"""The ``kanban_worktree_created`` lifecycle hook fired on kanban worktree creation.

A linked worktree is a fresh checkout: it holds tracked files and nothing else,
so every gitignored file is absent by definition — including the ``.env`` most
projects cannot start without. Which secrets a project needs is deployment
knowledge no code in this repo can hold, so ``_ensure_git_worktree`` fires a
plugin hook right after ``git worktree add`` succeeds and BEFORE the worker is
spawned. A Python plugin subscribes with ``ctx.register_hook``; a shell script
subscribes through the ``hooks:`` block in ``config.yaml`` (the same bridge
every other plugin-hook event uses).

Two things here are contracts, not implementation details, and neither may
drift silently:

* the payload external hooks are written against — ``worktree_path``,
  ``repo_root``, ``branch``, ``task_id`` — and the fact that the checkout
  already exists when the hook runs;
* the atomicity of failure. ``dispatch_once`` turns the raise into a released
  claim and retries the card, so a blocked hook MUST leave no worktree behind:
  otherwise the retry takes ``_ensure_git_worktree``'s idempotent early return,
  never re-fires the hook, and spawns the worker into the unseeded checkout the
  block existed to prevent.

The plugin-side cases register a callback on the real plugin manager; the
shell-side cases go through ``agent.shell_hooks.register_from_config`` with a
real script; the CLI cases run ``hermes kanban claim/dispatch/daemon`` bodies
against a real kanban DB and a ``hooks:`` block in a real ``config.yaml``. No
``subprocess`` or ``invoke_hook`` mocks — the one stand-in is the
removal-failure case, which stubs the module's ``_git`` wrapper for
``worktree remove`` only, and the CLI cases, which stub the worker spawn.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable

import psutil
import pytest

from agent import shell_hooks
from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli.plugins import get_plugin_manager

HOOK = "kanban_worktree_created"

# Mirrors ``_record_task_failure`` in hermes_cli/kanban_db_dispatch.py: the exception text is
# prefixed with ``workspace: `` by dispatch_once and cut to 500 chars before it lands on the card.
_CARD_REASON_LIMIT = 500


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME (no kanban DB needed — this is worktree plumbing)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


@pytest.fixture
def hook_registry(kanban_home: Path):
    """Snapshot the plugin manager's hook table and the shell-hook idempotence set, restore both."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    shell_hooks.reset_for_tests()
    try:
        yield mgr
    finally:
        mgr._hooks = saved
        shell_hooks.reset_for_tests()


def _subscribe(mgr, callback: Callable[..., Any]) -> None:
    mgr._hooks.setdefault(HOOK, []).append(callback)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A one-commit git repo to hang linked worktrees off."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    (root / "README.md").write_text("base\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "init")
    return root


def _worktrees(repo_root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    return [ln.split(" ", 1)[1] for ln in out.splitlines() if ln.startswith("worktree ")]


def _install_shell_hook(tmp_path: Path, body: str, **entry: Any) -> Path:
    """Write a Python hook script and register it exactly as ``config.yaml`` would.

    Python rather than ``/bin/sh`` so the same script runs on every host; the bridge splits the
    command line itself and never goes through a shell.
    """
    script = tmp_path / "worktree-created-hook.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    cfg = {"hooks": {HOOK: [{"command": f'"{sys.executable}" "{script}"', **entry}]}}
    registered = shell_hooks.register_from_config(cfg, accept_hooks=True)
    assert [s.event for s in registered] == [HOOK], "the hooks: entry must wire onto the event"
    return script


def _card_reason(exc: BaseException) -> str:
    return f"workspace: {exc}"[:_CARD_REASON_LIMIT]


# ---------------------------------------------------------------------------
# Plugin hook: payload, timing, idempotence
# ---------------------------------------------------------------------------


def test_hook_fires_once_after_checkout_with_worktree_repo_branch_and_task(
    kanban_home: Path, repo: Path, hook_registry,
) -> None:
    """Payload names the new worktree, its repo, its branch and the card; the checkout exists already."""
    seen: list[dict] = []

    def on_created(**kw):
        seen.append(dict(kw, readme_present=Path(kw["worktree_path"], "README.md").is_file()))

    _subscribe(hook_registry, on_created)
    target = repo / ".worktrees" / "t_abc"

    kbw._ensure_git_worktree(repo, target, "wt/t_abc", task_id="t_abc")

    assert target.is_dir()
    assert len(seen) == 1, f"hook fired {len(seen)} times, expected exactly once"
    kw = seen[0]
    assert Path(kw["worktree_path"]) == target
    assert Path(kw["repo_root"]) == repo
    assert kw["branch"] == "wt/t_abc"
    assert kw["task_id"] == "t_abc"
    assert kw["readme_present"], "the hook must run after the checkout is materialised, not before"


def test_existing_worktree_does_not_re_fire(kanban_home: Path, repo: Path, hook_registry) -> None:
    """A *created* event: re-resolving an existing worktree takes the idempotent path in silence."""
    calls: list[str] = []
    _subscribe(hook_registry, lambda **kw: calls.append(kw["worktree_path"]))
    target = repo / ".worktrees" / "t_once"

    kbw._ensure_git_worktree(repo, target, "wt/t_once", task_id="t_once")
    kbw._ensure_git_worktree(repo, target, "wt/t_once", task_id="t_once")

    assert calls == [str(target)]


def test_no_subscriber_is_a_no_op(kanban_home: Path, repo: Path, hook_registry) -> None:
    target = repo / ".worktrees" / "t_nohook"

    kbw._ensure_git_worktree(repo, target, "wt/t_nohook", task_id="t_nohook")

    assert (target / "README.md").is_file()


# ---------------------------------------------------------------------------
# Plugin hook: a block directive fails the card, atomically
# ---------------------------------------------------------------------------


def test_block_directive_raises_with_its_message_and_discards_the_worktree(
    kanban_home: Path, repo: Path, hook_registry,
) -> None:
    """``{"action": "block", "message"}`` raises with the message and leaves no worktree behind."""

    def half_seed_then_block(worktree_path: str, **kw):
        # Seed something first, so removal has to cope with a dirty/untracked tree —
        # which is what a hook that failed halfway actually leaves.
        Path(worktree_path, ".env").write_text("half-written\n", encoding="utf-8")
        return {"action": "block", "message": "no .env for this project"}

    _subscribe(hook_registry, half_seed_then_block)
    target = repo / ".worktrees" / "t_fail"

    with pytest.raises(RuntimeError) as excinfo:
        kbw._ensure_git_worktree(repo, target, "wt/t_fail", task_id="t_fail")

    message = str(excinfo.value)
    assert HOOK in message
    assert "no .env for this project" in message
    # The raise is about the seeding, not about git; it must not blame git.
    assert "git worktree add failed" not in message
    # Whether a retry is safe comes BEFORE the hook's own text (see the budget tests below).
    assert message.index("was removed") < message.index("no .env for this project")
    # Atomic: the unseeded tree is gone, and git no longer tracks it.
    assert not target.exists()
    assert str(target) not in _worktrees(repo)


def test_retry_after_a_block_re_creates_and_re_fires(kanban_home: Path, repo: Path, hook_registry) -> None:
    """The whole point of discarding the tree: the next attempt re-fires the hook.

    ``dispatch_once`` records the raise as a spawn failure and releases the
    claim, so the card comes back. If the blocked attempt had left its worktree
    in place, this second call would take the idempotent early return, skip the
    hook, and hand a worker the unseeded checkout with no error attached.
    """
    calls: list[str] = []

    def fail_once_then_seed(worktree_path: str, **kw):
        calls.append(worktree_path)
        if len(calls) == 1:
            return {"action": "block", "message": "boom"}
        Path(worktree_path, ".seeded").write_text("ok\n", encoding="utf-8")
        return None

    _subscribe(hook_registry, fail_once_then_seed)
    target = repo / ".worktrees" / "t_retry"

    with pytest.raises(RuntimeError, match=HOOK):
        kbw._ensure_git_worktree(repo, target, "wt/t_retry", task_id="t_retry")
    assert not target.exists()

    kbw._ensure_git_worktree(repo, target, "wt/t_retry", task_id="t_retry")

    assert calls == [str(target), str(target)]
    assert (target / ".seeded").is_file()
    assert (target / "README.md").is_file()  # a real checkout, not a leftover shell


# ---------------------------------------------------------------------------
# The card reason: the retry verdict survives the 500-char cut in both branches
# ---------------------------------------------------------------------------


def test_card_reason_keeps_the_retry_verdict_ahead_of_a_chatty_hook(
    kanban_home: Path, repo: Path, hook_registry,
) -> None:
    """A 5000-char hook message cannot push "the worktree was removed" off the card."""
    _subscribe(hook_registry, lambda **kw: {"action": "block", "message": "x" * 5000})
    target = repo / ".worktrees" / "t_chatty"

    with pytest.raises(RuntimeError) as excinfo:
        kbw._ensure_git_worktree(repo, target, "wt/t_chatty", task_id="t_chatty")

    reason = _card_reason(excinfo.value)
    assert "was removed" in reason
    assert "re-seeds it" in reason
    assert not target.exists()


def test_card_reason_keeps_remove_by_hand_when_removal_itself_fails(
    kanban_home: Path, repo: Path, hook_registry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If git cannot remove the tree, the operator instruction survives both the git noise and the hook's."""
    real_git = kbw._git

    def git_with_broken_remove(repo_root: Path, *args: str, timeout: int):
        if args[:2] == ("worktree", "remove"):
            return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: " + "y" * 5000)
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", git_with_broken_remove)
    _subscribe(hook_registry, lambda **kw: {"action": "block", "message": "x" * 5000})
    target = repo / ".worktrees" / "t_stuck"

    with pytest.raises(RuntimeError) as excinfo:
        kbw._ensure_git_worktree(repo, target, "wt/t_stuck", task_id="t_stuck")

    reason = _card_reason(excinfo.value)
    assert "could NOT be removed" in reason
    assert "remove it by hand" in reason
    assert str(target) in reason
    assert target.is_dir()  # the residual the message warns about is real


# ---------------------------------------------------------------------------
# Shell hook through the hooks: bridge — the same script an operator would install
# ---------------------------------------------------------------------------


def test_shell_hook_seeds_the_worktree_from_the_stdin_payload(
    kanban_home: Path, repo: Path, hook_registry, tmp_path: Path,
) -> None:
    """A ``hooks:`` entry gets the worktree in ``extra`` and can seed it; exit 0 lets the card proceed."""
    log = tmp_path / "payload.json"
    _install_shell_hook(
        tmp_path,
        f"""
        import json, pathlib, sys
        payload = json.load(sys.stdin)
        pathlib.Path({str(log)!r}).write_text(json.dumps(payload), encoding="utf-8")
        pathlib.Path(payload["extra"]["worktree_path"], ".env").write_text("SECRET=1\\n", encoding="utf-8")
        """,
        timeout=30,
    )
    target = repo / ".worktrees" / "t_shell"

    kbw._ensure_git_worktree(repo, target, "wt/t_shell", task_id="t_shell")

    assert (target / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
    payload = json.loads(log.read_text(encoding="utf-8"))
    assert payload["hook_event_name"] == HOOK
    extra = payload["extra"]
    assert Path(extra["worktree_path"]) == target
    assert Path(extra["repo_root"]) == repo
    assert extra["branch"] == "wt/t_shell"
    assert extra["task_id"] == "t_shell"


def test_shell_hook_exit_2_fails_the_card_with_its_stderr_and_discards_the_worktree(
    kanban_home: Path, repo: Path, hook_registry, tmp_path: Path,
) -> None:
    """Exit 2 is the bridge's block verb: the stderr line becomes the card reason, the tree is gone."""
    _install_shell_hook(
        tmp_path,
        """
        import sys
        sys.stdin.read()
        print("no .env for this project", file=sys.stderr)
        sys.exit(2)
        """,
        timeout=30,
    )
    target = repo / ".worktrees" / "t_shell_fail"

    with pytest.raises(RuntimeError) as excinfo:
        kbw._ensure_git_worktree(repo, target, "wt/t_shell_fail", task_id="t_shell_fail")

    assert "no .env for this project" in str(excinfo.value)
    assert not target.exists()
    assert str(target) not in _worktrees(repo)


def test_shell_hook_timeout_with_fail_closed_fails_the_card_and_kills_the_hook_tree(
    kanban_home: Path, repo: Path, hook_registry, tmp_path: Path,
) -> None:
    """A hung hook fails the card instead of wedging the dispatcher, and its children die with it.

    The hook spawns a grandchild (an ``npm ci`` stand-in) and then hangs. Killing only the hook
    would leave that grandchild writing into a worktree git has just force-removed, and the retry's
    ``git worktree add`` would then fail on "already exists".
    """
    pid_file = tmp_path / "grandchild.pid"
    _install_shell_hook(
        tmp_path,
        f"""
        import pathlib, subprocess, sys, time
        sys.stdin.read()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pathlib.Path({str(pid_file)!r}).write_text(str(child.pid), encoding="utf-8")
        time.sleep(60)
        """,
        # Floor is 1s; 3s leaves headroom for two CPython start-ups on a loaded runner before
        # the pid file is read below.
        timeout=3,
        fail_closed=True,
    )
    target = repo / ".worktrees" / "t_hang"

    with pytest.raises(RuntimeError) as excinfo:
        kbw._ensure_git_worktree(repo, target, "wt/t_hang", task_id="t_hang")

    assert "timed out" in str(excinfo.value)
    assert not target.exists()
    assert str(target) not in _worktrees(repo)
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while psutil.pid_exists(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(grandchild), "the hook's grandchild survived the timeout kill"


# ---------------------------------------------------------------------------
# CLI entry points: the hooks: block is wired before any of them creates a worktree
# ---------------------------------------------------------------------------
#
# ``hermes chat`` / ``gateway run`` / ``cron run`` register config.yaml's ``hooks:`` in
# ``_prepare_agent_startup`` (hermes_cli/main.py) and the gateway-hosted dispatcher in
# ``GatewayStartupMixin._register_config_hooks``. ``hermes kanban`` is not an agent command,
# so ``claim``, ``dispatch`` and ``daemon --force`` must wire the block themselves — otherwise a
# configured seed script silently never runs and the unseeded tree is reused by the next tick.


@pytest.fixture
def cli_home(kanban_home: Path, hook_registry, repo: Path, tmp_path: Path) -> Path:
    """Real kanban DB plus a config.yaml whose hooks: block seeds ``.env`` into a new worktree."""
    kb.init_db()
    script = tmp_path / "seed-worktree.py"
    script.write_text(textwrap.dedent(
        """
        import json, pathlib, sys
        payload = json.load(sys.stdin)
        pathlib.Path(payload["extra"]["worktree_path"], ".env").write_text("SEEDED=1\\n", encoding="utf-8")
        """
    ), encoding="utf-8")
    (kanban_home / "config.yaml").write_text(json.dumps({
        "hooks_auto_accept": True,  # the headless consent channel the docs name
        "hooks": {HOOK: [{"command": f'"{sys.executable}" "{script}"', "timeout": 30}]},
    }), encoding="utf-8")
    return kanban_home


def _create_worktree_card(repo: Path, *, assignee: str | None = None) -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(
            conn, title="needs .env", assignee=assignee,
            workspace_kind="worktree", workspace_path=str(repo),
        )


def test_hermes_kanban_claim_registers_config_hooks_before_resolving_the_workspace(
    cli_home: Path, repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``hermes kanban claim`` seeds through the ``hooks:`` block, not only through Python plugins."""
    tid = _create_worktree_card(repo)

    rc = kanban_cli._cmd_claim(argparse.Namespace(task_id=tid, ttl=600, accept_hooks=False))

    assert rc == 0, capsys.readouterr()
    worktree = repo / ".worktrees" / tid
    assert (worktree / ".env").read_text(encoding="utf-8") == "SEEDED=1\n"
    with kbc.connect_closing() as conn:
        assert Path(kb.get_task(conn, tid).workspace_path) == worktree


def _stub_worker_spawn(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``hermes kanban dispatch/daemon`` spawn a real ``hermes chat`` worker; record the workspace instead."""
    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, workspace, *, board=None: spawned.append(workspace))
    return spawned


def test_hermes_kanban_dispatch_registers_config_hooks_before_dispatch_once(
    cli_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A one-shot ``hermes kanban dispatch`` hands the worker a worktree the ``hooks:`` script has seeded."""
    tid = _create_worktree_card(repo, assignee="default")
    spawned = _stub_worker_spawn(monkeypatch)

    rc = kanban_cli._cmd_dispatch(
        argparse.Namespace(dry_run=False, max=None, failure_limit=2, json=False, accept_hooks=False),
    )

    assert rc == 0, capsys.readouterr()
    assert [Path(p) for p in spawned] == [repo / ".worktrees" / tid]
    assert (repo / ".worktrees" / tid / ".env").read_text(encoding="utf-8") == "SEEDED=1\n"


def test_hermes_kanban_daemon_force_registers_config_hooks_before_the_loop(
    cli_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The standalone loop (``daemon --force``) ticks with the ``hooks:`` block already wired."""
    tid = _create_worktree_card(repo, assignee="default")
    spawned = _stub_worker_spawn(monkeypatch)

    def one_tick(*, interval, max_spawn, failure_limit, on_tick):
        # The body of ``kbd.run_daemon``'s loop, once, instead of sleeping until a signal.
        with contextlib.closing(kbc.connect()) as conn:
            on_tick(kbd.dispatch_once(conn, max_spawn=max_spawn, failure_limit=failure_limit))

    monkeypatch.setattr(kbd, "run_daemon", one_tick)

    rc = kanban_cli._cmd_daemon(argparse.Namespace(
        force=True, pidfile=None, interval=5, max=None, failure_limit=2, verbose=False, accept_hooks=False,
    ))

    assert rc == 0, capsys.readouterr()
    assert [Path(p) for p in spawned] == [repo / ".worktrees" / tid]
    assert (repo / ".worktrees" / tid / ".env").read_text(encoding="utf-8") == "SEEDED=1\n"
