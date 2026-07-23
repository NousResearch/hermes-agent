"""HER-95 — RED probes for the functional integration blockers.

The two exact-head reviews of PR #69955 rejected the change because the gate was
only a manual CLI: it was never wired into the real Hermes pre-mutation path, and
the ``hermes-immo`` domain denial depended on a caller remembering to pass
``--profile`` / ``--domain-prefixes``.

These tests exercise the REAL control flow that ``run_agent._invoke_tool`` uses:
``register_from_config()`` wires ``scripts/factory_admission_hook.py`` onto the
plugin manager as a ``pre_tool_call`` hook, and
``plugins.get_pre_tool_call_block_message()`` (the exact call site that vetoes a
build-capable tool before it executes) must surface the block. The profile /
domain flags live in the hook's configured command line — i.e. in the profile's
``cli-config.yaml`` — so the denial is automatic, not caller-remembered.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "factory_lane.py"
HOOK = REPO_ROOT / "scripts" / "factory_admission_hook.py"


def run_lane(registry, *args, check=False):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry), *args],
        capture_output=True, text=True, timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result


def make_git_worktree(path: Path):
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True)
    (path / "README.md").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def read_owner(registry: Path, key: str):
    return json.loads((registry / "locks" / key / "owner.json").read_text())


def hook_command(registry, agent, *, profile=None, domain_prefixes=None):
    parts = [sys.executable, str(HOOK), "--registry", str(registry), "--agent", agent]
    if profile:
        parts += ["--profile", profile]
    if domain_prefixes:
        parts += ["--domain-prefixes", domain_prefixes]
    return " ".join(parts)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Fresh plugin manager + shell-hook registry, isolated HERMES_HOME."""
    from hermes_cli import plugins
    from agent import shell_hooks

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    yield plugins, shell_hooks
    shell_hooks.reset_for_tests()


def register_gate(shell_hooks, command, matcher="terminal|patch|write_file|str_replace_editor"):
    cfg = {"hooks": {"pre_tool_call": [{"matcher": matcher, "command": command}]}}
    registered = shell_hooks.register_from_config(cfg, accept_hooks=True)
    assert len(registered) == 1, "gate hook did not register"


# ---------------------------------------------------------------------------
# Functional blocker 1 — real pre-mutation wiring
# ---------------------------------------------------------------------------

def test_gate_blocks_build_capable_tool_in_foreign_owned_worktree(wired, tmp_path, monkeypatch):
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    # A live owner (parent identity transported via --owner-pid) owns the tree.
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)

        register_gate(shell_hooks, hook_command(registry, "default"))

        # Intruder session tries to run `terminal` inside that worktree.
        monkeypatch.chdir(worktree)
        msg = plugins.get_pre_tool_call_block_message(
            tool_name="terminal", args={"command": "rm -rf /"}, session_id="intruder",
        )
        assert msg is not None, "gate did not block a mutation in a foreign-owned worktree"
        assert "HER-95" in msg
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_gate_uses_terminal_workdir_when_session_cwd_is_elsewhere(wired, tmp_path, monkeypatch):
    """The original SCA-740 collision used an explicit tool workdir while the
    gateway session itself was outside the worktree.  The gate must evaluate
    that effective target, not merely the gateway process cwd."""
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    outside = tmp_path / "home"
    make_git_worktree(worktree)
    outside.mkdir()

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)
        register_gate(shell_hooks, hook_command(registry, "default"))

        monkeypatch.chdir(outside)
        msg = plugins.get_pre_tool_call_block_message(
            tool_name="terminal",
            args={"command": "touch blocked.txt", "workdir": str(worktree)},
            session_id="intruder",
        )
        assert msg is not None, "explicit terminal workdir bypassed worktree admission"
        assert "HER-95" in msg
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_gate_uses_write_file_path_when_session_cwd_is_elsewhere(wired, tmp_path, monkeypatch):
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    outside = tmp_path / "home"
    make_git_worktree(worktree)
    outside.mkdir()

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)
        register_gate(shell_hooks, hook_command(registry, "default"))

        monkeypatch.chdir(outside)
        msg = plugins.get_pre_tool_call_block_message(
            tool_name="write_file",
            args={"path": str(worktree / "blocked.txt"), "content": "no"},
            session_id="intruder",
        )
        assert msg is not None, "absolute write_file path bypassed worktree admission"
        assert "HER-95" in msg
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_gate_detects_owned_worktree_referenced_in_terminal_command(wired, tmp_path, monkeypatch):
    """An explicit workdir is not mandatory in the terminal contract.  Direct
    absolute paths in the command (for example ``git -C`` or ``cd``) must also
    be evaluated against live owners."""
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    outside = tmp_path / "home"
    make_git_worktree(worktree)
    outside.mkdir()

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)
        register_gate(shell_hooks, hook_command(registry, "default"))

        monkeypatch.chdir(outside)
        msg = plugins.get_pre_tool_call_block_message(
            tool_name="terminal",
            args={"command": f"git -C {worktree} status"},
            session_id="intruder",
        )
        assert msg is not None, "absolute path in terminal command bypassed admission"
        assert "HER-95" in msg
    finally:
        holder.terminate()
        holder.wait(timeout=10)


@pytest.mark.parametrize(
    "command",
    ["git -C repo status", "cd repo && touch blocked.txt", "touch repo/blocked.txt"],
)
def test_gate_detects_relative_worktree_in_terminal_command(
    wired, tmp_path, monkeypatch, command,
):
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)
        register_gate(shell_hooks, hook_command(registry, "default"))
        monkeypatch.chdir(tmp_path)

        msg = plugins.get_pre_tool_call_block_message(
            tool_name="terminal", args={"command": command}, session_id="intruder",
        )
        assert msg is not None, f"relative terminal target bypassed admission: {command}"
        assert "HER-95" in msg
    finally:
        holder.terminate()
        holder.wait(timeout=10)


@pytest.mark.parametrize("relative", [True, False])
def test_gate_reads_apply_patch_change_paths(wired, tmp_path, monkeypatch, relative):
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)
        register_gate(
            shell_hooks,
            hook_command(registry, "default"),
            matcher="terminal|patch|write_file|str_replace_editor|apply_patch",
        )
        monkeypatch.chdir(tmp_path)
        path = "repo/a.py" if relative else str(worktree / "a.py")

        msg = plugins.get_pre_tool_call_block_message(
            tool_name="apply_patch",
            args={"changes": [{"kind": "add", "path": path}]},
            session_id="intruder",
        )
        assert msg is not None, "apply_patch changes[*].path bypassed admission"
        assert "HER-95" in msg
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_gate_allows_owning_session_mutation(wired, tmp_path, monkeypatch):
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "owner", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)

        register_gate(shell_hooks, hook_command(registry, "default"))

        monkeypatch.chdir(worktree)
        msg = plugins.get_pre_tool_call_block_message(
            tool_name="terminal", args={"command": "ls"}, session_id="owner",
        )
        assert msg is None, f"gate wrongly blocked the owning session: {msg!r}"
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_gate_ignores_readonly_tool_not_matched(wired, tmp_path, monkeypatch):
    """A read-only reviewer tool (not in the mutation matcher) never fires the
    gate, so a read-only reviewer coexists with the owner unimpeded."""
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "owner",
             "--worktree", str(worktree), check=True)
    register_gate(shell_hooks, hook_command(registry, "default"))

    monkeypatch.chdir(worktree)
    msg = plugins.get_pre_tool_call_block_message(
        tool_name="read_file", args={"path": "README.md"}, session_id="reviewer",
    )
    assert msg is None


def test_gate_fails_open_when_no_claim(wired, tmp_path, monkeypatch):
    """Advisory posture: an ownerless / empty registry lets the tool proceed."""
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    register_gate(shell_hooks, hook_command(registry, "default"))

    monkeypatch.chdir(worktree)
    msg = plugins.get_pre_tool_call_block_message(
        tool_name="terminal", args={"command": "ls"}, session_id="anyone",
    )
    assert msg is None


# ---------------------------------------------------------------------------
# Functional blocker 2 — automatic business-profile domain denial
# ---------------------------------------------------------------------------

def test_gate_blocks_business_profile_out_of_domain_without_caller_flags(wired, tmp_path, monkeypatch):
    """The caller passes NO profile/domain flags at tool time — they come from
    the hook's configured command line (the hermes-immo profile config). Even
    when the same session owns the lane (no ownership conflict), a product lane
    outside the business domain is refused automatically."""
    plugins, shell_hooks = wired
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)

    # hermes-immo itself owns a product lane (claim does not enforce domain).
    run_lane(registry, "claim", "SCA-740", "--agent", "hermes-immo", "--session", "s1",
             "--worktree", str(worktree), check=True)

    register_gate(
        shell_hooks,
        hook_command(registry, "hermes-immo", profile="hermes-immo", domain_prefixes="JYI,HER"),
    )

    monkeypatch.chdir(worktree)
    msg = plugins.get_pre_tool_call_block_message(
        tool_name="patch", args={"path": "x"}, session_id="s1",
    )
    assert msg is not None, "business profile mutated a lane outside its domain"
    assert "SCA-740" in msg


# ---------------------------------------------------------------------------
# Functional blocker 4 — parent process identity transport (not the subprocess)
# ---------------------------------------------------------------------------

def test_admit_records_transported_parent_pid_not_subprocess(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
                 "--session", "s1", "--worktree", str(worktree),
                 "--owner-pid", str(holder.pid), check=True)
        owner = read_owner(registry, "HER-95")
        assert owner["pid"] == holder.pid, "owner.json did not record the transported parent pid"
        assert owner.get("process_start_time"), "parent start-time baseline was not recorded"
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_admit_refuses_transported_identity_of_dead_process(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=10)

    result = run_lane(
        registry, "admit", "HER-95", "--mode", "owner", "--agent", "default",
        "--session", "s1", "--worktree", str(worktree), "--owner-pid", str(dead.pid),
    )
    assert result.returncode != 0, "admit accepted a dead transported owner pid"
    assert not (registry / "locks" / "HER-95" / "owner.json").exists()
