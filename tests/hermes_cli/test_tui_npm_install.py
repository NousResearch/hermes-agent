"""TUI launch consumes the shared dependency preparation and compiler."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from hermes_cli import main_tui_launch
from tests.hermes_cli.test_source_build import source_checkout, source_products, _events


def _touch_tui_entry(root: Path) -> None:
    entry = root / "dist" / "entry.js"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("console.log('tui')")


def test_need_rebuild_when_tui_bundle_missing(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "entry.tsx").write_text("console.log('src')")

    assert main_tui_launch._tui_need_rebuild(tmp_path) is True


def test_unreceipted_bundle_is_stale_even_when_newer(tmp_path: Path) -> None:
    _touch_tui_entry(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "entry.tsx").write_text("console.log('src')")
    os.utime(src / "entry.tsx", (100, 100))
    os.utime(tmp_path / "dist" / "entry.js", (200, 200))

    assert main_tui_launch._tui_need_rebuild(tmp_path) is True



@pytest.fixture
def tui_source(source_products, monkeypatch):
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.delenv("HERMES_TUI_FORCE_BUILD", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setenv("HERMES_NODE", shutil.which("node"))
    monkeypatch.setattr(main_tui_launch, "_find_bundled_tui", lambda: None)
    return source_products


@pytest.mark.platforms("posix")
def test_source_launch_prepares_base_union_but_compiles_only_tui(tui_source):
    root, acquired = tui_source
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert cwd == root / "ui-tui"
    assert argv[-1] == str(cwd / "dist/entry.js")
    assert (cwd / "dist/entry.js").read_text() == "tui"
    assert [event["step"] for event in _events(root)] == ["deps", "tui"]
    assert acquired == ["pnpm"]
    assert (root / "ui-tui/node_modules").exists()
    assert (root / "web/node_modules").exists()
    assert not (root / "apps/desktop/node_modules").exists()
    # A launch someone waits for, which can fall back, gives up on an unreachable registry sooner.
    assert _events(root)[0]["retries"] == "1"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("where", ["environment", "pnpm global config", "pnpm-workspace.yaml"])
def test_source_launch_keeps_the_users_download_retries(tui_source, tmp_path, monkeypatch, where):
    """A retry count the user chose (for a flaky mirror) is theirs: a launch lowers only pnpm's default."""
    root, _ = tui_source
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    if where == "environment":
        monkeypatch.setenv("pnpm_config_fetch_retries", "5")
    elif where == "pnpm global config":
        (tmp_path / "xdg/pnpm").mkdir(parents=True)
        (tmp_path / "xdg/pnpm/config.yaml").write_text("fetchRetries: 5\n", encoding="utf-8")
    else:
        workspace = root / "pnpm-workspace.yaml"
        workspace.write_text("fetchRetries: 5\n" + workspace.read_text(encoding="utf-8"), encoding="utf-8")
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert _events(root)[0].get("retries") == ("5" if where == "environment" else None)


@pytest.mark.platforms("posix")
def test_source_launch_ignores_an_npm_retry_count_pnpm_ignores(tui_source, monkeypatch):
    """pnpm 12 reads no fetch-retries from an npmrc, nor npm_config_fetch_retries: a count set only
    there would leave pnpm's default of 2 retries, so the launch still lowers it."""
    import os

    root, _ = tui_source
    npmrc = Path(os.environ["HERMES_HOME"]) / "npmrc"
    npmrc.write_text(npmrc.read_text(encoding="utf-8") + "fetch-retries=5\n", encoding="utf-8")
    monkeypatch.setenv("npm_config_fetch_retries", "5")
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert _events(root)[0]["retries"] == "1"


@pytest.mark.platforms("posix")
def test_source_compile_failure_stops_launch_without_reinstall(tui_source):
    root, acquired = tui_source
    (root / "fail-tui").touch()
    with pytest.raises(subprocess.CalledProcessError):
        main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert [event["step"] for event in _events(root)] == ["deps", "tui"]
    assert acquired == ["pnpm"]


@pytest.mark.platforms("posix")
def test_failed_dependency_preparation_launches_the_previous_build(tui_source, monkeypatch, capsys):
    """#150031: after an update moved the TUI's inputs, an install that cannot run (here lazy
    installs are disabled; offline, the registry is unreachable) still starts the previous bundle,
    and hands the caller the warning to show again after the TUI's screen clear."""
    root, _ = tui_source
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    with pytest.raises(subprocess.CalledProcessError):
        main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    _touch_tui_entry(root / "ui-tui")
    notices = []
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False, notices=notices)
    assert argv[-1] == str(root / "ui-tui/dist/entry.js")
    assert subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip() == "tui"
    assert "previous TUI build was started" in capsys.readouterr().err
    assert len(notices) == 1 and "hermes update" in notices[0]
    assert _events(root) == []
    # --dev runs the sources, and a forced build asks for them: both need the dependencies.
    with pytest.raises(subprocess.CalledProcessError):
        main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=True)
    monkeypatch.setenv("HERMES_TUI_FORCE_BUILD", "1")
    with pytest.raises(subprocess.CalledProcessError):
        main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)


@pytest.mark.platforms("posix")
def test_fallback_names_a_refusal_that_printed_nothing(tui_source, monkeypatch, capsys, caplog):
    """PM refusing to fetch pnpm raises before any command prints: the fallback names that cause."""
    from hermes_cli import source_build

    root, _ = tui_source
    _touch_tui_entry(root / "ui-tui")

    def refuse():
        raise RuntimeError("pnpm: not installed and lazy installs are disabled")

    monkeypatch.setattr(source_build, "source_build_env", refuse)
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert "lazy installs are disabled" in capsys.readouterr().err
    assert "lazy installs are disabled" in caplog.text


# The TUI's first write (ui-tui/src/entry.tsx): clear the screen and the scrollback.
_TUI_CLEAR = "\\x1b[2J\\x1b[H\\x1b[3J"
_FALLBACK_LAUNCH = f"""
import fcntl, sys, termios
fcntl.ioctl(0, termios.TIOCSCTTY, 0)  # this new session's controlling terminal, as a shell has
from hermes_cli import main_tui_launch
def argv(tui_dir, tui_dev, notices=None):
    print("fallback notice", file=sys.stderr)
    if notices is not None:
        notices.append("fallback notice")
    return [sys.executable, "-c", "import sys; sys.stdout.write({_TUI_CLEAR!r} + 'TUI RAN')"], tui_dir.parent
main_tui_launch._make_tui_argv = argv
main_tui_launch._print_tui_exit_summary = lambda *_: None
main_tui_launch._NOTICE_WAIT_S = float(sys.argv[1])
if sys.argv[3]:  # a background job: its own process group, while this one keeps the terminal
    import os
    pid = os.fork()
    if pid:
        sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
    os.setpgid(0, 0)
main_tui_launch._launch_tui(query=sys.argv[2] or None)
"""


class _Terminal:
    """A pty that is the launch's controlling terminal, with the launch in its foreground process
    group (as a shell runs it) unless *background*."""

    def __init__(self, wait_s: float, *, background: bool = False, query: str = "", stderr=None):
        import pty
        import sys

        self.leader, follower = pty.openpty()
        self.child = subprocess.Popen(
            [sys.executable, "-c", _FALLBACK_LAUNCH, str(wait_s), query, "1" if background else ""],
            stdin=follower, stdout=follower,
            stderr=follower if stderr is None else stderr, close_fds=True, cwd=Path(__file__).resolve().parents[2],
            start_new_session=True,
        )
        os.close(follower)
        self.seen = b""

    def read_until(self, marker: bytes) -> float:
        import select
        import time

        started = time.monotonic()
        while marker not in self.seen:
            ready, _, _ = select.select([self.leader], [], [], 30)
            assert ready, f"timed out waiting for {marker!r}; got {self.seen!r}"
            try:
                chunk = os.read(self.leader, 4096)
            except OSError:  # EIO: the child closed the terminal
                chunk = b""
            assert chunk, f"terminal closed before {marker!r}; got {self.seen!r}"
            self.seen += chunk
        return time.monotonic() - started

    def close(self) -> None:
        self.child.kill()
        self.child.wait()
        os.close(self.leader)


@pytest.mark.platforms("posix")
def test_fallback_notice_survives_the_tui_screen_clear(monkeypatch):
    """In a terminal the warning waits for Enter before the TUI clears the screen, and is printed
    again once the TUI exits, below whatever it drew."""
    import sys

    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    terminal = _Terminal(60)
    try:
        terminal.read_until(b"Press Enter")
        assert b"TUI RAN" not in terminal.seen
        terminal.seen = b""
        os.write(terminal.leader, b"\r")
        terminal.read_until(b"TUI RAN")
        terminal.read_until(b"fallback notice")
        assert terminal.child.wait(timeout=30) == 0
    finally:
        terminal.close()
    assert b"fallback notice" in terminal.seen.partition(b"TUI RAN")[2]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", ["nobody answers", "background job", "query given", "stderr redirected"])
def test_fallback_notice_never_holds_an_unwatched_launch(monkeypatch, tmp_path, case):
    """A terminal nobody answers (tmux or expect automation) starts the TUI once the wait ends; a
    launch in the background, one that carries its query (-q), or one whose stderr (where the
    prompt goes) is redirected starts it at once. The warning still follows the TUI's output."""
    import sys

    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    log = tmp_path / "stderr.log"
    with log.open("wb") as stderr:
        terminal = _Terminal(2 if case == "nobody answers" else 60, background=case == "background job",
                             query="hello" if case == "query given" else "",
                             stderr=stderr if case == "stderr redirected" else None)
    try:
        # Nothing is ever typed: in "nobody answers" only the wait's own timeout starts the TUI.
        waited = terminal.read_until(b"TUI RAN")
        before, _, terminal.seen = terminal.seen.partition(b"TUI RAN")
        if case != "stderr redirected":
            terminal.read_until(b"fallback notice")  # printed again below the TUI's output
        assert terminal.child.wait(timeout=30) == 0
    finally:
        terminal.close()
    if case == "stderr redirected":
        before = log.read_bytes()
        assert before.count(b"fallback notice") == 2  # and printed again once the TUI exits
    # The prompt is printed only by a launch that waits; how long it waited is the child's clock,
    # not this process's, so only the unwatched launches' upper bound is asserted.
    if case == "nobody answers":
        assert b"Press Enter" in before
    else:
        assert b"Press Enter" not in before and waited < 30


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("termux", [False, True])
def test_fresh_bundle_does_not_prepare_or_compile(tui_source, monkeypatch, termux):
    root, acquired = tui_source
    _touch_tui_entry(root / "ui-tui")
    from tests.hermes_cli.test_source_build import stamp_product
    stamp_product(root, "tui", root / "ui-tui/dist")
    if termux:
        monkeypatch.setenv("TERMUX_VERSION", "test")
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert argv[-1] == str(cwd / "dist/entry.js")
    assert _events(root) == []
    assert acquired == []


@pytest.mark.platforms("posix")
def test_prebuilt_bundle_launch_does_not_touch_missing_source(tmp_path, monkeypatch):
    bundled = tmp_path / "bundle/entry.js"
    bundled.parent.mkdir()
    bundled.write_text("console.log('prebuilt')")
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setenv("HERMES_NODE", shutil.which("node"))
    monkeypatch.setattr(main_tui_launch, "_find_bundled_tui", lambda: bundled)
    argv, cwd = main_tui_launch._make_tui_argv(tmp_path / "missing-source", tui_dev=False)
    result = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "prebuilt"
    assert not (tmp_path / "missing-source").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("local_tsx", [False, True])
def test_dev_launch_builds_ink_before_running_source(tui_source, local_tsx):
    root, acquired = tui_source
    ink = root / "ui-tui/packages/hermes-ink"
    ink.mkdir(parents=True)
    (ink / "package.json").write_text(json.dumps({
        "name": "fixture-ink", "version": "1.0.0", "scripts": {"build": "node build.mjs"},
    }))
    (ink / "build.mjs").write_text("import { writeFileSync } from 'node:fs'; writeFileSync('built', 'ink');")
    workspace = root / "pnpm-workspace.yaml"
    workspace.write_text(workspace.read_text() + "  - ui-tui/packages/hermes-ink\n")
    subprocess.run([shutil.which("pnpm"), "install", "--lockfile-only", "--ignore-scripts", "--offline"], cwd=root, check=True,
                   env={**os.environ, **{f"pnpm_config_{name}_dir": str(root.parent / "pnpm" / name)
                                         for name in ("store", "cache", "state")}})
    tsx = root / "ui-tui/node_modules/.bin/tsx"
    if local_tsx:
        hook = root / "log.mjs"
        hook.write_text(hook.read_text() + "\nimport { mkdirSync, writeFileSync } from 'node:fs'; "
                        "mkdirSync('ui-tui/node_modules/.bin', {recursive: true}); "
                        "writeFileSync('ui-tui/node_modules/.bin/tsx', 'fixture tsx');\n")
    argv, cwd = main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=True)
    assert (ink / "built").read_text() == "ink"
    assert argv == ([str(tsx), "src/entry.tsx"] if local_tsx else [shutil.which("pnpm"), "start"])
    assert cwd == root / "ui-tui"
    assert acquired == ["pnpm"]
    assert [event["step"] for event in _events(root)] == ["deps"]


@pytest.mark.platforms("posix")
def test_runtime_node_comes_from_pm_without_legacy_repair(tmp_path, monkeypatch):
    import pm
    from pm.package import Runner

    managed = tmp_path / "bin/node"
    managed.parent.mkdir()
    managed.write_text("#!/bin/sh\nexit 0\n")
    managed.chmod(0o755)
    monkeypatch.delenv("HERMES_NODE", raising=False)
    acquired = []

    def acquire(name):
        acquired.append(name)
        return Runner(name, {"PATH": str(managed.parent)})

    monkeypatch.setattr(pm, "ensure", acquire)
    assert main_tui_launch._tui_node_bin("node") == str(managed)
    assert acquired == ["node"]


@pytest.mark.platforms("linux")
def test_tui_rebuild_preserves_the_prepared_desktop_and_web_union(tui_source):
    from hermes_cli.source_build import build_update_products

    root, acquired = tui_source
    build_update_products(root, desktop=True)
    before = _events(root)
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert _events(root) == before, "fresh launch must not prepare or rebuild"
    src = root / "ui-tui/src/entry.tsx"
    src.parent.mkdir()
    src.write_text("changed tui source")
    os.utime(root / "ui-tui/dist/entry.js", (1, 1))
    main_tui_launch._make_tui_argv(root / "ui-tui", tui_dev=False)
    assert _events(root) == [*before, {"step": "tui"}]
    assert acquired == ["pnpm", "pnpm"]
    assert (root / "web/node_modules").exists()
    assert (root / "apps/desktop/node_modules").exists()
