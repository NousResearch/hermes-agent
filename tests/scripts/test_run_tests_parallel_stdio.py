"""The runner's status glyphs must not crash narrow console encodings.

On native Windows, piped or legacy-console stdio defaults to cp1252, which
cannot encode the runner's ✓/✗ progress glyphs — before the fix, the first
per-file status line killed the whole run with UnicodeEncodeError. The
failure depends only on the stream's encoding, so these tests pin it on
every OS by building a cp1252 stream explicitly.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from subprocess import TimeoutExpired
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("cleanup_verified", [True, False])
def test_child_failure_reports_status_or_unverified_cleanup(
    tmp_path: Path, monkeypatch, cleanup_verified: bool
) -> None:
    mod = _load_runner()
    failure = mod.subprocess.TimeoutExpired(["synthetic-child"], 30)
    uncertain = mod._OwnedProcessTerminationUnverified(123, failure)
    roots: list[Path] = []

    def communicate(*args, **kwargs):
        if not cleanup_verified:
            raise uncertain
        return "", 23

    def create_child(*args, **kwargs):
        root = Path(kwargs["env"]["TMPDIR"])
        roots.append(root)
        (root / "output").write_text("owned output", encoding="utf-8")
        return SimpleNamespace(
            pid=123,
            returncode=23 if cleanup_verified else None,
            communicate=communicate,
        )

    monkeypatch.setattr(mod.subprocess, "Popen", create_child)
    monkeypatch.setattr(mod, "_communicate_owned_posix", communicate)
    monkeypatch.setattr(mod, "_kill_windows_tree", lambda child: None)
    monkeypatch.setattr(mod, "_runner_scratch_root", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: 0))
    monkeypatch.setattr(
        mod,
        "signal",
        SimpleNamespace(SIGCHLD=object(), SIG_DFL=0, getsignal=lambda _: 0),
    )
    file = tmp_path / "test.py"
    if cleanup_verified:
        result = mod._run_one_file_once(file, [], tmp_path, 30)
        assert (result, roots[0].exists()) == (
            (
                file,
                23,
                f"pytest child exited 23 (0x00000017) with no output: {file}\n",
                {},
                0,
            ),
            False,
        )
        return

    with pytest.raises(mod._OwnedProcessTerminationUnverified) as raised:
        mod._run_one_file_once(file, [], tmp_path, 30)
    assert (
        raised.value.original_error,
        raised.value.cleanup_errors,
        raised.value.args,
        (roots[0] / "output").read_text(encoding="utf-8"),
    ) == (
        failure,
        (),
        (
            f"Could not verify termination of child 123 and its process group; temporary outputs retained at {roots[0]}",
        ),
        "owned output",
    )


class _OwnedChildProbe:
    def __init__(self, scenario: str) -> None:
        self.scenario = (
            "wait_failure" if scenario == "ambient_wait_failure" else scenario
        )
        self.pid = 123
        self.args = ["synthetic-child"]
        self.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
        self.clock: float = 0.0
        self.cleanup_deadline: float | None = None
        self.events: list[str] = []
        self.chunks = iter([b"complete \xe2", b"\x82\xac\r\noutput\r", b""])
        self.rows: dict[int, SimpleNamespace] = {}
        self.reaped = False
        self.signalled = False

    def wait(self, timeout: float) -> int:
        assert self.signalled and not self.reaped
        assert self.cleanup_deadline is not None
        assert timeout <= max(0, self.cleanup_deadline - self.clock)
        self.events.append("wait")
        if self.scenario == "wait_failure":
            self.clock += timeout
            raise TimeoutExpired(self.args, timeout)
        self.reaped = True
        return -9 if self.scenario == "timeout" else 7

    def waitid(self, kind: str, pid: int, flags: int) -> SimpleNamespace | None:
        assert (kind, pid, flags, self.reaped) == ("owned-pid", self.pid, 7, False)
        if self.scenario == "lost_identity":
            raise ChildProcessError("Synthetic child ownership lost")
        if self.scenario == "timeout" and not self.signalled:
            return None
        return SimpleNamespace(
            si_pid=self.pid,
            si_code="killed" if self.scenario == "timeout" else "exited",
            si_status=9 if self.scenario == "timeout" else 7,
        )

    def killpg(self, pid: int, sig: str) -> None:
        assert (pid, sig, self.reaped, self.signalled) == (
            self.pid,
            "kill",
            False,
            False,
        )
        self.events.append("signal")
        self.signalled = True
        self.cleanup_deadline = self.clock + 10

    def read(self, fd: int, count: int) -> bytes:
        assert (fd, count) == (11, 32768)
        return next(self.chunks)

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def select(self, timeout: float) -> list[tuple[SimpleNamespace, int]]:
        self.clock += timeout
        if self.scenario == "drain_deadline" and self.signalled:
            assert self.cleanup_deadline is not None
            self.clock = self.cleanup_deadline
            return []
        return [(self.rows[11], 1)]

    def __enter__(self) -> "_OwnedChildProbe":
        return self

    def __exit__(self, *args) -> None:
        return None

    def register(self, stream: io.TextIOWrapper, events: int) -> None:
        self.rows[11] = SimpleNamespace(fd=11, fileobj=stream)

    def unregister(self, stream: io.TextIOWrapper) -> None:
        self.rows.clear()

    def get_map(self) -> dict[int, SimpleNamespace]:
        return self.rows


@pytest.mark.parametrize(
    "scenario",
    [
        "complete",
        "timeout",
        "lost_identity",
        "drain_deadline",
        "wait_failure",
        "ambient_wait_failure",
    ],
)
def test_owned_child_cleanup_precedes_reap(monkeypatch, scenario: str) -> None:
    mod = _load_runner()
    child = _OwnedChildProbe(scenario)
    monkeypatch.setattr(
        mod,
        "os",
        SimpleNamespace(
            P_PID="owned-pid",
            WEXITED=1,
            WNOHANG=2,
            WNOWAIT=4,
            CLD_EXITED="exited",
            CLD_KILLED="killed",
            CLD_DUMPED="dumped",
            waitid=child.waitid,
            killpg=child.killpg,
            read=child.read,
        ),
    )
    monkeypatch.setattr(mod, "signal", SimpleNamespace(SIGKILL="kill"))
    monkeypatch.setattr(
        mod, "selectors", SimpleNamespace(DefaultSelector=lambda: child, EVENT_READ=1)
    )
    monkeypatch.setattr(
        mod, "time", SimpleNamespace(monotonic=lambda: child.clock, sleep=child.sleep)
    )

    def communicate() -> tuple[str, int]:
        if scenario != "ambient_wait_failure":
            return mod._communicate_owned_posix(child, 2)
        try:
            raise ValueError("Unrelated caller exception")
        except ValueError:
            return mod._communicate_owned_posix(child, 2)

    if scenario in (
        "lost_identity",
        "drain_deadline",
        "wait_failure",
        "ambient_wait_failure",
    ):
        with pytest.raises(mod._OwnedProcessTerminationUnverified) as raised:
            communicate()
        original = ChildProcessError if scenario == "lost_identity" else TimeoutExpired
        expected_events = [] if scenario == "lost_identity" else ["signal", "wait"]
        assert (
            type(raised.value.original_error),
            child.events,
            child.stdout.closed,
            child.reaped,
        ) == (original, expected_events, True, scenario == "drain_deadline")
        return

    result = communicate()
    expected_output = "complete €\noutput\n"
    expected_code = 7
    if scenario == "timeout":
        expected_output = "(2s exceeded; process tree SIGKILL'd)\n" + expected_output
        expected_code = 124
    assert (result, child.events, child.stdout.closed, child.reaped) == (
        (expected_output, expected_code),
        ["signal", "wait"],
        True,
        True,
    )


def _cp1252_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding="cp1252", errors="strict"), raw


def test_glyph_safe_stdio_survives_cp1252(monkeypatch) -> None:
    mod = _load_runner()
    stream, raw = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)

    mod._make_stdio_glyph_safe()
    print("✓ tests/foo.py (3 tests, 1.2s) ✗")
    sys.stdout.flush()

    out = raw.getvalue()
    assert "✓".encode() in out, "stream should now carry UTF-8 glyphs"
    assert b"tests/foo.py (3 tests, 1.2s)" in out, "line content must survive"


def test_glyph_safe_stdio_noop_without_reconfigure(monkeypatch) -> None:
    # Streams without .reconfigure (e.g. pytest's capture buffers, plain
    # StringIO) must pass through untouched instead of raising.
    mod = _load_runner()
    plain = io.StringIO()
    monkeypatch.setattr(sys, "stdout", plain)
    monkeypatch.setattr(sys, "stderr", plain)

    mod._make_stdio_glyph_safe()
    print("✓ still fine")

    assert "✓ still fine" in plain.getvalue()


@pytest.mark.platforms("macos")
@pytest.mark.live_system_guard_bypass
def test_exited_group_leader_is_cleaned_before_reap() -> None:
    mod = _load_runner()
    child = mod.subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import signal; signal.alarm(15); print('done'); raise SystemExit(7)",
        ],
        stdin=mod.subprocess.DEVNULL,
        stdout=mod.subprocess.PIPE,
        stderr=mod.subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        start_new_session=True,
    )
    try:
        state = mod.os.waitid(mod.os.P_PID, child.pid, mod.os.WEXITED | mod.os.WNOWAIT)
        assert state.si_pid == child.pid
        result = mod._communicate_owned_posix(child, 5)
        assert (result, child.returncode, child.stdout.closed) == (
            ("done\n", 7),
            7,
            True,
        )
    finally:
        child.wait(timeout=20)
        child.stdout.close()
