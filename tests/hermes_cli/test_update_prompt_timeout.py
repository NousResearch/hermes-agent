"""`hermes update` must not hang forever on a prompt nobody can answer (#92303).

The reported shape is a Windows Scheduled Task launched with
``-WindowStyle Hidden``. That process gets a **real** console, so
``sys.stdin.isatty()`` is ``True`` and every isatty-based guard in
``update_cmd`` classifies the run as interactive, and the handle is open, so
``input()`` never raises ``EOFError`` either. It simply blocks. The report has a
44 minute hang ending in exit code ``0x40010004``, an external kill rather than
an exit.

So the prompts are bounded instead of predicted. A prompt that goes unanswered
takes its documented default, says so, and latches the run as unattended, which
is a stronger signal than any isatty check can produce because it is an
observation rather than an inference.

Every test here pins one of three things:

* the bound exists, and the default it takes is the safe one
* the answered paths are **unchanged**, including the exception paths that
  predate this fix
* the latch, which is both the generalisation the reporter asked for and the
  thing that stops a parked reader thread from stealing the next prompt's answer
"""

import threading

import pytest

from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _clear_latch():
    """The latch is module state, so a leaked one would silently pass tests."""
    update_cmd._reset_prompt_interactivity()
    yield
    update_cmd._reset_prompt_interactivity()


class _Blocker:
    """A reader that never returns, like a hidden console nobody types into."""

    def __init__(self):
        self.released = threading.Event()
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        # The stash prompt calls bare `input()`; the upstream prompt calls
        # `input("Add official repo...")`. Same stub serves both.
        self.calls += 1
        self.released.wait(30)  # bounded so a broken test cannot wedge CI
        return "y"

    def release(self):
        self.released.set()


@pytest.fixture
def blocker():
    b = _Blocker()
    try:
        yield b
    finally:
        b.release()


# ── the bound itself ───────────────────────────────────────────────────


class TestBoundedRead:
    def test_an_unanswerable_prompt_gives_up(self, blocker):
        response, timed_out = update_cmd._read_line_with_timeout(
            "n", timeout=0.05, read_fn=blocker
        )

        assert timed_out is True
        assert response == "n", (
            "the caller must get the documented default, not an empty string "
            "that a downstream `in {'', 'y', 'yes'}` check would read as YES"
        )

    def test_an_answered_prompt_is_not_disturbed(self):
        response, timed_out = update_cmd._read_line_with_timeout(
            "n", timeout=5, read_fn=lambda: "  YeS  "
        )

        assert timed_out is False
        assert response == "  YeS  ", (
            "the helper must not normalise; callers still own .strip().lower()"
        )

    @pytest.mark.parametrize("boom", [EOFError, UnicodeDecodeError, KeyboardInterrupt])
    def test_the_pre_existing_exception_paths_still_take_the_default(self, boom):
        """These three were already handled before #92303 and must stay handled.

        They are also not timeouts: a closed stdin answers immediately, so
        latching the run unattended on them would be wrong.
        """
        if boom is UnicodeDecodeError:
            exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        else:
            exc = boom()

        def _raise():
            raise exc

        response, timed_out = update_cmd._read_line_with_timeout(
            "n", timeout=5, read_fn=_raise
        )

        assert response == "n"
        assert timed_out is False
        assert update_cmd._prompt_proven_unattended is False

    def test_a_non_positive_timeout_restores_the_blocking_read(self):
        """The escape hatch, so the old behaviour is reachable without a revert."""
        response, timed_out = update_cmd._read_line_with_timeout(
            "n", timeout=0, read_fn=lambda: "y"
        )

        assert (response, timed_out) == ("y", False)

    def test_the_reader_thread_cannot_keep_the_process_alive(self, blocker):
        """The fix must not become the bug.

        On timeout the reader is still parked on stdin for the life of the
        process. A non-daemon thread there would block interpreter shutdown,
        which is the same unbounded wait wearing a different hat.
        """
        before = set(threading.enumerate())
        update_cmd._read_line_with_timeout("n", timeout=0.05, read_fn=blocker)
        leaked = [t for t in threading.enumerate() if t not in before]

        assert leaked, "expected the parked reader to still be running"
        assert all(t.daemon for t in leaked), (
            f"reader thread must be a daemon; got {[(t.name, t.daemon) for t in leaked]}"
        )


# ── the latch ──────────────────────────────────────────────────────────


class TestUnattendedLatch:
    def test_a_timeout_proves_the_run_unattended(self, blocker):
        assert update_cmd._prompt_proven_unattended is False
        update_cmd._read_line_with_timeout("n", timeout=0.05, read_fn=blocker)
        assert update_cmd._prompt_proven_unattended is True

    def test_a_later_prompt_does_not_ask_again(self, blocker):
        update_cmd._read_line_with_timeout("n", timeout=0.05, read_fn=blocker)
        first_calls = blocker.calls

        second = _Blocker()
        try:
            response, timed_out = update_cmd._read_line_with_timeout(
                "n", timeout=30, read_fn=second
            )
        finally:
            second.release()

        assert (response, timed_out) == ("n", True)
        assert second.calls == 0, (
            "a second reader on the same stdin can have its answer swallowed by "
            "the first, still-parked one; the latch exists to prevent that"
        )
        assert blocker.calls == first_calls
        assert timed_out is True, "and it must return fast, not after 30s"

    def test_an_answered_prompt_never_latches(self):
        update_cmd._read_line_with_timeout("n", timeout=5, read_fn=lambda: "y")
        assert update_cmd._prompt_proven_unattended is False


# ── the reported call site ─────────────────────────────────────────────


class TestStashRestorePrompt:
    """`_restore_stashed_changes` is where the reporter's update actually hung."""

    def test_the_reported_hang_now_ends(self, tmp_path, capsys, blocker, monkeypatch):
        monkeypatch.setattr(update_cmd, "_UPDATE_PROMPT_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr("builtins.input", blocker)

        result = update_cmd._restore_stashed_changes(
            ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
        )

        assert result is False, (
            "unanswered must mean skip-restore: it is the only default that "
            "cannot lose work, since the stash stays on disk"
        )
        out = capsys.readouterr().out
        assert "unattended" in out, (
            f"an unattended log ending at a bare prompt is unreadable; got {out!r}"
        )
        assert "--yes" in out, "the log must say how to avoid the prompt next time"
        assert "git stash apply stash@{0}" in out

    def test_a_human_answering_yes_is_unaffected(self, tmp_path, monkeypatch):
        """The guard must be invisible to everyone it was not built for."""
        monkeypatch.setattr("builtins.input", lambda: "y")
        calls = []
        monkeypatch.setattr(
            update_cmd.subprocess,
            "run",
            lambda *a, **kw: calls.append(a) or _ok(),
        )

        update_cmd._restore_stashed_changes(
            ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
        )

        assert calls, "an answered 'y' must still reach `git stash apply`"

    def test_a_human_answering_no_is_unaffected(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda: "n")

        result = update_cmd._restore_stashed_changes(
            ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
        )

        assert result is False
        out = capsys.readouterr().out
        assert "Skipped restoring local changes" in out
        assert "unattended" not in out, (
            "a deliberate 'no' is not an unattended run and must not be "
            "reported as one"
        )

    def test_the_gateway_input_fn_path_is_untouched(self, tmp_path, monkeypatch):
        """Gateway mode answers over file-based IPC and never touches stdin."""
        monkeypatch.setattr(update_cmd, "_UPDATE_PROMPT_TIMEOUT_SECONDS", 0.05)
        seen = []
        monkeypatch.setattr(
            update_cmd.subprocess, "run", lambda *a, **kw: _ok(),
        )

        update_cmd._restore_stashed_changes(
            ["git"],
            tmp_path,
            "stash@{0}",
            prompt_user=True,
            input_fn=lambda prompt, default: seen.append(prompt) or "y",
        )

        assert seen, "input_fn must still be consulted"
        assert update_cmd._prompt_proven_unattended is False


# ── the site that is reached even earlier ──────────────────────────────


class TestUpstreamRemotePrompt:
    """This one runs BEFORE the reported prompt, so it can hang first."""

    def _run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(update_cmd, "_has_upstream_remote", lambda *a, **kw: False)
        monkeypatch.setattr(update_cmd, "_should_skip_upstream_prompt", lambda: False)
        added = []
        monkeypatch.setattr(
            update_cmd, "_add_upstream_remote", lambda *a, **kw: added.append(1) or True
        )
        update_cmd._sync_with_upstream_if_needed(["git"], tmp_path)
        return added

    def test_it_stops_waiting_too(self, tmp_path, capsys, blocker, monkeypatch):
        monkeypatch.setattr(update_cmd, "_UPDATE_PROMPT_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr("builtins.input", blocker)

        added = self._run(monkeypatch, tmp_path)

        assert added == [], "unanswered must not silently add a remote"
        assert "unattended" in capsys.readouterr().out

    def test_eof_keeps_the_blank_line_it_always_printed(
        self, tmp_path, capsys, monkeypatch
    ):
        """Guards a spacing regression the obvious refactor introduces.

        The original printed a blank line only on the exception path. Routing
        this through the shared helper makes it easy to print it on the
        success path instead, which is backwards and invisible to any
        assertion about the return value.
        """
        def _raise(*_args, **_kwargs):
            raise EOFError()

        monkeypatch.setattr("builtins.input", _raise)
        self._run(monkeypatch, tmp_path)
        eof_out = capsys.readouterr().out

        monkeypatch.setattr("builtins.input", lambda _p: "n")
        self._run(monkeypatch, tmp_path)
        answered_out = capsys.readouterr().out

        # Counting "\n\n" does not work: str.count does not overlap, so a
        # triple newline scores the same 1 as a double. Compare exactly.
        assert eof_out != answered_out, "the two paths must not print the same"
        assert eof_out.replace("\n\n\n", "\n\n", 1) == answered_out, (
            "the ONLY difference between the two paths should be one blank "
            f"line on the exception path.\n  eof: {eof_out!r}\n  ans: {answered_out!r}"
        )


class _ok:
    """Stand-in for a successful ``subprocess.run`` result."""

    returncode = 0
    stdout = ""
    stderr = ""
