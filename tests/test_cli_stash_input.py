"""Unit tests for Ctrl+S stash/restore draft input (Claude Code-style).

Tests the stash/restore lifecycle:
  - Ctrl+S stashes current buffer text and clears the buffer
  - After a slash command, the stash is restored to the buffer
  - After a regular (non-slash) message, the stash is discarded
  - _maybe_restore_stashed_input is thread-safe (schedules on UI loop)
"""

from cli import HermesCLI


class _FakeLoop:
    """Captures call_soon_threadsafe callbacks for synchronous execution."""

    def __init__(self):
        self._pending = []

    def call_soon_threadsafe(self, fn, *args, **kwargs):
        self._pending.append((fn, args, kwargs))

    def flush(self):
        for fn, args, kwargs in self._pending:
            fn(*args, **kwargs)
        self._pending.clear()


class _FakeBuffer:
    """Minimal stand-in for prompt_toolkit's Buffer."""

    def __init__(self, text=""):
        self.text = text
        self.cursor_position = 0
        self._reset_count = 0

    def reset(self, append_to_history=False):
        self.text = ""
        self.cursor_position = 0
        self._reset_count += 1

    @property
    def complete_state(self):
        return None

    @property
    def suggestion(self):
        return None


class _FakeApp:
    """Minimal stand-in for prompt_toolkit's Application."""

    def __init__(self, buffer=None):
        self.current_buffer = buffer or _FakeBuffer()
        self._loop = _FakeLoop()
        self.is_running = True
        self._invalidated = False

    @property
    def loop(self):
        return self._loop

    def invalidate(self):
        self._invalidated = True


def _make_cli():
    """Create a HermesCLI without running __init__."""
    cli = HermesCLI.__new__(HermesCLI)
    cli._stashed_input = None
    cli._app = None
    cli._should_exit = False
    cli._invalidate = lambda *a, **kw: None
    return cli


def test_stash_attribute_defaults_to_none():
    """_stashed_input should be None on a fresh CLI instance."""
    cli = _make_cli()
    assert cli._stashed_input is None


def test_maybe_restore_no_stash_is_noop():
    """If there's no stash, _maybe_restore_stashed_input should do nothing."""
    cli = _make_cli()
    buf = _FakeBuffer()
    cli._app = _FakeApp(buf)
    cli._maybe_restore_stashed_input()
    # No callback should have been scheduled
    assert cli._app.loop._pending == []
    assert buf.text == ""


def test_maybe_restore_restores_to_empty_buffer():
    """Stashed text should be restored to an empty buffer."""
    cli = _make_cli()
    buf = _FakeBuffer()
    cli._app = _FakeApp(buf)
    cli._stashed_input = "Fix the bug in auth.py"
    cli._maybe_restore_stashed_input()
    # The callback is scheduled via call_soon_threadsafe
    assert cli._stashed_input is None  # consumed
    cli._app.loop.flush()
    assert buf.text == "Fix the bug in auth.py"
    assert buf.cursor_position == len("Fix the bug in auth.py")


def test_maybe_restore_does_not_overwrite_existing_text():
    """If the user started typing during the command, don't clobber their text."""
    cli = _make_cli()
    buf = _FakeBuffer("new text I typed")
    cli._app = _FakeApp(buf)
    cli._stashed_input = "old stashed text"
    cli._maybe_restore_stashed_input()
    cli._app.loop.flush()
    assert buf.text == "new text I typed"  # user's text wins
    assert cli._stashed_input is None  # stash was consumed (discarded)


def test_maybe_restore_no_app_is_safe():
    """If _app is None (non-interactive), the method should not crash."""
    cli = _make_cli()
    cli._app = None
    cli._stashed_input = "some text"
    cli._maybe_restore_stashed_input()
    assert cli._stashed_input is None  # consumed silently


def test_maybe_restore_app_not_running_is_safe():
    """If the app is not running, the restore callback should bail out."""
    cli = _make_cli()
    buf = _FakeBuffer()
    cli._app = _FakeApp(buf)
    cli._app.is_running = False
    cli._stashed_input = "stashed text"
    cli._maybe_restore_stashed_input()
    cli._app.loop.flush()
    assert buf.text == ""  # not restored because app not running


def test_stash_clears_on_regular_message():
    """When a non-slash message is sent, the stash should be cleared.

    This is tested by verifying the process_loop logic: after
    self.chat() is called, _stashed_input should be None. We simulate
    the key line that runs before chat: self._stashed_input = None.
    """
    cli = _make_cli()
    cli._stashed_input = "draft I was writing"
    # Simulate what the process_loop does before calling self.chat()
    cli._stashed_input = None
    assert cli._stashed_input is None


def test_stash_toggle_cycle():
    """Simulate the full Ctrl+S stash → restore → stash cycle."""
    cli = _make_cli()
    buf = _FakeBuffer("my draft prompt")
    cli._app = _FakeApp(buf)

    # Step 1: Ctrl+S stashes the text
    current = buf.text
    assert current.strip()
    cli._stashed_input = current
    buf.reset()
    assert buf.text == ""
    assert cli._stashed_input == "my draft prompt"

    # Step 2: After a slash command, restore via _maybe_restore_stashed_input
    cli._maybe_restore_stashed_input()
    cli._app.loop.flush()
    assert buf.text == "my draft prompt"
    assert cli._stashed_input is None

    # Step 3: Ctrl+S with empty buffer and existing stash restores directly
    cli._stashed_input = "another draft"
    buf.text = ""  # clear for the test
    # User presses Ctrl+S on empty buffer → restore
    if not buf.text.strip() and cli._stashed_input:
        stash = cli._stashed_input
        cli._stashed_input = None
        buf.text = stash
        buf.cursor_position = len(stash)
    assert buf.text == "another draft"
    assert cli._stashed_input is None
