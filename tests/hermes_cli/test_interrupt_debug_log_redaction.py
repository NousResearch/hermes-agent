"""#75461: interrupt_debug.log is a persistent on-disk file, so a credential in a
steered/queued user message must never reach it verbatim.

Behavior contract, not a source-text check: drive the real write path with a
Telegram-token-shaped payload and assert the bytes that land in the file.

Every case runs with global redaction switched OFF (``security.redact_secrets:
false``). That is the only configuration in which ``force=True`` at the write
sites changes anything -- with redaction enabled, ``redact_sensitive_text``
masks the token whether or not ``force`` is passed, so a test that only asserts
"the log holds no raw token" passes against a write site that never passes
``force`` at all. See ``test_write_sites_force_redaction_when_disabled``.
"""

import queue
import types
from pathlib import Path

import pytest

from agent import redact as redact_mod
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin

# Shape that triggered the original leak.
TOKEN = "1234567890:" + "A" * 35
# force=True alone does NOT mask userinfo passwords (_redact_url_credentials is
# opt-in), so this payload pins redact_url_credentials=True at the call sites.
USERINFO_URL = "https://alice:CorrectHorseBatteryStaple@example.com/p"


@pytest.fixture(autouse=True)
def _redaction_disabled(monkeypatch):
    """Model a user running with ``security.redact_secrets: false``.

    ``_redact_enabled()`` reads the launch-time global ``_REDACT_ENABLED`` only
    when no HERMES_HOME override is active; with an override it resolves the
    profile's own policy and caches it per home. Patch the global AND clear the
    per-home cache, then assert the switch really took effect -- otherwise the
    whole file degenerates into tests that pass with ``force=True`` deleted.
    """
    monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False)
    monkeypatch.setattr(redact_mod, "_REDACT_ENABLED_BY_HOME", {})
    assert redact_mod._redact_enabled() is False, (
        "redaction switch did not take effect -- these tests cannot detect a "
        "missing force=True without it"
    )
    # A token must survive the non-forced call, or the cases below prove nothing.
    assert TOKEN in redact_mod.redact_sensitive_text(TOKEN)


def _assert_token_masked(contents):
    assert "A" * 35 not in contents
    assert "1234567890:***" in contents


def _assert_userinfo_masked(contents):
    assert "CorrectHorseBatteryStaple" not in contents
    assert "alice:***@example.com" in contents


def _read_log(tmp_path):
    """Return the raw bytes the write paths actually produced.

    Reads the file directly and compares against booleans, so no display-layer
    redaction of a captured traceback can fake a pass. Also asserts the write
    site really wrote -- a path that swallows its own exceptions (``except
    Exception: pass``) would otherwise leave a missing file looking like a pass.
    """
    log = Path(tmp_path) / "interrupt_debug.log"
    assert log.exists(), f"write site never wrote {log}"
    return log.read_bytes().decode("utf-8", "replace")


def _run_busy_submit(tmp_path, monkeypatch, text):
    """Drive the legacy interrupt-queue branch of the busy-submit path."""
    import cli

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    mixin = CLITuiMixin.__new__(CLITuiMixin)
    mixin.busy_input_mode = "interrupt"
    mixin.agent = types.SimpleNamespace()  # no redirect() -> legacy queue branch
    mixin._agent_running = True
    mixin._interrupt_queue = types.SimpleNamespace(
        put=lambda payload: None,
    )
    CLITuiMixin._tui_enter_while_busy(mixin, text, [], {"text": text})
    return Path(tmp_path) / "interrupt_debug.log"


def test_queued_interrupt_message_is_redacted_on_disk(tmp_path, monkeypatch):
    _run_busy_submit(tmp_path, monkeypatch, f"my key is {TOKEN}")

    _assert_token_masked(_read_log(tmp_path))


def test_queued_interrupt_masks_url_userinfo_credentials(tmp_path, monkeypatch):
    _run_busy_submit(tmp_path, monkeypatch, f"see {USERINFO_URL}")

    _assert_userinfo_masked(_read_log(tmp_path))


def _fire_running_agent_interrupt(tmp_path, monkeypatch, text):
    """Drive the agent-thread side: the queued message arriving at a live turn.

    ``_chat_monitor_agent_thread`` polls ``self._interrupt_queue`` while the
    agent thread is alive, so the stub thread reports itself alive exactly once
    — long enough for the queued message to be dequeued, written, and for the
    loop to exit on the next poll.
    """
    import cli

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)

    pending = [text]

    class _InterruptQueue:
        def get(self, timeout=None):
            if pending:
                return pending.pop(0)
            raise queue.Empty()

    mixin = CLIChatTurnMixin.__new__(CLIChatTurnMixin)
    mixin._interrupt_queue = _InterruptQueue()
    mixin._pending_input = types.SimpleNamespace(put=lambda payload: pending.append(payload))
    mixin._clarify_state = None
    mixin._clarify_freetext = None
    mixin._voice_mode = False
    mixin.agent = types.SimpleNamespace(
        interrupt=lambda msg: None,
        _active_children=[],
        _interrupt_requested=True,
    )
    mixin._clear_active_overlays_for_interrupt = lambda: None

    turn = types.SimpleNamespace(stop_event=types.SimpleNamespace(set=lambda: None))
    entered = []

    def _is_alive():
        # True on the first poll (enters the loop), False afterwards (exits).
        if not entered:
            entered.append(True)
            return True
        return False

    CLIChatTurnMixin._chat_monitor_agent_thread(
        mixin, turn, types.SimpleNamespace(is_alive=_is_alive, join=lambda timeout=None: None)
    )
    return Path(tmp_path) / "interrupt_debug.log"


def test_running_agent_interrupt_is_redacted_on_disk(tmp_path, monkeypatch):
    _fire_running_agent_interrupt(tmp_path, monkeypatch, f"my key is {TOKEN}")

    _assert_token_masked(_read_log(tmp_path))


def test_running_agent_interrupt_masks_url_userinfo_credentials(tmp_path, monkeypatch):
    _fire_running_agent_interrupt(tmp_path, monkeypatch, f"see {USERINFO_URL}")

    _assert_userinfo_masked(_read_log(tmp_path))


def test_write_sites_force_redaction_when_disabled(tmp_path, monkeypatch):
    """Both sites are pinned at once, each with its own log file.

    The two call shapes differ, so one message exercises each: the TUI site
    ``str()``s a dict payload (the URL stays whole inside the repr), while the
    agent-thread site logs a plain string message. Separate home directories
    keep each site's line attributable instead of merged in one append-only
    file.
    """
    tui_home = tmp_path / "tui"
    thread_home = tmp_path / "thread"
    tui_home.mkdir()
    thread_home.mkdir()

    _run_busy_submit(tui_home, monkeypatch, f"key {TOKEN} url {USERINFO_URL}")
    body = _read_log(tui_home)
    assert TOKEN not in body, "TUI write site logged a raw token"
    assert "CorrectHorseBatteryStaple" not in body, (
        "TUI write site logged a raw userinfo password -- redact_url_credentials "
        "is not load-bearing there"
    )

    _fire_running_agent_interrupt(thread_home, monkeypatch, f"key {TOKEN} url {USERINFO_URL}")
    body2 = _read_log(thread_home)
    assert TOKEN not in body2, "agent-thread write site logged a raw token"
    assert "CorrectHorseBatteryStaple" not in body2, (
        "agent-thread write site logged a raw userinfo password -- "
        "redact_url_credentials is not load-bearing there"
    )


def test_hermes_home_override_resolves_its_own_policy(tmp_path, monkeypatch):
    """Guard the fixture above: under a context-local home override
    ``_redact_enabled`` resolves the profile's own policy and caches it, so a
    test that patches only the global can silently keep redaction ON.

    The override is a ContextVar (``hermes_constants._HERMES_HOME_OVERRIDE``),
    not ``os.environ["HERMES_HOME"]`` -- setting the env var alone leaves
    ``get_hermes_home_override()`` at None. This pins both facts so a future
    change cannot turn every case in this file back into a benign test.
    """
    import hermes_constants

    # Two profiles: one whose .env leaves redaction on (the control), one whose
    # .env turns it off. Reading the profile's own file is the whole point, so
    # the control must use a home that does NOT contain the disabling .env.
    on_home = tmp_path / "on"
    off_home = tmp_path / "off"
    on_home.mkdir()
    off_home.mkdir()
    (off_home / ".env").write_text("HERMES_REDACT_SECRETS=false\n", encoding="utf-8")

    token = hermes_constants.set_hermes_home_override(str(on_home))
    try:
        assert hermes_constants.get_hermes_home_override() == str(on_home)

        # Control: the global is on and this profile does not opt out -> on.
        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", True)
        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED_BY_HOME", {})
        assert redact_mod._redact_enabled() is True

        # Same process, same global: only the profile's own .env differs, and
        # it wins -- so patching the global alone cannot model a user who set
        # security.redact_secrets: false under a home override.
        hermes_constants.set_hermes_home_override(str(off_home))
        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED_BY_HOME", {})
        assert redact_mod._redact_enabled() is False, (
            "the per-home override branch no longer honours the profile policy; "
            "the autouse fixture may be patching a switch that is not consulted"
        )
    finally:
        hermes_constants.reset_hermes_home_override(token)
