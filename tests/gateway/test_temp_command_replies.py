"""/temp on chat platforms: replies come from locales, and every branch answers.

The bug class this guards against is silent: a missing locale key makes ``t()``
return the key itself, so Telegram shows a user "gateway.temp.started" instead
of an explanation, and nothing raises. Assert on rendered text.
"""

import asyncio

import pytest

from agent.i18n import t
from gateway import slash_commands as sc


TEMP_KEYS = (
    "usage",
    "started",
    "already_on",
    "ended",
    "not_active",
    "status_on",
    "status_off",
    "reminder",
)


@pytest.mark.parametrize("key", TEMP_KEYS)
def test_every_temp_reply_resolves_to_real_copy(key):
    value = t(f"gateway.temp.{key}")
    assert value, f"gateway.temp.{key} is empty"
    # t() echoes the key back when it is missing from locales/en.yaml.
    assert not value.startswith("gateway.temp"), (
        f"gateway.temp.{key} is missing from locales/en.yaml -- Telegram would "
        f"render the raw key to the user"
    )


def test_temp_copy_uses_the_incognito_glyph_not_a_padlock():
    """A padlock claims "encrypted"; temporary chats promise "not recorded".

    The desktop badge and the CLI prompt both use the incognito glyph. The chat
    platforms shipped a padlock, which made one feature say two different
    things depending on where the user met it.
    """
    joined = "".join(t(f"gateway.temp.{k}") for k in TEMP_KEYS)
    assert "\U0001f512" not in joined, "padlock in /temp copy"
    assert "\U0001f575" in joined, "incognito glyph missing from /temp copy"


def test_usage_follows_the_house_format():
    """Matches /background: a Usage line, an Example, then what it does."""
    usage = t("gateway.temp.usage")
    assert usage.startswith("Usage: /temp"), usage
    assert "Example:" in usage, usage


class _Source:
    platform = "telegram"
    chat_id = "1"
    user_id = "1"


class _Event:
    """Minimal MessageEvent stand-in: only what _handle_temp_command touches."""

    def __init__(self, args=""):
        self.source = _Source()
        self._args = args

    def get_command_args(self):
        return self._args


class _Harness:
    """Drives the real handler with the session plumbing stubbed out."""

    def __init__(self, ephemeral=False):
        self.flag = ephemeral
        self.resets = 0

    _handle_temp_command = sc.GatewaySlashCommandsMixin._handle_temp_command

    def _session_key_for_source(self, source):
        return "k"

    def _session_is_ephemeral(self, key):
        return self.flag

    def _set_session_ephemeral(self, key, value):
        self.flag = value

    async def _handle_reset_command(self, event):
        self.resets += 1


def _run(harness, args=""):
    return asyncio.run(harness._handle_temp_command(_Event(args)))


def test_starting_a_temporary_chat_rotates_the_session_first():
    h = _Harness(ephemeral=False)
    reply = _run(h)
    assert reply == t("gateway.temp.started")
    assert h.flag is True
    # Without the rotation the preceding real conversation is left open and can
    # be flushed into the temporary session.
    assert h.resets == 1


def test_ending_a_temporary_chat_rotates_before_clearing_the_flag():
    h = _Harness(ephemeral=True)
    reply = _run(h, "off")
    assert reply == t("gateway.temp.ended")
    assert h.flag is False
    assert h.resets == 1


def test_status_reports_both_states():
    assert _run(_Harness(ephemeral=True), "status") == t("gateway.temp.status_on")
    assert _run(_Harness(ephemeral=False), "status") == t("gateway.temp.status_off")


def test_no_op_toggles_explain_rather_than_rotate():
    on = _Harness(ephemeral=True)
    assert _run(on, "") == t("gateway.temp.already_on")
    assert on.resets == 0, "a no-op toggle must not rotate the session"

    off = _Harness(ephemeral=False)
    assert _run(off, "off") == t("gateway.temp.not_active")
    assert off.resets == 0


def test_unknown_argument_shows_usage_instead_of_silently_starting():
    """"/temp halp" must not quietly discard what the user was doing."""
    h = _Harness(ephemeral=False)
    assert _run(h, "halp") == t("gateway.temp.usage")
    assert h.flag is False
    assert h.resets == 0


@pytest.mark.parametrize("arg", ["on", "yes", "1", "true", "start"])
def test_on_is_accepted_as_the_mirror_of_off(arg):
    h = _Harness(ephemeral=False)
    assert _run(h, arg) == t("gateway.temp.started")
    assert h.flag is True


# ---------------------------------------------------------------------------
# Periodic reminder cadence. Chat platforms have no persistent badge and the
# ephemeral flag survives restarts, so replies carry a reminder line: never
# right after the started-banner, every Nth reply thereafter, and immediately
# on the first reply a fresh gateway process sends into a temp chat it did
# not itself start (the restart case).
# ---------------------------------------------------------------------------
def _reminded(text: str) -> bool:
    return t("gateway.temp.reminder") in text


def test_reminder_cadence_after_temp_start():
    from types import SimpleNamespace

    from gateway.run import _TEMP_REMINDER_EVERY, _maybe_append_temp_reminder

    runner = SimpleNamespace()
    # /temp seeds the counter at 0 (see _set_session_ephemeral).
    runner._temp_reminder_turns = {"k": 0}

    outcomes = [
        _reminded(_maybe_append_temp_reminder(runner, "k", "reply"))
        for _ in range(_TEMP_REMINDER_EVERY * 2)
    ]
    # Replies 1..N-1 are clean; reply N and reply 2N carry the reminder.
    assert outcomes.count(True) == 2
    assert outcomes[_TEMP_REMINDER_EVERY - 1] is True
    assert outcomes[-1] is True
    assert not any(outcomes[: _TEMP_REMINDER_EVERY - 1]), (
        "the reply right after the started-banner must not be re-nagged"
    )


def test_reminder_fires_immediately_after_gateway_restart():
    from types import SimpleNamespace

    from gateway.run import _maybe_append_temp_reminder

    runner = SimpleNamespace()  # no counter dict at all — fresh process
    first = _maybe_append_temp_reminder(runner, "k", "reply")
    assert _reminded(first), (
        "a restarted gateway's first reply into a temp chat must restate "
        "that nothing is being saved"
    )
    second = _maybe_append_temp_reminder(runner, "k", "reply")
    assert not _reminded(second)


def test_reminder_appends_rather_than_replaces():
    from types import SimpleNamespace

    from gateway.run import _maybe_append_temp_reminder

    runner = SimpleNamespace()
    out = _maybe_append_temp_reminder(runner, "k", "the actual answer")
    assert out.startswith("the actual answer")
    assert _reminded(out)
