"""One mailbox, several verified sender identities.

Gmail "send mail as", Microsoft 365 shared mailboxes and catch-all domains all put several
addresses on one physical mailbox. The adapter keeps the mailbox apart from the identities it
may send as:

    EMAIL_ADDRESS      the mailbox it watches and logs in with
    EMAIL_FROM_ADDRESS the default visible sender (defaults to EMAIL_ADDRESS)
    EMAIL_FROM_ALIASES every other identity this account may send as

A reply goes out from the identity the sender wrote to; an alias that is not configured is
refused (fail-closed), never sent. With none of the extra variables set the behaviour is
byte-for-byte the pre-alias behaviour.
"""

from __future__ import annotations

import email as email_lib
import os
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig


MAILBOX = "soporte@example.com"
ALIAS = "hr@example.com"
OTHER_ALIAS = "billing@example.com"

BASE_ENV = {
    "EMAIL_ADDRESS": MAILBOX,
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.example.com",
    "EMAIL_SMTP_HOST": "smtp.example.com",
}


def make_adapter(**env):
    """Build an adapter with a clean EMAIL_* environment plus the given overrides."""
    from plugins.platforms.email.adapter import EmailAdapter

    scrubbed = {k: "" for k in ("EMAIL_FROM_ADDRESS", "EMAIL_FROM_ALIASES")}
    with patch.dict(os.environ, {**scrubbed, **BASE_ENV, **env}, clear=False):
        return EmailAdapter(PlatformConfig(enabled=True))


def make_message(**headers) -> email_lib.message.Message:
    msg = email_lib.message.EmailMessage()
    for key, value in headers.items():
        msg[key.replace("_", "-")] = value
    return msg


# ── defaults: unconfigured deployments keep the old behaviour ──────────────


def test_defaults_collapse_to_email_address():
    adapter = make_adapter()
    assert adapter._from_address == MAILBOX
    assert adapter._from_aliases == {MAILBOX: MAILBOX}
    assert adapter._resolve_from_address("someone@else.com") == MAILBOX
    assert adapter._message_id_domain() == "example.com"


def test_from_address_without_at_falls_back_to_localhost():
    adapter = make_adapter(EMAIL_ADDRESS="not-an-address")
    assert adapter._message_id_domain() == "localhost"


# ── configuration parsing ──────────────────────────────────────────────────


def test_default_sender_and_aliases_are_independent_of_the_mailbox():
    adapter = make_adapter(EMAIL_FROM_ADDRESS=ALIAS, EMAIL_FROM_ALIASES=f"{OTHER_ALIAS}, {MAILBOX}")
    assert adapter._from_address == ALIAS
    assert set(adapter._from_aliases) == {ALIAS, OTHER_ALIAS, MAILBOX}


def test_alias_parsing_ignores_blanks_and_non_addresses():
    from plugins.platforms.email.adapter import _parse_from_aliases

    parsed = _parse_from_aliases(f" {ALIAS} , , not-an-address ,{OTHER_ALIAS}", MAILBOX)
    assert set(parsed) == {ALIAS, OTHER_ALIAS, MAILBOX}
    assert parsed[ALIAS] == ALIAS  # the configured spelling is what goes on the wire


def test_alias_parsing_accepts_a_config_yaml_list():
    from plugins.platforms.email.adapter import _parse_from_aliases

    assert set(_parse_from_aliases([ALIAS, OTHER_ALIAS], MAILBOX)) == {ALIAS, OTHER_ALIAS, MAILBOX}


def test_alias_lookup_is_case_insensitive_but_preserves_the_configured_spelling():
    adapter = make_adapter(EMAIL_FROM_ALIASES="HR@Example.com")
    assert adapter._resolve_from_address("x@y.com", "hr@EXAMPLE.COM") == "HR@Example.com"


# ── reply follows the identity the sender addressed ────────────────────────


@pytest.mark.parametrize("header", ["To", "Cc", "Delivered-To", "X-Original-To"])
def test_addressed_alias_is_read_from_every_recipient_header(header):
    adapter = make_adapter(EMAIL_FROM_ALIASES=ALIAS)
    msg = make_message(**{header.replace("-", "_"): f"Support <{ALIAS}>"})
    assert adapter._addressed_alias(msg) == ALIAS


def test_unknown_recipient_falls_back_to_the_default_sender():
    adapter = make_adapter(EMAIL_FROM_ADDRESS=ALIAS, EMAIL_FROM_ALIASES=OTHER_ALIAS)
    assert adapter._addressed_alias(make_message(To="stranger@nowhere.com")) == ALIAS


def test_reply_goes_out_from_the_alias_the_thread_arrived_on():
    adapter = make_adapter(EMAIL_FROM_ALIASES=ALIAS)
    adapter._thread_context["client@corp.com"] = {
        "subject": "Invoice", "message_id": "<abc@corp.com>", "reply_from": ALIAS}
    msg, msg_id, _ = adapter._new_reply("client@corp.com", "body")
    assert msg["From"] == ALIAS
    assert msg_id.endswith("@example.com>")


def test_reply_without_thread_context_uses_the_default_sender():
    adapter = make_adapter(EMAIL_FROM_ADDRESS=ALIAS, EMAIL_FROM_ALIASES=OTHER_ALIAS)
    msg, _, _ = adapter._new_reply("client@corp.com", "body")
    assert msg["From"] == ALIAS


# ── fail-closed on an unconfigured alias ───────────────────────────────────


def test_requested_alias_is_honoured_when_configured():
    adapter = make_adapter(EMAIL_FROM_ALIASES=f"{ALIAS},{OTHER_ALIAS}")
    assert adapter._resolve_from_address("client@corp.com", OTHER_ALIAS) == OTHER_ALIAS


def test_requested_alias_outside_the_configured_set_is_refused():
    adapter = make_adapter(EMAIL_FROM_ADDRESS=ALIAS, EMAIL_FROM_ALIASES=ALIAS)
    assert adapter._resolve_from_address("client@corp.com", "ceo@victim.com") == ALIAS
    msg, _, _ = adapter._new_reply("client@corp.com", "body", requested_from="ceo@victim.com")
    assert msg["From"] == ALIAS


def test_refused_alias_does_not_win_over_the_thread_alias():
    adapter = make_adapter(EMAIL_FROM_ALIASES=f"{ALIAS},{OTHER_ALIAS}")
    adapter._thread_context["client@corp.com"] = {"reply_from": OTHER_ALIAS}
    assert adapter._resolve_from_address("client@corp.com", "ceo@victim.com") == OTHER_ALIAS


# ── self-detection covers every identity ───────────────────────────────────


@pytest.mark.parametrize("sender", [MAILBOX, ALIAS, OTHER_ALIAS])
def test_mail_from_any_of_our_identities_is_not_answered(sender):
    adapter = make_adapter(EMAIL_FROM_ADDRESS=ALIAS, EMAIL_FROM_ALIASES=OTHER_ALIAS)
    assert adapter._sender_accepted(sender, {}) is False


def test_a_real_sender_is_still_accepted():
    adapter = make_adapter(EMAIL_FROM_ALIASES=ALIAS, EMAIL_ALLOWED_USERS="client@corp.com")
    with patch.dict(os.environ, {"EMAIL_ALLOWED_USERS": "client@corp.com"}, clear=False):
        assert adapter._sender_accepted("client@corp.com", {"sender_authenticated": True}) is True


# ── the alias changes the visible sender, not the credential ───────────────


def test_smtp_still_logs_in_with_the_mailbox_address():
    adapter = make_adapter(EMAIL_FROM_ADDRESS=ALIAS)
    sent = {}

    class FakeSMTP:
        def login(self, user, password):
            sent["user"] = user

        def send_message(self, msg):
            sent["from"] = msg["From"]

        def quit(self):
            sent["quit"] = True

    with patch.object(adapter, "_connect_smtp", return_value=FakeSMTP()):
        adapter._send_email("client@corp.com", "hello")

    assert sent["user"] == MAILBOX
    assert sent["from"] == ALIAS
    assert sent["quit"] is True
