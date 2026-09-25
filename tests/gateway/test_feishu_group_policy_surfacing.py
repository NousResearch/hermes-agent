"""Operator-facing surfacing of the empty-allowlist default deny in Feishu group admission.

Secret scoping is deliberate: a secondary multiplex profile never inherits the launch
profile's FEISHU_GROUP_POLICY. What must not happen is the resulting deny being visible only
at DEBUG (#111420) — the first such drop is a WARNING naming the keys to set.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from tests.gateway.feishu_helpers import (
    install_dedup_state,
    make_adapter_skeleton,
    make_message,
    make_sender,
)

_LOGGER = "plugins.platforms.feishu.adapter"


def _group_event(message_id: str, chat_id: str = "oc_chat") -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            sender=make_sender(open_id="ou_human"),
            message=make_message(message_id=message_id, chat_type="group", chat_id=chat_id),
        )
    )


def _drops(adapter, caplog, *events) -> list[logging.LogRecord]:
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for ev in events:
            asyncio.run(adapter._handle_message_event_data(ev))
    return [r for r in caplog.records if "group" in r.getMessage().lower() and "dropp" in r.getMessage().lower()]


def test_empty_allowlist_default_deny_warns_once_with_config_keys(caplog):
    adapter = make_adapter_skeleton(group_policy="allowlist")
    install_dedup_state(adapter)

    records = _drops(adapter, caplog, _group_event("om_1"), _group_event("om_2"), _group_event("om_3"))

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in records]
    text = warnings[0].getMessage()
    assert "FEISHU_ALLOWED_USERS" in text and "FEISHU_GROUP_POLICY" in text and "oc_chat" in text
    # Later drops stay at DEBUG — the warning is a one-shot diagnostic, not per-message noise.
    assert sum(r.levelno == logging.DEBUG for r in records) == 3


def test_configured_group_deny_stays_at_debug(caplog):
    """An operator-chosen deny (populated allowlist, or a per-chat rule) is not the misconfiguration."""
    adapter = make_adapter_skeleton(group_policy="allowlist")
    install_dedup_state(adapter)
    adapter._allowed_group_users = frozenset({"ou_someone_else"})

    ruled = make_adapter_skeleton(group_policy="allowlist")
    install_dedup_state(ruled)
    from plugins.platforms.feishu.adapter import FeishuGroupRule
    ruled._group_rules = {"oc_chat": FeishuGroupRule(policy="disabled", allowlist=set(), blacklist=set(), require_mention=None)}

    records = _drops(adapter, caplog, _group_event("om_a")) + _drops(ruled, caplog, _group_event("om_b"))

    assert records and all(r.levelno == logging.DEBUG for r in records), [r.getMessage() for r in records]
