"""Tests for the opt-in Feishu reply-card path (``platforms.feishu.reply_card``).

Behavior contracts:
- ``FEISHU_REPLY_CARD`` env beats YAML ``extra["reply_card"]`` — the repo's
  documented env-over-YAML precedence for per-profile settings
  (``gateway/platforms/_shared.py::extra_or_secret``).
- Truthy parsing accepts ``true``/``1``/``yes``/``on``; the switch defaults off.
- ``_build_outbound_payload`` returns ``interactive`` Card 2.0 payloads when the
  switch is on (``send()`` and ``edit_message()`` share this builder) and keeps
  the legacy ``post``/``text`` behavior when off.
"""

from __future__ import annotations

import json

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


# --- Config precedence -------------------------------------------------------


@pytest.mark.parametrize(
    "env_value, extra, expected",
    [
        (None, {}, False),  # default off
        ("true", {}, True),  # env on
        ("false", {"reply_card": True}, False),  # env beats YAML
        (None, {"reply_card": True}, True),  # YAML on, env unset
        ("yes", {}, True),  # broadened truthy set
        ("on", {"reply_card": False}, True),  # env "on" beats YAML off
        ("0", {"reply_card": True}, False),  # "0" is falsy
    ],
)
def test_feishu_reply_card_env_beats_yaml(monkeypatch, env_value, extra, expected):
    if env_value is None:
        monkeypatch.delenv("FEISHU_REPLY_CARD", raising=False)
    else:
        monkeypatch.setenv("FEISHU_REPLY_CARD", env_value)

    settings = FeishuAdapter._load_settings(extra=extra)
    assert settings.reply_card is expected


# --- Payload builder ---------------------------------------------------------


def _bare_payload(content: str, reply_card: bool) -> tuple[str, str]:
    inst = object.__new__(FeishuAdapter)
    if reply_card:
        inst._reply_card = True
    return inst._build_outbound_payload(content)


def test_reply_card_off_keeps_legacy_payload_types():
    msg_type, _ = _bare_payload("just some plain prose", reply_card=False)
    assert msg_type == "text"


def test_reply_card_on_returns_interactive_card():
    msg_type, payload = _bare_payload(
        "\u3010Deploy done|green\u3011\nAll services are up.", reply_card=True
    )
    assert msg_type == "interactive"
    card = json.loads(payload)
    assert card["schema"] == "2.0"
    assert card["header"]["title"]["content"] == "Deploy done"
    assert card["header"]["template"] == "green"
    body_text = card["body"]["elements"][0]["content"]
    assert "All services are up." in body_text
    assert "Deploy done" not in body_text  # title not duplicated into the body


def test_reply_card_oversize_falls_back_to_legacy():
    msg_type, _ = _bare_payload("x" * 30_000, reply_card=True)
    assert msg_type in {"post", "text"}
