"""Selective Telegram link previews by domain + per-message override (#120029).

Global ``disable_link_previews`` is all-or-nothing; calendar-event
notifications want calendar links quiet while keeping useful previews
elsewhere. Covered behavior:

- ``link_preview_disabled_domains`` extra suppresses previews only when the
  message text contains a URL on a listed domain (exact or subdomain,
  case-insensitive).
- Non-matching messages keep previews.
- ``metadata["disable_link_preview"]`` is a per-message override for
  notification producers (True forces off, False forces on).
- Rich-message payloads respect the same policy.
"""

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra):
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))


def _is_disabled(kwargs):
    """True when the kwargs actually suppress the preview (either SDK shape)."""
    if not kwargs:
        return False
    lp = kwargs.get("link_preview_options")
    if lp is None:
        return bool(kwargs.get("disable_web_page_preview"))
    return bool(getattr(lp, "is_disabled", False))


CAL = "https://calendar.google.com/calendar/event?eid=abc123"
OTHER = "https://example.com/some/article"


def test_default_previews_enabled():
    a = _make_adapter({})
    assert a._link_preview_kwargs("see " + CAL) == {}


def test_global_disable_still_disables_everything():
    a = _make_adapter({"disable_link_previews": True})
    assert _is_disabled(a._link_preview_kwargs("plain text, no links"))
    assert _is_disabled(a._link_preview_kwargs("see " + OTHER))


def test_domain_match_disables_calendar_link():
    a = _make_adapter({"link_preview_disabled_domains": ["calendar.google.com"]})
    assert _is_disabled(a._link_preview_kwargs("event: " + CAL))


def test_domain_nonmatch_keeps_preview():
    a = _make_adapter({"link_preview_disabled_domains": ["calendar.google.com"]})
    assert a._link_preview_kwargs("read " + OTHER) == {}


def test_subdomain_and_case_insensitive_match():
    a = _make_adapter({"link_preview_disabled_domains": ["Google.com"]})
    assert _is_disabled(a._link_preview_kwargs("meet https://MEET.Google.COM/xyz"))
    assert a._link_preview_kwargs("read " + OTHER) == {}


def test_comma_string_config_shape():
    a = _make_adapter({"link_preview_disabled_domains": "calendar.google.com, meet.google.com"})
    assert _is_disabled(a._link_preview_kwargs("event: " + CAL))


def test_metadata_disable_forces_off_without_domain_match():
    a = _make_adapter({})
    assert _is_disabled(
        a._link_preview_kwargs("read " + OTHER, {"disable_link_preview": True}))


def test_metadata_enable_forces_on_over_global_and_domain():
    a = _make_adapter({
        "disable_link_previews": True,
        "link_preview_disabled_domains": ["calendar.google.com"],
    })
    assert a._link_preview_kwargs("event: " + CAL, {"disable_link_preview": False}) == {}


def test_rich_payload_respects_domain_policy():
    a = _make_adapter({"link_preview_disabled_domains": ["calendar.google.com"]})
    payload = a._rich_payload_base("123", "event: " + CAL)
    assert payload.get("link_preview_options", {}).get("is_disabled") is True
    payload_ok = a._rich_payload_base("123", "read " + OTHER)
    assert "link_preview_options" not in payload_ok
