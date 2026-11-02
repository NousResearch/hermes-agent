"""`display.status_diagnostics` — delivery gate for the platform's own status diagnostics (#REQ-INS1-SAAS-034).

A client-facing profile's chat surface must carry the client's answers, not the platform's
route/health chatter. The one status measured landing on such a surface is the model-route
fallback notice (`agent/chat_completion_helpers.py::_buffer_fallback_notice`), delivered through
the single status chokepoint `gateway.run._prepare_gateway_status_message`.

Default is ON: the gate is a no-op until a deployment sets
`display.status_diagnostics: false` (or `display.platforms.<platform>.status_diagnostics: false`),
so the pre-change behaviour is what the first test pins. Suppression is DELIVERY only — the status
callback logs it and the line stays in the agent log.
"""

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message

CHAT_PLATFORMS = ["telegram", "discord", "slack", "whatsapp"]
RAW_PLATFORMS = ["local", "api_server", "webhook", "msgraph_webhook"]

# The real notice shape, as measured on a live firm channel (2026-09-13): 27 of the last 100
# messages of the channel were this line. `_buffer_fallback_notice` appends the retry-eligibility
# clause when the primary route is rate limited.
MODEL_FALLBACK_NOTICE = (
    "⚠️ Model fallback: muse-spark-1.3-contributor-free via opencode-free unavailable "
    "(rate limit); using deepseek-v4-flash via deepseek."
)
MODEL_FALLBACK_NOTICE_WITH_COOLDOWN = (
    MODEL_FALLBACK_NOTICE + " Primary retry eligible in ~42 s; recovery is not guaranteed."
)

# Statuses that are not platform diagnostics: a deployment that silences the platform's own
# diagnostics must still receive everything a user's request produces.
NON_DIAGNOSTIC_STATUSES = [
    "Compressed: 30 → 12 messages",
    "Compression aborted: 30 messages preserved",
    "🗜️ Context reduced to 120,000 tokens, retrying...",  # compression retry chatter, own filter
    "Todo list updated (3 open items)",
]


@pytest.fixture
def diagnostics_default(monkeypatch):
    """Config without the key — chat surfaces receive the platform's diagnostics (pre-change behaviour)."""
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})


@pytest.fixture
def diagnostics_off(monkeypatch):
    monkeypatch.setattr(
        gateway_run, "_load_gateway_config", lambda: {"display": {"status_diagnostics": False}}
    )


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize(
    "message", [MODEL_FALLBACK_NOTICE, MODEL_FALLBACK_NOTICE_WITH_COOLDOWN], ids=["notice", "notice+cooldown"]
)
def test_default_config_delivers_model_fallback_notice(diagnostics_default, platform, message):
    """Negative-witness 'before' half: on the unchanged default the notice is delivered verbatim."""
    assert _prepare_gateway_status_message(platform, "warn", message) == message


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize(
    "message", [MODEL_FALLBACK_NOTICE, MODEL_FALLBACK_NOTICE_WITH_COOLDOWN], ids=["notice", "notice+cooldown"]
)
def test_gate_off_drops_model_fallback_notice(diagnostics_off, platform, message):
    """Negative-witness 'after' half: same input, same platforms — dropped once the surface opts out."""
    assert _prepare_gateway_status_message(platform, "warn", message) is None


def test_gate_off_keeps_raw_platforms(diagnostics_off):
    """Programmatic surfaces keep raw status text; the switch is about chat delivery only."""
    for platform in RAW_PLATFORMS:
        assert (
            _prepare_gateway_status_message(platform, "warn", MODEL_FALLBACK_NOTICE)
            == MODEL_FALLBACK_NOTICE
        )


@pytest.mark.parametrize("message", NON_DIAGNOSTIC_STATUSES, ids=lambda m: m[:32])
def test_gate_off_does_not_swallow_other_statuses(diagnostics_off, message):
    """The gate is scoped to platform diagnostics, not to `status_diagnostics: false` meaning 'silence chat'."""
    assert _prepare_gateway_status_message("discord", "warn", message) == message


def test_gate_off_applies_per_platform(monkeypatch):
    """`display.platforms.<platform>` overrides the profile-wide value for that surface only."""
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"platforms": {"discord": {"status_diagnostics": False}}}},
    )
    assert _prepare_gateway_status_message("discord", "warn", MODEL_FALLBACK_NOTICE) is None
    assert (
        _prepare_gateway_status_message("telegram", "warn", MODEL_FALLBACK_NOTICE)
        == MODEL_FALLBACK_NOTICE
    )


@pytest.mark.parametrize("value", ["false", "off", "no", "0", False], ids=str)
def test_string_and_bool_false_forms_all_disable_the_gate(monkeypatch, value):
    """YAML quirks: a quoted `"false"`, `off`/`no`/`0` and a real bool all disable delivery."""
    monkeypatch.setattr(
        gateway_run, "_load_gateway_config", lambda: {"display": {"status_diagnostics": value}}
    )
    assert _prepare_gateway_status_message("discord", "warn", MODEL_FALLBACK_NOTICE) is None


def test_unreadable_config_fails_open(monkeypatch):
    """A config read error must not start dropping statuses: the default is delivery."""

    def _boom():
        raise OSError("config unreadable")

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _boom)
    assert gateway_run._gateway_status_diagnostics_enabled(Platform.DISCORD) is True
    assert (
        _prepare_gateway_status_message("discord", "warn", MODEL_FALLBACK_NOTICE)
        == MODEL_FALLBACK_NOTICE
    )


def test_diagnostic_regex_covers_the_emitted_notice_shape():
    """Coupling guard: the notice `_buffer_fallback_notice` emits must match the gate's pattern.

    Rewording the notice without updating `_GATEWAY_STATUS_DIAGNOSTIC_RE` would silently re-open
    delivery on every surface that opted out.
    """
    assert gateway_run._GATEWAY_STATUS_DIAGNOSTIC_RE.search(MODEL_FALLBACK_NOTICE)
    assert gateway_run._GATEWAY_STATUS_DIAGNOSTIC_RE.search(MODEL_FALLBACK_NOTICE_WITH_COOLDOWN)
