"""X-Hermes-Deliver — request-bound delivery for OpenAI-compatible API-server turns.

The mail listener used to decide where a tick's report went by *asking the model in prose*
("Report to #projects."), so routing drifted silently for weeks. These tests pin the
replacement contract: the destination is a request header, it must be API-key authenticated,
and it must match the operator's configured allow-list exactly. Delivery itself stays
best-effort — it can never turn a completed turn into an error response.

Deliberately covered here and not in the handler tests:
  * allow-list normalisation (what an operator may configure),
  * header rejection shapes (what a caller may not escalate to),
  * router outcomes -> ``X-Hermes-Delivered`` token, including the raising-router case.
"""

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

TARGET = "discord:1471029524551565456:1553817978175234149"


def _adapter(**extra) -> APIServerAdapter:
    """Adapter with production-ish auth; ``extra`` overrides api_server config keys."""
    cfg = {"key": "test-api-key"}
    cfg.update(extra)
    return APIServerAdapter(PlatformConfig(enabled=True, extra=cfg))


def _request(**headers) -> SimpleNamespace:
    """Minimal stand-in for ``web.Request`` — the parser only reads ``.headers``."""
    return SimpleNamespace(headers=headers)


class _Router:
    """Records what the adapter asked the gateway to deliver."""

    def __init__(self, *, result=None, raises=None):
        self.calls = []
        self._result = result
        self._raises = raises

    async def deliver(self, content, targets, **_kw):
        self.calls.append((content, [t.to_string() for t in targets]))
        if self._raises is not None:
            raise self._raises
        if self._result is not None:
            return self._result
        return {targets[0].to_string(): {"success": True, "result": "ok"}}


def _attach(adapter: APIServerAdapter, router) -> APIServerAdapter:
    adapter.gateway_runner = SimpleNamespace(delivery_router=router)
    return adapter


# --------------------------------------------------------------------------------------
# Allow-list normalisation
# --------------------------------------------------------------------------------------


def test_absent_allow_list_disables_the_header():
    adapter = _adapter()
    assert adapter._deliver_allow == frozenset()
    target, err = adapter._parse_deliver_header(_request(**{"X-Hermes-Deliver": TARGET}))
    assert target is None
    assert err is not None and err.status == 403


def test_allow_list_keeps_concrete_targets_and_normalises_platform_case():
    adapter = _adapter(deliver_allow=[TARGET.replace("discord", "Discord")])
    assert adapter._deliver_allow == frozenset({TARGET})


def test_allow_list_drops_home_channel_origin_local_and_unknown_platforms(caplog):
    adapter = _adapter(
        deliver_allow=["discord", "origin", "local", "not_a_platform:1:2", "", None, TARGET]
    )
    # Only the concrete platform:chat:thread entry survives; the rest would be free-form
    # routing (home channel) or non-routable and must never be silently upgraded.
    assert adapter._deliver_allow == frozenset({TARGET})


def test_allow_list_accepts_a_bare_string():
    adapter = _adapter(deliver_allow=TARGET)
    assert adapter._deliver_allow == frozenset({TARGET})


def test_allow_list_ignores_a_non_list_mapping():
    adapter = _adapter(deliver_allow={"discord": TARGET})
    assert adapter._deliver_allow == frozenset()


# --------------------------------------------------------------------------------------
# Header validation
# --------------------------------------------------------------------------------------


def test_absent_header_means_no_delivery():
    adapter = _adapter(deliver_allow=[TARGET])
    assert adapter._parse_deliver_header(_request()) == (None, None)


def test_authorised_target_is_accepted():
    adapter = _adapter(deliver_allow=[TARGET])
    assert adapter._parse_deliver_header(_request(**{"X-Hermes-Deliver": TARGET})) == (TARGET, None)


def test_target_outside_allow_list_is_403():
    adapter = _adapter(deliver_allow=[TARGET])
    target, err = adapter._parse_deliver_header(
        _request(**{"X-Hermes-Deliver": "discord:999:999"})
    )
    assert target is None and err.status == 403


def test_home_channel_shorthand_is_not_a_valid_authorised_target():
    """A bare platform name resolves to the home channel — never acceptable as a binding."""
    adapter = _adapter(deliver_allow=["discord"])
    assert adapter._deliver_allow == frozenset()
    target, err = adapter._parse_deliver_header(_request(**{"X-Hermes-Deliver": "discord"}))
    assert target is None and err.status == 403


def test_local_target_is_rejected():
    adapter = _adapter(deliver_allow=[TARGET, "local:1"])
    target, err = adapter._parse_deliver_header(_request(**{"X-Hermes-Deliver": "local:1"}))
    assert target is None and err.status == 403


def test_header_requires_api_key_authentication():
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "", "deliver_allow": [TARGET]})
    )
    target, err = adapter._parse_deliver_header(_request(**{"X-Hermes-Deliver": TARGET}))
    assert target is None and err.status == 403


@pytest.mark.parametrize("header", ["discord:1\r\nX-Evil: 1", "discord:1\x00", "discord:1\n"])
def test_control_characters_are_rejected(header):
    adapter = _adapter(deliver_allow=[TARGET])
    target, err = adapter._parse_deliver_header(_request(**{"X-Hermes-Deliver": header}))
    assert target is None and err.status == 400


def test_overlong_header_is_rejected():
    adapter = _adapter(deliver_allow=[TARGET])
    target, err = adapter._parse_deliver_header(
        _request(**{"X-Hermes-Deliver": "discord:" + "9" * 300})
    )
    assert target is None and err.status == 400


# --------------------------------------------------------------------------------------
# Delivery outcome -> X-Hermes-Delivered token
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_router_reports_ok_and_passes_the_target():
    router = _Router()
    adapter = _attach(_adapter(deliver_allow=[TARGET]), router)
    assert await adapter.deliver_final_response(TARGET, "  report body  ") == "ok"
    assert router.calls == [("report body", [TARGET])]


@pytest.mark.asyncio
async def test_missing_runner_reports_unavailable_and_sends_nothing():
    adapter = _adapter(deliver_allow=[TARGET])
    assert await adapter.deliver_final_response(TARGET, "body") == "unavailable"


@pytest.mark.asyncio
async def test_empty_text_sends_nothing():
    router = _Router()
    adapter = _attach(_adapter(deliver_allow=[TARGET]), router)
    assert await adapter.deliver_final_response(TARGET, "   ") == "empty"
    assert router.calls == []


@pytest.mark.asyncio
async def test_raising_router_is_swallowed_and_reports_error():
    adapter = _attach(_adapter(deliver_allow=[TARGET]), _Router(raises=RuntimeError("boom")))
    assert await adapter.deliver_final_response(TARGET, "body") == "error"


@pytest.mark.asyncio
async def test_unsuccessful_router_result_reports_error():
    router = _Router(result={TARGET: {"success": False, "error": "not connected"}})
    adapter = _attach(_adapter(deliver_allow=[TARGET]), router)
    assert await adapter.deliver_final_response(TARGET, "body") == "error"


@pytest.mark.asyncio
async def test_dead_target_skip_reports_error_not_ok():
    router = _Router(result={TARGET: {"success": False, "skipped": "dead_target"}})
    adapter = _attach(_adapter(deliver_allow=[TARGET]), router)
    assert await adapter.deliver_final_response(TARGET, "body") == "error"
