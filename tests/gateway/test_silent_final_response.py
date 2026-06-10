from __future__ import annotations

import pytest

from gateway.run import _sanitize_gateway_final_response


@pytest.mark.parametrize("sentinel", ["[SILENT]", " [[SILENT]] ", "<silent>"])
def test_silent_final_response_suppresses_delivery(sentinel):
    assert _sanitize_gateway_final_response("whatsapp", sentinel) == ""


def test_non_sentinel_response_is_preserved():
    assert _sanitize_gateway_final_response("whatsapp", "Actual reply") == "Actual reply"
