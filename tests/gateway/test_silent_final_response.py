from __future__ import annotations

import pytest

from gateway.platforms.base import _has_visible_delivery_text
from gateway.run import _sanitize_gateway_final_response


@pytest.mark.parametrize(
    "sentinel",
    [
        "[SILENT]",
        " [[SILENT]] ",
        "<silent>",
        "\u200b[SILENT]\u200b",
        "silencio",
        "Silencio — no fui aludido directamente.",
        "`silencio`",
    ],
)
def test_silent_final_response_suppresses_delivery(sentinel):
    assert _sanitize_gateway_final_response("whatsapp", sentinel) == ""


@pytest.mark.parametrize("blankish", ["\u200b", "\u200b \n\u2060", "\ufeff\u200d"])
def test_invisible_only_final_response_suppresses_delivery(blankish):
    assert _sanitize_gateway_final_response("whatsapp", blankish) == ""
    assert not _has_visible_delivery_text(blankish)


def test_base_delivery_guard_rejects_internal_silent_sentinels():
    assert not _has_visible_delivery_text("[SILENT]")
    assert not _has_visible_delivery_text("Silencio — no fui aludido directamente.")
    assert _has_visible_delivery_text("Actual reply")
    assert _has_visible_delivery_text("Necesitamos guardar silencio durante la llamada")


def test_non_sentinel_response_is_preserved():
    assert _sanitize_gateway_final_response("whatsapp", "Actual reply") == "Actual reply"
