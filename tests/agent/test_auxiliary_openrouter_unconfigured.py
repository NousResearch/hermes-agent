"""Contratos de comportamiento: ausencia de credencial no es una caída de proveedor.

Antes de este cambio, `_try_openrouter()` marcaba OpenRouter como *unhealthy* y
registraba un WARNING que afirmaba «payment / credit error» cuando la causa real
era que **no hay credencial configurada**. Nada se intentó: no hubo llamada de
red ni proveedor agotado, así que marcar el endpoint como caído era falso y
producía un aviso recurrente y engañoso en cada llamada auxiliar.

Estos tests afirman comportamiento observable (estado del caché de salud y la
línea registrada), no el texto del código fuente.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _clean_health_cache():
    """Cada test parte de un caché de salud limpio y lo deja limpio."""
    from agent import auxiliary_client as ac

    ac._aux_unhealthy_until.clear()
    ac._aux_unhealthy_logged_at.clear()
    yield
    ac._aux_unhealthy_until.clear()
    ac._aux_unhealthy_logged_at.clear()


def test_openrouter_without_credential_is_not_a_provider_outage(monkeypatch, caplog):
    """Sin credencial: no hay cliente, el endpoint sigue sano y no se habla de pago.

    El caché de salud existe para saltar un proveedor comprobadamente caído y ahorrar un RTT.
    Aquí no se intentó nada, así que entrar al caché es incorrecto y el aviso afirma una causa
    (pago/crédito) que no se observó.
    """
    from agent import auxiliary_client as ac

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ac, "_aux_openrouter_settings", lambda: (False, "test/model:free"))
    # Sin entrada en el credential pool: el camino realista de un perfil sin OpenRouter.
    monkeypatch.setattr(ac, "_select_pool_entry", lambda provider: (False, None))

    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
        client, model = ac._try_openrouter()

    assert (client, model) == (None, None), "sin credencial no debe devolver cliente"
    assert not ac._is_provider_unhealthy("openrouter"), (
        "la ausencia de credencial no es una caída del proveedor: no debe marcarse unhealthy"
    )
    falsos = [
        r.getMessage() for r in caplog.records
        if "payment" in r.getMessage().lower() or "credit" in r.getMessage().lower()
    ]
    assert not falsos, f"no debe atribuirse a pago/crédito una credencial ausente: {falsos}"


def test_mark_unhealthy_reports_the_reason_it_is_given(caplog):
    """Contrato: el mensaje usa el motivo recibido; sin motivo, conserva pago/crédito."""
    from agent import auxiliary_client as ac

    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
        ac._mark_provider_unhealthy("openrouter", ttl=60, reason="credential not configured")
        ac._mark_provider_unhealthy("openrouter", ttl=60)

    msgs = [r.getMessage() for r in caplog.records if "unhealthy" in r.getMessage()]
    assert len(msgs) == 2, f"deben registrarse los dos avisos: {msgs}"
    assert "credential not configured" in msgs[0], "el mensaje cita el motivo recibido"
    assert "payment" not in msgs[0].lower(), "no afirma pago cuando el motivo es otro"
    assert "payment" in msgs[1].lower(), "el motivo por defecto sigue siendo el error de pago"
