"""Known-bad control para H3 (sin dependencia de pytest: corre igual en 3.11 y 3.12)

Original doc:
Known-bad control para H3: un cron nunca tiene mas permisos que quien lo creo.

Debe correr ROJO contra el codigo de hoy (`_resolve_cron_enabled_toolsets` da
precedencia absoluta a `job['enabled_toolsets']` sin mirar `origin.platform`) y
VERDE con el fix B4-8 (b). Hermetico: no lee configs de la maquina.

Destino con el fix: tests/cron/test_cron_toolset_origin_intersect.py
"""
from cron.scheduler import _resolve_cron_enabled_toolsets

# Config minima: whatsapp_cloud sin terminal / code_execution, como la flota.
CFG = {
    "tools": {
        "platform_toolsets": {
            "whatsapp_cloud": ["web", "file", "memory"],
            "telegram": ["web", "file", "memory"],
        }
    }
}


def _job(**kw):
    job = {
        "id": "test0001",
        "enabled": True,
        "origin": {"platform": "whatsapp_cloud"},
    }
    job.update(kw)
    return job


def test_job_no_puede_darse_terminal_desde_una_plataforma_sin_terminal():
    """El agente pide terminal en un cron creado desde WhatsApp: no debe resolverlo."""
    resolved = _resolve_cron_enabled_toolsets(
        _job(enabled_toolsets=["terminal", "file", "web"]), CFG
    ) or []
    assert "terminal" not in resolved, (
        "un cron creado desde whatsapp_cloud resolvio terminal: "
        f"{sorted(resolved)}"
    )


def test_job_conserva_lo_que_su_plataforma_si_permite():
    """La interseccion no debe vaciar el job: lo permitido sobrevive."""
    resolved = _resolve_cron_enabled_toolsets(
        _job(enabled_toolsets=["terminal", "file", "web"]), CFG
    ) or []
    assert "file" in resolved and "web" in resolved, sorted(resolved)


def test_sin_lista_hereda_de_su_plataforma_de_origen_no_de_cron():
    """58/60 jobs de Mia no traen lista. Hoy caen a `cron` (28 toolsets con terminal)."""
    resolved = _resolve_cron_enabled_toolsets(_job(), CFG) or []
    assert "terminal" not in resolved, (
        f"job sin lista heredo terminal: {sorted(resolved)}"
    )


def _terminal_no_se_gana_por_cron(plataforma):
    resolved = _resolve_cron_enabled_toolsets(
        _job(origin={"platform": plataforma}, enabled_toolsets=["terminal"]), CFG
    ) or []
    assert "terminal" not in resolved, f"{plataforma}: {sorted(resolved)}"


def test_whatsapp_cloud_no_gana_terminal_por_cron():
    _terminal_no_se_gana_por_cron("whatsapp_cloud")


def test_telegram_no_gana_terminal_por_cron():
    _terminal_no_se_gana_por_cron("telegram")


def test_resolver_que_falla_es_fail_closed_no_default_completo():
    """(d) de Fable: si el resolver revienta, el job NO debe recibir terminal.

    Hoy `_resolve_cron_enabled_toolsets` devuelve None en el `except`, y None
    significa 'AIAgent carga el set default completo' — con terminal,
    code_execution, browser y computer_use. Es la fuente
    `full_default_on_resolve_fail` que Fable vio en los 7 jobs de arq-staging.

    Sin `monkeypatch` a proposito: asi este caso tambien corre en el
    interprete real del gateway (3.11, sin pytest), no solo bajo pytest.
    """
    import hermes_cli.tools_config as tc

    def _boom(*a, **kw):
        raise RuntimeError("config ilegible")

    original = tc._get_platform_tools
    tc._get_platform_tools = _boom
    try:
        resolved = _resolve_cron_enabled_toolsets(_job(), CFG)
    finally:
        tc._get_platform_tools = original

    assert resolved is not None, (
        "el resolver fallo y devolvio None = set default completo con terminal "
        "(fail-open); debe ser fail-closed"
    )
    assert "terminal" not in resolved, sorted(resolved)
