"""Known-bad control for H3-b (no pytest: same file on 3.11 gateway and 3.12).

Two holes in fa9735ca:

1. A job with origin and no per-job list inherits the *full* origin platform.
   Origin often has toolsets the ``cron`` platform does not (messaging, moa).
   That is a swap, not a reduction. Fix: no-list branch = origin ∩ cron.

2. ``clamp_cron_enabled_toolsets_to_origin`` returns ``clamped or None``, and
   the resolver treats ``None`` and ``[]`` the same (``if per_job:``). An empty
   intersection therefore inherits the origin list at fire. Fix: persist ``[]``
   and distinguish ``None`` (inherit) from ``[]`` (explicit empty).

Must run RED against fa9735ca and GREEN after H3-b. Hermetic: does not read
the machine's config.yaml. No pytest fixtures (ley #339).
"""
from cron.scheduler import (
    clamp_cron_enabled_toolsets_to_origin,
    _resolve_cron_enabled_toolsets,
)

# Origin has extras cron does not (messaging, moa). Cron has extras origin
# does not (terminal). Overlap is file/web/memory/cronjob.
CFG = {
    "tools": {
        "platform_toolsets": {
            "whatsapp_cloud": [
                "web",
                "file",
                "memory",
                "cronjob",
                "messaging",
                "moa",
            ],
            "telegram": [
                "web",
                "file",
                "memory",
                "cronjob",
                "messaging",
                "moa",
            ],
            "cron": ["web", "file", "memory", "cronjob", "terminal", "skills"],
        }
    },
    "mcp_servers": {},
}


def _job(**kw):
    job = {
        "id": "h3b0001",
        "enabled": True,
        "origin": {"platform": "whatsapp_cloud"},
    }
    job.update(kw)
    return job


def test_sin_lista_no_gana_lo_que_cron_no_tiene():
    """No-list job must not pick up origin-only toolsets (messaging, moa)."""
    resolved = _resolve_cron_enabled_toolsets(_job(), CFG) or []
    extra = sorted(set(resolved) & {"messaging", "moa"})
    assert extra == [], (
        "job sin lista gano toolsets que cron no tiene: "
        f"{extra} ; resolved={sorted(resolved)}"
    )


def test_sin_lista_no_gana_terminal_del_cron():
    """Intersection is monotone both ways: cron-only terminal stays off."""
    resolved = _resolve_cron_enabled_toolsets(_job(), CFG) or []
    assert "terminal" not in resolved, sorted(resolved)


def test_sin_lista_conserva_el_solape():
    resolved = _resolve_cron_enabled_toolsets(_job(), CFG) or []
    for name in ("file", "web", "memory"):
        assert name in resolved, sorted(resolved)


def test_lista_vacia_no_hereda_origen():
    """Explicit [] is not None. Must not inherit origin (messaging/moa/…)."""
    resolved = _resolve_cron_enabled_toolsets(
        _job(enabled_toolsets=[]), CFG
    )
    assert resolved == [], (
        "enabled_toolsets=[] heredo origen: "
        f"{sorted(resolved) if resolved else resolved}"
    )


def test_clamp_interseccion_vacia_es_lista_vacia_no_none():
    out = clamp_cron_enabled_toolsets_to_origin(
        ["terminal"],
        {"platform": "whatsapp_cloud"},
        CFG,
    )
    assert out == [], f"clamp vacio debio persistir [] no {out!r}"
