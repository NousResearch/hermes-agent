from __future__ import annotations


class _TtfbDisabled:
    """Distinct, non-falsy sentinel for an operator-disabled TTFB watchdog.

    Returned by :func:`get_provider_stream_ttfb_timeout` when the operator
    writes an explicit ``0``. It is intentionally NOT ``None`` (which means
    "unset — use the built-in default") and NOT ``False``/``0`` (which would
    collide with falsy/zero checks at call sites, e.g. ``if not ttfb:`` or
    ``if cfg == 0:``).

    Compared by VALUE (``==``), not identity. This module can be imported
    under two names in some run contexts (``hermes_cli.timeouts`` AND a
    top-level ``timeouts`` when ``sys.path`` contains the repo root), which
    would create two distinct instances and break an ``is`` check at the call
    site — silently failing to disable the watchdog. Value equality is
    reload-proof: any ``_TtfbDisabled`` equals any other.
    """

    __slots__ = ()

    _is_ttfb_disabled_sentinel = True

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "TTFB_DISABLED"

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        # Defensive: even though callers compare by value, make a stray
        # truthiness test fail loudly-safe (treated as "no positive timeout").
        return False

    def __eq__(self, other: object) -> bool:
        # Duck-typed, NOT isinstance: when this module is imported under two
        # names (hermes_cli.timeouts AND a top-level timeouts), there are two
        # distinct _TtfbDisabled CLASSES, so isinstance() would fail across
        # them. Match on the marker attribute instead so the sentinel compares
        # equal regardless of which module object produced it.
        return getattr(other, "_is_ttfb_disabled_sentinel", False) is True

    def __hash__(self) -> int:
        return hash("hermes._TtfbDisabled")


# Singleton sentinel. ``None`` means "not configured — caller should use its
# built-in default (120s)"; ``TTFB_DISABLED`` means "explicitly disabled by the
# operator (value == 0)". This is intentionally different from the idle
# resolver (which conflates unset and <=0 into ``None``), because the idle
# watchdog is the whole point of the Anthropic stream contract — there is no
# operator-facing "disable" sentinel for it. Compare with ``==`` (value
# equality), NOT ``is`` — see the double-import note in _TtfbDisabled.
TTFB_DISABLED = _TtfbDisabled()


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def get_provider_stream_idle_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured per-event idle timeout in seconds, if any.

    Contract (intentionally different from the TTFB resolver):

    * ``unset``/``null``/``<=0`` → ``None`` (caller substitutes a
      context-scaled default; the idle watchdog cannot be operator-disabled
      because a wedged Anthropic socket is the whole reason the watchdog
      exists — there is no shipped default that turns it off).
    * positive float → override.

    Reads ``providers.<id>.models.<model>.stream_idle_timeout_seconds`` first,
    falling back to ``providers.<id>.stream_idle_timeout_seconds``.
    """
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stream_idle_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stream_idle_timeout_seconds"))


def get_provider_stream_ttfb_timeout(
    provider_id: str, model: str | None = None
):
    """Return a configured no-first-byte (TTFB) timeout in seconds, or sentinel.

    Contract (intentionally different from the idle resolver):

    * ``unset``/``null`` → ``None`` (caller substitutes the 120s default).
    * explicit ``0`` → :data:`TTFB_DISABLED` (a distinct non-falsy sentinel
      object) — the TTFB watchdog is disabled for this provider/model.
    * positive float → override.

    A YAML file cannot express ``absent`` the way an env-var resolver can, so
    we MUST distinguish "operator wrote 0" from "operator wrote nothing". The
    distinction is load-bearing: idle ``0`` falls back to the default
    (anti-foot-gun), TTFB ``0`` is operator-disable.
    """
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    def _resolve_ttfb(node: dict[str, object]):
        if "stream_ttfb_timeout_seconds" not in node:
            return None  # truly absent — caller default
        raw = node.get("stream_ttfb_timeout_seconds")
        if raw is None:
            return None  # YAML ``~`` / null treated as unset
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if val == 0:
            return TTFB_DISABLED
        if val < 0:
            return None  # garbage — treat as unset
        return val

    model_config = _get_model_config(provider_config, model)
    if model_config is not None and "stream_ttfb_timeout_seconds" in model_config:
        resolved = _resolve_ttfb(model_config)
        if resolved is not None or model_config.get("stream_ttfb_timeout_seconds") == 0:
            return resolved

    return _resolve_ttfb(provider_config)


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
