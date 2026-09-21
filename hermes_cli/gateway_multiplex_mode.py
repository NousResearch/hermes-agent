"""Boot-time verdict for ``gateway.multiplex_profiles`` (the default is on, and there is no opt-out).

``GatewayConfig.from_dict`` leaves the flag ``None`` when neither config.yaml nor
``GATEWAY_MULTIPLEX_PROFILES`` set it. Turning the default on must not make a default gateway
double-bind a fleet that still runs per-profile gateways (two pollers on one bot token, port
fights), so the implicit default is a *request*: the gateway runs the same preflight
``hermes gateway migrate --multiplex`` runs and multiplexes only when the fold would have been
safe. An explicit ``true`` is never second-guessed.

An explicit ``false`` is RETIRED (multiplex-only ruling): it parses, it is logged, and it is then
resolved exactly like an unset key. The key itself survives because it is still the RUNTIME
mode flag every scoped code path reads (``config.multiplex_profiles``) — what it can no longer do
is pin a second gateway process onto this host.

The refusal is logged, never fatal: the gateway comes up standalone exactly as before the
default flipped, and the log names the blocker plus ``hermes gateway migrate --multiplex``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SINGLE_PROFILE_REASON = "only one profile exists (nothing to multiplex)"

#: ``gateway.multiplex_profiles: false`` is no longer an opt-out from the one-gateway-per-host
#: topology; it parses, it is reported, and it is ignored.
RETIRED_OPT_OUT_REASON = (
    "gateway.multiplex_profiles: false is retired and was ignored: one gateway per host serves "
    "every profile. Remove the key; `hermes gateway migrate --multiplex` folds any leftover "
    "per-profile gateway.")


def explicit_multiplex_flag(default_home: Path) -> Optional[bool]:
    """The operator's explicit choice for the DEFAULT profile's gateway: a recognized
    ``GATEWAY_MULTIPLEX_PROFILES``, else ``gateway.multiplex_profiles`` (or the top-level alias) as
    written in its config.yaml; ``None`` when neither is set. Raw read on purpose: the callers are
    other processes (``hermes -p X ...`` has X's config loaded) asking about the default's file."""
    from gateway.config import _bool_token, _env_multiplex_profiles_override
    env = _env_multiplex_profiles_override()
    if env is not None:
        return env
    cfg_path = Path(default_home) / "config.yaml"
    if not cfg_path.exists():
        return None
    from hermes_cli.config import read_user_config_raw
    cfg = read_user_config_raw(cfg_path) or {}
    gateway_section = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
    value = cfg.get("multiplex_profiles")
    if value is None:
        value = gateway_section.get("multiplex_profiles")
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _bool_token(value)
        return True if parsed is None else parsed
    return bool(value)


def default_gateway_multiplexes(default_home: Optional[Path] = None) -> bool:
    """Does the default profile's gateway serve every profile? For CLI/dashboard processes: the LIVE
    gateway's ``served_profiles`` record when one runs (it settled the unset default itself), else the
    explicit flag, else False — an unset flag is decided by the gateway at boot, never guessed here."""
    from hermes_constants import get_default_hermes_root
    from hermes_cli.gateway_multiplex_served import recorded_served_profiles
    root = Path(default_home) if default_home is not None else get_default_hermes_root()
    recorded = recorded_served_profiles(root)
    if recorded is not None:
        return bool(recorded)
    return bool(explicit_multiplex_flag(root))


@dataclass(frozen=True)
class MultiplexDecision:
    enabled: bool
    # "config" (config.yaml / env override — explicit), "default" (implicit default applied),
    # "guard" (implicit default refused; ``reason`` names the blocker).
    source: str
    reason: str = ""


def implicit_multiplex_blocker() -> Optional[str]:
    """Why THIS process must not multiplex right now, or None when it may.

    Mirrors what makes ``hermes gateway migrate --multiplex`` refuse or leave a per-profile gateway
    in place: hosts whose per-profile gateways the preflight cannot see (s6 slots) stay standalone;
    a secondary that still runs its own gateway (live process or installed service) or a preflight
    blocker (duplicate bot credential, port binder without a ``/p/<profile>/`` ingress) keeps this
    gateway standalone.

    Every blocker here is a TRANSIENT, fixable condition, which is why this function is now also
    the whole answer for an explicit ``gateway.multiplex_profiles: false`` (see
    :func:`resolve_multiplex_mode`): the host converges the moment the blocker is gone.

    The launching profile's IDENTITY is deliberately not a blocker: multiplex-only means "the one
    host process", whichever profile started it. Gating on ``active == 'default'`` made a host
    whose only gateway runs under a named profile permanently standalone — and every lifecycle
    verb built on "the default's multiplexer" blind to the process actually serving the host.
    """
    from hermes_cli.profiles import profiles_to_serve
    # Cheap and first: a single-profile install has nothing to multiplex, and the fail-closed secret
    # scope the multiplexer arms buys it nothing. (Also keeps every embedded/test runner off the
    # service-manager probes below.) Create a second profile and restart to start serving it.
    if len(profiles_to_serve(multiplex=True)) < 2:
        return SINGLE_PROFILE_REASON
    from hermes_cli.gateway_migrate import MIGRATE_COMMAND, _host_supports_migration, build_migration_plan
    host_reason = _host_supports_migration()
    if host_reason:
        return host_reason
    plan = build_migration_plan()
    if plan.standalone_secondaries:
        owned = ", ".join(
            f"'{p.name}' ({'pid ' + str(p.pid) if p.pid else p.service_label()})"
            for p in plan.standalone_secondaries)
        return f"profile(s) {owned} still run their own gateway; fold them with `{MIGRATE_COMMAND}`"
    if plan.blocked:
        return "; ".join(plan.blockers)
    return None


def resolve_multiplex_mode(config) -> MultiplexDecision:
    """Settle ``config.multiplex_profiles`` for one gateway boot; the config is updated in place.

    ``gateway.multiplex_profiles: false`` is RETIRED as a topology opt-out (multiplex-only ruling).
    The key still parses and still drives the runtime mode this function writes back, but an
    explicit ``false`` no longer pins a per-profile fleet: it is warned about and resolved exactly
    like an unset key. That is safe because the unset path is not optimistic — it refuses to
    multiplex while any real blocker holds (an s6 container, a secondary that still owns a gateway,
    a duplicate bot credential), so a host that genuinely cannot fold still comes up standalone and
    says why, and it converges by itself once ``hermes gateway migrate --multiplex`` has run.
    """
    current = getattr(config, "multiplex_profiles", None)
    if current:
        return MultiplexDecision(True, "config")
    retired_opt_out = current is False
    try:
        blocker = implicit_multiplex_blocker()
    except Exception as exc:  # a broken preflight must not take the gateway down with it
        logger.warning("Multiplex preflight failed; starting standalone: %s", exc, exc_info=True)
        blocker = f"preflight failed ({exc})"
    if blocker:
        decision = MultiplexDecision(False, "guard", blocker)
    elif retired_opt_out:
        decision = MultiplexDecision(True, "retired-opt-out", RETIRED_OPT_OUT_REASON)
    else:
        decision = MultiplexDecision(True, "default", "gateway.multiplex_profiles unset; default applies")
    config.multiplex_profiles = decision.enabled
    return decision


def record_multiplex_decision(decision: MultiplexDecision) -> None:
    """Persist a guard refusal into ``gateway_state.json`` so `hermes gateway status` can show why this
    gateway serves one profile while the default says multiplex; any other verdict clears the field."""
    try:
        from gateway.status import write_runtime_status
        write_runtime_status(multiplex_standalone_reason=decision.reason if decision.source == "guard" else None)
    except Exception:
        logger.debug("could not record the multiplex decision", exc_info=True)


def log_multiplex_decision(decision: MultiplexDecision) -> None:
    record_multiplex_decision(decision)
    if decision.source == "retired-opt-out":
        logger.warning("%s", RETIRED_OPT_OUT_REASON)
    elif decision.source == "guard" and decision.reason == SINGLE_PROFILE_REASON:
        logger.info("Single-profile install: gateway.multiplex_profiles unset, serving the default profile only.")
    elif decision.source == "guard":
        logger.warning(
            "This gateway stays standalone: %s. It serves the default profile only; the host "
            "converges once that is resolved.",
            decision.reason)
    elif decision.source == "default":
        logger.info("Serving every profile on this host (gateway.multiplex_profiles unset; default on).")
