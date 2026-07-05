"""hc-401 (was patch_native_agent_first_turn_ack.py / hc-214) — first-turn
"received, working" ack for native no-edit CN-IM entries, as a zero-in-place
overlay seam.

What this replaces
==================
Native no-edit IM tiers (wecom/weixin/dingtalk/qqbot) run an in-container
gateway whose adapters have NO ``edit_message`` and whose display tier
(``_TIER_LOW``) deliberately disables streaming and the long-running heartbeat.
Result: the ~14.6s first-inference window after the gateway is up is dead
silence (PD-214 / hc-214). The existing ``long_running_notifications`` heartbeat
only fires after ``_NOTIFY_INTERVAL`` (~180s), so it can't cover a sub-15s
window.

The original cloud fix was a build-time in-place patch touching TWO upstream
files:
  * ``gateway/display_config.py`` — a new per-platform ``first_turn_ack``
    display setting (default ON for the no-edit IM tier + qqbot, OFF elsewhere),
    added to the tier dicts + the ``_normalise`` bool-coercion set.
  * ``gateway/run.py`` — send one short ack at the start of a fresh turn, right
    before the ``agent:start`` hook (i.e. before the first model call),
    fire-and-forget and debounced per session.

This module re-expresses BOTH halves as a zero-in-place overlay so those hot
upstream files stay byte-for-byte upstream. It is a HYBRID seam:

  DATA (config):  the per-platform ``first_turn_ack: true`` values for the five
                  CN no-edit IM platforms live in ``cli-config.yaml.example``
                  under ``display.platforms.<platform>`` (a clean config seam,
                  same as the other display settings). Config overrides always win.

  BEHAVIOR (this module, two monkey-patches):
    1. ``gateway.display_config.resolve_display_setting`` — wrapped so that when
       asked for ``first_turn_ack`` it (a) coerces the resolved value to bool and
       (b) supplies the tier-derived default (True for the no-edit tier platforms
       {weixin, wecom, wecom_callback, dingtalk, qqbot}, else False) when nothing
       is configured. Because the upstream ``_normalise`` bool-set and the tier
       default dicts are in the (untouched) upstream file, we cannot register
       ``first_turn_ack`` there; instead the wrapper owns the coercion + default
       for this one key and delegates EVERY other setting to the original resolve
       unchanged. A config override under ``display.platforms.<p>.first_turn_ack``
       still flows through the original resolve first and wins over our default.
    2. ``GatewayRunner._handle_message_with_agent`` — wrapped so the ack fires at
       the very start of the turn (before the original body runs, hence before
       the ``agent:start`` emit + first model call). The original patch inserted
       the call literally in front of ``hooks.emit("agent:start")``; firing at
       method entry is functionally equivalent-or-earlier and far cleaner as a
       monkey-patch (no need to splice into the middle of a 300-line method). The
       per-session cooldown makes the exact fire point immaterial.

The ack helper body (``_send_first_turn_ack``) reproduces the original
``_hc214_send_first_turn_ack`` faithfully: same env kill-switches, same
per-session cooldown, same fire-and-forget send. The only change is that it
resolves the gate through this module's ``resolve_display_setting`` (which is the
patched one after ``apply()``), so the tier default is honored even when the
config example wasn't shipped.

HONEST COVERAGE BOUNDARY (unchanged from the original): this only shortens the
*first-inference* silence. The ~17s cold-start prefix (container boot + WS
connect + queue redelivery) happens before any in-container code runs and native
entries don't go through master, so it is physically un-ackable here — that
belongs to hc-175 (cold-start) / hc-142 (residency vs cost).

Kill switches / config:
  * ``HERMES_GATEWAY_FIRST_TURN_ACK_ENABLED``  (default ``true``)
  * ``HERMES_GATEWAY_FIRST_TURN_ACK_COOLDOWN`` (seconds, default ``30``)
  * ``HERMES_GATEWAY_FIRST_TURN_ACK_TEXT``     (default ``⏳ 收到，正在处理中…``)
  * per-platform: ``display.platforms.<platform>.first_turn_ack: true|false``

Idempotent (``_MARK`` sentinels + module ``_APPLIED``) and fail-safe. ``apply()``
returns False only if a target symbol is missing (the seam-test turns that into a
loud CI failure).
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_SETTING = "first_turn_ack"

# Upstream targets we monkey-patch — centralized so the seam-test pins them.
_TARGET_DISPLAY_MODULE = "gateway.display_config"
_TARGET_RESOLVE_FN = "resolve_display_setting"
_TARGET_RUN_MODULE = "gateway.run"
_TARGET_RUNNER_CLASS = "GatewayRunner"
_TARGET_TURN_METHOD = "_handle_message_with_agent"

# Platforms that get the ack by default (the no-edit IM tier + qqbot). Mirrors
# the original patch: _TIER_LOW CN entries + qqbot. Config can still override
# per-platform either way.
_ACK_DEFAULT_ON_PLATFORMS = frozenset(
    {"weixin", "wecom", "wecom_callback", "dingtalk", "qqbot"}
)

_DEFAULT_ACK_TEXT = "⏳ 收到，正在处理中…"  # "⏳ 收到，正在处理中…"
_DEFAULT_COOLDOWN_SECONDS = 30.0

_APPLIED = False
_MARK_RESOLVE = "_apex_overlay_first_turn_ack_resolve"
_MARK_TURN = "_apex_overlay_first_turn_ack_turn"


# ---------------------------------------------------------------------------
# Half 1: resolve_display_setting wrapper (the default + bool coercion)
# ---------------------------------------------------------------------------

def resolve_display_setting(user_config, platform_key, setting, fallback=None):
    """Overlay-aware resolve: owns ``first_turn_ack``, delegates everything else.

    For ``first_turn_ack``:
      * a per-platform / global config override is honored (via the original
        resolve, called with fallback=None so "unset" is distinguishable), and
        coerced to bool;
      * when nothing is configured, the tier default applies (True for the
        no-edit CN-IM platforms, else False).
    For every other setting this is a straight pass-through to the ORIGINAL
    upstream resolve — identical behavior.
    """
    import importlib

    display_mod = importlib.import_module(_TARGET_DISPLAY_MODULE)
    orig = getattr(display_mod, _TARGET_RESOLVE_FN)
    # If we somehow wrapped ourselves, unwrap to the real upstream via __wrapped__.
    real = getattr(orig, "__wrapped__", orig) if getattr(orig, _MARK_RESOLVE, False) else orig

    if setting != _SETTING:
        return real(user_config, platform_key, setting, fallback)

    # first_turn_ack: config override wins; else tier default.
    configured = real(user_config, platform_key, setting, None)
    if configured is not None:
        if isinstance(configured, str):
            return configured.strip().lower() in {"true", "1", "yes", "on"}
        return bool(configured)
    return (platform_key in _ACK_DEFAULT_ON_PLATFORMS)


def _wrap_resolve_display_setting(orig):
    @functools.wraps(orig)
    def wrapper(user_config, platform_key, setting, fallback=None):
        try:
            if setting == _SETTING:
                # Delegate to our resolver but bind the ORIGINAL as __wrapped__
                # so it can reach real upstream even mid-swap.
                configured = orig(user_config, platform_key, setting, None)
                if configured is not None:
                    if isinstance(configured, str):
                        return configured.strip().lower() in {"true", "1", "yes", "on"}
                    return bool(configured)
                return platform_key in _ACK_DEFAULT_ON_PLATFORMS
        except Exception:
            # Never break a display lookup — fall through to upstream.
            pass
        return orig(user_config, platform_key, setting, fallback)

    setattr(wrapper, _MARK_RESOLVE, True)
    return wrapper


# ---------------------------------------------------------------------------
# Half 2: the ack send helper + the turn-start method wrapper
# ---------------------------------------------------------------------------

async def _send_first_turn_ack(runner, source, session_key) -> None:
    """One-shot "received, working" ack at the start of a fresh turn.

    Gated by the per-platform ``first_turn_ack`` display setting (default on only
    for the no-edit IM tier: wecom/weixin/dingtalk/wecom_callback/qqbot).
    Fire-and-forget and fully defensive: it must never delay, block, or break a
    turn. Cannot and does NOT cover the cold-start prefix — see module docstring.

    Faithful reproduction of the original ``_hc214_send_first_turn_ack``, with
    the gate resolved through this overlay's ``resolve_display_setting`` (the
    patched one) so the tier default is honored without the config example.
    """
    import asyncio
    import os
    import time

    try:
        if os.environ.get("HERMES_GATEWAY_FIRST_TURN_ACK_ENABLED", "true").strip().lower() != "true":
            return
        platform = getattr(source, "platform", None)
        if platform is None:
            return

        from gateway.run import _load_gateway_config, _platform_config_key

        platform_key = _platform_config_key(platform)
        if not bool(
            resolve_display_setting(
                _load_gateway_config(), platform_key, _SETTING, False
            )
        ):
            return

        adapter = runner.adapters.get(platform)
        chat_id = getattr(source, "chat_id", None)
        if not adapter or not chat_id:
            return

        now = time.time()
        cooldown = _float_env(
            "HERMES_GATEWAY_FIRST_TURN_ACK_COOLDOWN", _DEFAULT_COOLDOWN_SECONDS
        )
        ts_map = getattr(runner, "_apex_first_turn_ack_ts", None)
        if ts_map is None:
            ts_map = {}
            runner._apex_first_turn_ack_ts = ts_map
        if now - ts_map.get(session_key, 0) < cooldown:
            return
        ts_map[session_key] = now

        text = os.environ.get(
            "HERMES_GATEWAY_FIRST_TURN_ACK_TEXT", _DEFAULT_ACK_TEXT
        )

        async def _send() -> None:
            try:
                await adapter.send(chat_id, text)
            except Exception as _se:  # noqa: BLE001
                logger.debug("apex_overlay: first_turn_ack send failed: %s", _se)

        asyncio.create_task(_send())
    except Exception as _e:  # noqa: BLE001
        logger.debug("apex_overlay: first_turn_ack skipped: %s", _e)


def _float_env(name: str, default: float) -> float:
    """Local copy of gateway.run._float_env (avoids depending on its export)."""
    import os

    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _wrap_turn_method(orig):
    """Wrap ``_handle_message_with_agent`` to fire the ack at turn start.

    Fires before the original body (before ``agent:start`` + first model call),
    using ``_quick_key`` (the stable per-session key the running-agents sentinel
    uses) as the cooldown key. Fully non-blocking: the ack is a fire-and-forget
    task; the original method proceeds regardless.
    """

    @functools.wraps(orig)
    async def wrapper(self, event, source, _quick_key, run_generation, *args, **kwargs):
        try:
            await _send_first_turn_ack(self, source, _quick_key)
        except Exception:
            # The ack must never block or break the turn.
            pass
        return await orig(self, event, source, _quick_key, run_generation, *args, **kwargs)

    setattr(wrapper, _MARK_TURN, True)
    return wrapper


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Install both halves of the first_turn_ack seam. Idempotent, fail-safe.

    Returns True when both patches are in place (or already present), False if a
    target symbol is missing (the seam-test turns that into a hard CI failure).
    Never raises into plugin discovery.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    ok = True

    # Patch 1: resolve_display_setting (default + bool coercion for first_turn_ack).
    try:
        display_mod = importlib.import_module(_TARGET_DISPLAY_MODULE)
        orig_resolve = getattr(display_mod, _TARGET_RESOLVE_FN)
        if not getattr(orig_resolve, _MARK_RESOLVE, False):
            setattr(
                display_mod, _TARGET_RESOLVE_FN,
                _wrap_resolve_display_setting(orig_resolve),
            )
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not patch %s.%s — first_turn_ack default is NOT "
            "active (the ack still fires only if config sets it explicitly). "
            "Upstream may have renamed/moved it. (%s)",
            _TARGET_DISPLAY_MODULE, _TARGET_RESOLVE_FN, exc,
        )
        ok = False

    # Patch 2: the turn-start method (send the ack before the first model call).
    try:
        run_mod = importlib.import_module(_TARGET_RUN_MODULE)
        runner_cls = getattr(run_mod, _TARGET_RUNNER_CLASS)
        orig_turn = getattr(runner_cls, _TARGET_TURN_METHOD)
        if not getattr(orig_turn, _MARK_TURN, False):
            setattr(runner_cls, _TARGET_TURN_METHOD, _wrap_turn_method(orig_turn))
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not patch %s.%s.%s — first_turn_ack SEND half is "
            "NOT active (no ack will be sent). Upstream may have renamed/moved it. "
            "(%s)",
            _TARGET_RUN_MODULE, _TARGET_RUNNER_CLASS, _TARGET_TURN_METHOD, exc,
        )
        ok = False

    _APPLIED = ok
    if ok:
        logger.debug("apex_overlay: first_turn_ack seam applied (both halves)")
    return ok
