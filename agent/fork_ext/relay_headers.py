"""Relay-pool routing headers for fork-owned Claude API proxy lanes."""

from __future__ import annotations

import os


def _pool_lane(agent, aux_task=None) -> str:
    """Classify a pool request into the interactive vs background lane (Phase 2,
    spec 2026-07-05 claude-relay-lanes). The relay reserves headroom / de-prioritizes
    background under contention so a burst of background work can't degrade a live
    interactive turn.

      * A top-level (non-delegated, non-cron/headless) main turn → ``interactive``.
      * A subagent (``_delegate_depth > 0``) or a cron/headless run → ``background``.
      * Auxiliary calls (``aux_task`` set): CRITICAL aux (compaction/title/vision on a
        live top-level turn's critical path — the user turn blocks on it) → ``interactive``
        (B1: never damp the compaction a live turn is waiting on); OFF-PATH aux (for a
        subagent/cron principal) → ``background``.
    """
    delegated = int(getattr(agent, "_delegate_depth", 0) or 0) > 0
    noninteractive = _is_noninteractive_principal(agent)
    if aux_task is not None:
        # critical aux (on a live top-level turn's path) stays interactive (B1);
        # off-path aux (subagent/cron/headless principal) is background.
        return "background" if (delegated or noninteractive) else "interactive"
    if delegated or noninteractive:
        return "background"
    return "interactive"


# The messaging platforms that represent a LIVE, human-facing conversation whose turn a
# person is actively waiting on. A request whose source is NOT one of these (cron,
# headless CLI, systemd/docker service, background job) is non-interactive → background.
# This mirrors the codebase idiom `agent.platform or HERMES_SESSION_SOURCE (default cli)`
# used by background_review / conversation_compression, rather than matching the single
# literal "cron" (which false-negatived every headless run to interactive — Greptile #206).
_INTERACTIVE_PLATFORMS = frozenset({
    "discord", "telegram", "slack", "whatsapp", "imessage", "signal", "sms",
    "messenger", "instagram", "matrix", "teams", "line", "wechat", "webhook",
    "tui", "desktop", "web", "api",
})


def _is_noninteractive_principal(agent) -> bool:
    """True when the request's PRINCIPAL is not a live human-facing conversation — a
    cron/scheduled run, a headless CLI/service run, or anything whose source isn't a
    known interactive messaging surface. Resolved the same way the rest of the codebase
    resolves the session source: ``agent.platform`` first, else ``HERMES_SESSION_SOURCE``
    (default ``cli`` → non-interactive). Empty/unknown → treat as non-interactive
    (background) so scheduled/headless bursts can NOT claim interactive headroom."""
    src = (getattr(agent, "platform", "") or "").strip().lower()
    if not src:
        src = (os.environ.get("HERMES_SESSION_SOURCE", "") or "").strip().lower()
    if not src:
        return True   # no signal at all → non-interactive (safe: don't grant headroom)
    return src not in _INTERACTIVE_PLATFORMS


def _pool_lane_src(agent, aux_task=None) -> str:
    """Compact classifier-inputs header (x-hermes-lane-src) so the relay logs the raw
    signals (platform, delegate_depth, aux_task) alongside the lane verdict — the lane
    must be validatable against its inputs, not against itself. Routing-only, stripped
    upstream (never egresses)."""
    platform = (getattr(agent, "platform", "") or "").strip().lower()
    # reflect the SAME source the classifier used (platform, else HERMES_SESSION_SOURCE)
    # so the logged inputs actually explain the lane verdict, not a partial view.
    if not platform:
        platform = (os.environ.get("HERMES_SESSION_SOURCE", "") or "").strip().lower() or "-"
    dd = int(getattr(agent, "_delegate_depth", 0) or 0)
    task = aux_task if aux_task is not None else "-"
    return f"platform={platform};delegate_depth={dd};aux_task={task}"


# The api-proxy pool, canonical name `claude-apr` (the api-proxy multi-sub relay,
# api_mode `anthropic_messages`). A frozenset (not a bare `==`) so the gate is
# rename-proof — a stale single literal here silently killed affinity/lane
# stamping when the pool was renamed claude-app→claude-apr (caught 2026-07-08,
# #241). The legacy `claude-app` alias was fully retired 2026-07-08 (no live
# config/session/cron references remain), so it is no longer accepted here.
_POOL_AFFINITY_PROVIDERS = frozenset({"claude-apr"})


def _pool_affinity_headers(agent, aux_task=None) -> dict:
    """Return the routing-only headers for the claude relay POOL: the x-hermes-session
    affinity id AND the x-hermes-lane / x-hermes-lane-src lane classification.

    The pool uses the session id to pin a conversation to one subscription for
    prompt-cache preservation, and the lane to reserve headroom for interactive turns
    under contention (reset-weighted router + lanes, spec 2026-07-05). All are:
      * PER-REQUEST — read off the live ``agent`` at call-build time, so the session id
        rotates correctly when compaction mints a child id (NOT a static default_header,
        NOT the HERMES_SESSION_ID ContextVar which could go stale across the httpx
        worker-thread boundary → cross-conversation key bleed).
      * POOL-SCOPED — only stamped for ``claude-apr`` (the api-proxy pool, api_mode
        ``anthropic_messages``; legacy alias ``claude-app`` also accepted), so they are never sent to a direct Anthropic endpoint
        or any third party. The relay strips them before dispatching upstream (routing
        metadata on a loopback hop, no egress, no telemetry — satisfies the
        no-outbound-attribution rubric).

    SCOPE NOTE (Greptile #205): ``claude-bpr`` (the bridge pool, formerly ``claude-bpp``) resolves to api_mode
    ``chat_completions`` — a DIFFERENT branch of ``build_api_kwargs`` — and is a
    secondary failover surface with its own separate daemon that agents rarely route
    to as primary. It is deliberately OUT of scope here so this helper only claims what
    the anthropic_messages wiring actually stamps. Wiring the bpp path is a documented
    follow-up, not a silent gap.

    ``aux_task`` (when this is called from the auxiliary-client path) drives the lane
    criticality split; a main turn passes ``aux_task=None``.
    """
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    if provider not in _POOL_AFFINITY_PROVIDERS:
        return {}
    sid = getattr(agent, "session_id", None)
    out = {}
    if sid and isinstance(sid, str):
        out["x-hermes-session"] = sid
    out["x-hermes-lane"] = _pool_lane(agent, aux_task)
    out["x-hermes-lane-src"] = _pool_lane_src(agent, aux_task)
    return out
