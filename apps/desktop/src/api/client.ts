import { JsonRpcGatewayClient } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'

// Desktop startup fires a burst of read-only data calls (config, profiles,
// model info/options, cron) the moment the backend passes readiness. On a
// profile-heavy or remote install these can each take tens of seconds — e.g.
// /api/profiles runs list_profiles(), which does a recursive skill-tree walk
// per profile — so the 15s default (DEFAULT_FETCH_TIMEOUT_MS in hardening.ts)
// times out a backend that is alive-but-busy, surfacing as a spurious
// "Timed out connecting to Hermes backend" that hangs the UI (#48504).
//
// Give the boot burst a generous per-call timeout instead of raising the
// global default: interactive/runtime calls and the liveness poll (/api/status)
// keep the short default so a genuinely-dead backend is still detected fast.
export const STARTUP_REQUEST_TIMEOUT_MS = 60_000
const DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS = 30_000
// prompt.submit is effectively fire-and-forget: turn completion is signaled by
// stream / message.complete events, NOT by the RPC return. A long turn (MoA
// presets running references + aggregator in series, deep reasoning, large tool
// chains) can legitimately take minutes to ACK, so bounding the ack by the
// generic 30s default surfaces a false "request timed out" toast while the turn
// is still running and will succeed (issue #55024). Match the backend's
// agent-turn ceiling (agent.gateway_timeout = 1800s) so the ack timeout only
// ever fires when the turn itself would have been abandoned server-side.
export const PROMPT_SUBMIT_REQUEST_TIMEOUT_MS = 1_800_000

export class HermesGateway extends JsonRpcGatewayClient {
  constructor() {
    super({
      closedErrorMessage: 'Hermes gateway connection closed',
      connectErrorMessage: 'Could not connect to Hermes gateway',
      createRequestId: nextId => nextId,
      notConnectedErrorMessage: 'Hermes gateway is not connected',
      requestTimeoutMs: DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS
    })
  }
}

// Profile that profile-scoped REST settings (config/env/skills/tools/model/…)
// should target. Mirrors $activeGatewayProfile, pushed in from the store via
// setApiRequestProfile so this module needs no store import (avoids a cycle).
// Electron main consumes request.profile as request scope. Local calls whose
// REST handlers accept profile reuse the primary dashboard via ?profile=;
// unscoped handlers retain a profile backend. Remote overrides still route to
// their owning backend. Null → primary, so single-profile users are unaffected.
let _apiProfile: null | string = null

export function setApiRequestProfile(profile: null | string): void {
  _apiProfile = profile || null
}

export function profileScoped(profile?: null | string): { profile?: string } {
  const selected = profile === undefined ? _apiProfile : profile

  return selected ? { profile: selected } : {}
}

/** Profile that profile-scoped REST/WS calls should target (null → primary).
 *  Read-only twin of setApiRequestProfile for modules (e.g. voice playback)
 *  that build their own connection URLs and must stay on the same backend. */
export function getApiRequestProfile(): null | string {
  return _apiProfile
}

// Registry connection serving the active gateway (null → the local pool).
// Pushed from store/gateway's setActive — the single seam BOTH
// ensureGatewayProfile and ensureGatewayAgent funnel through — so WS calls
// that dial their own backend (pluginSocket) resolve it through the SAME
// source of truth those paths maintain for $connection. That makes the plugin
// socket follow registry-agent activations too, not just profile switches.
// Same no-store-import contract as _apiProfile (avoids a cycle).
let _apiConnectionId: null | string = null

export function setApiRequestConnection(connectionId: null | string): void {
  _apiConnectionId = connectionId || null
}

// Registry connection scope for a REST request. A registered remote gateway
// owns its own state.db — cron jobs and their run sessions live THERE — so
// requests for gateway-owned data must carry the connection id for the main
// process to route them to that host (hermes:api's registry branch). Null
// resolves to no tag, keeping single-source users byte-identical; explicit
// 'local' must remain tagged when the legacy primary points elsewhere.
export function connectionScoped(): { connectionId?: string } {
  return _apiConnectionId ? { connectionId: _apiConnectionId } : {}
}

/** Send a REST request to the renderer's active registry source. Request-level
 *  routing may override the active source for an explicitly-owned resource.
 *
 *  Helpers under `api/` go through here rather than calling the preload bridge
 *  directly, so the connection tag cannot be forgotten on a new one — with one
 *  exception. A capabilityScoped() helper must NOT: that scope says "the local
 *  pool" by omitting `connectionId` entirely, and an absent key cannot override
 *  the ambient tag spread underneath it, so a 'local' pin would silently route
 *  to whatever remote gateway happened to be active. Those helpers call the
 *  bridge directly and own their routing end to end. */
export function hermesApi<T>(request: HermesApiRequest): Promise<T> {
  return window.hermesDesktop.api<T>({ ...connectionScoped(), ...request })
}

// ── Capability scope: (connection, profile) routing for the Capabilities
// surface (skills / toolsets / MCP / hub / env / toolset config) ────────────
//
// A profile is not a machine-global name — it belongs to ONE gateway. The
// Capabilities surface can be pointed at any (connection, profile) pair
// (SkillsView's scope selector, Bot Mode's fixedProfile/fixedConnection), so
// its REST helpers accept either the legacy string form or an explicit scope
// object:
//
//   - `undefined` / string → the legacy profile path, PLUS the active registry
//     connection tag (connectionScoped, same contract the cron helpers adopted
//     in #87882). Without the tag, a window activated onto a registered remote
//     gateway read the LOCAL pool's skills/tools/MCP — the wrong machine.
//   - `{ connectionId, profile }` → explicit pin. `''`/`'local'` connection
//     ids mean the local pool and deliberately DROP the ambient connection
//     tag, so a local-profile pick made while a remote gateway is active still
//     routes to the local machine.
export type ProfileScope = undefined | null | string | { connectionId?: null | string; profile?: null | string }

export function capabilityScoped(scope?: ProfileScope): { connectionId?: string; profile?: string } {
  if (scope && typeof scope === 'object') {
    const profile = (scope.profile ?? '').trim()
    const connectionId = (scope.connectionId ?? '').trim()

    return {
      ...(profile ? { profile } : {}),
      ...(connectionId && connectionId !== 'local' ? { connectionId } : {})
    }
  }

  return { ...profileScoped(scope), ...connectionScoped() }
}

/** Stable cache-key for a capability scope: `profile` for the local/legacy
 *  path, `connectionId::profile` for an explicit remote pin. Mirrors
 *  normalizeProfileKey for plain strings so existing keys stay byte-identical. */
export function profileScopeKey(scope?: ProfileScope): string {
  if (scope && typeof scope === 'object') {
    const profile = (scope.profile ?? '').trim() || 'default'
    const connectionId = (scope.connectionId ?? '').trim()

    return connectionId && connectionId !== 'local' ? `${connectionId}::${profile}` : profile
  }

  return (scope ?? '').trim() || 'default'
}

/** Registry connection id that connection-scoped WS calls should target
 *  (null → the local pool). Read-only twin of setApiRequestConnection. */
export function getApiRequestConnection(): null | string {
  return _apiConnectionId
}
