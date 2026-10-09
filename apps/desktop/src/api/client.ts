import {
  type GatewayEvent,
  type GatewayEventName,
  JsonRpcGatewayClient,
  type ServerRequest,
  type ServerRequestHandler
} from '@hermes/shared'
import { map, type MapStore } from 'nanostores'

import type { HermesApiRequest } from '@/global'

import { CANONICAL_GATEWAY_PROTOCOL, CanonicalDesktopProtocol } from './canonical-protocol'

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

export const GATEWAY_NOT_CONNECTED_MESSAGE = 'Hermes gateway is not connected'

// Canonical shared prompts (`gateway/session_pending_controls.py`) travel as
// `approval.request` / `clarify.request` EVENTS keyed by `prompt_id` and are
// answered through `approval.respond` / `clarify.respond`. The renderer only
// knows server→client REQUESTS (#110521), so each prompt event is delivered to
// the same `onRequest` registry as a request whose `respond` issues the RPC.
const CANONICAL_PROMPT_EVENTS: Record<string, 'approval' | 'clarify'> = { 'approval.request': 'approval', 'clarify.request': 'clarify' }

const ATTACH_REQUIRED = new Set(['prompt.submit', 'approval.respond', 'clarify.respond', 'session.interrupt', 'prompt.cancel'])
// A branch is a CAS mutation on the PARENT: an un-attached parent (right-click on a sidebar row
// that was never opened) has no cached revision, so attach it first to learn one.
const PARENT_ATTACH_REQUIRED = new Set(['session.branch_stored', 'session.branch_whole'])

// Canonical-only identity keys. The legacy `hermes serve` contract refuses an unknown key as
// version skew (4000), so they are stripped on a non-canonical dial. `prompt.submit` keeps its
// `submission_id` on purpose: the submit path retries identityless on that exact refusal and
// records the send as legacy-attempted, which a silent strip here would hide.
const LEGACY_STRIP: Record<string, string[]> = { 'session.create': ['request_id'], 'session.branch_stored': ['request_id'] }

function legacyParams(method: string, params: Record<string, unknown>): Record<string, unknown> {
  const strip = LEGACY_STRIP[method]

  if (!strip || !strip.some(key => key in params)) { return params }

  return Object.fromEntries(Object.entries(params).filter(([key]) => !strip.includes(key)))
}

export class HermesGateway extends JsonRpcGatewayClient {
  private canonical = false
  private readonly protocol = new CanonicalDesktopProtocol()
  private readonly promptHandlers = new Set<ServerRequestHandler>()
  // Delivered canonical prompt id → its session (the fenced respond needs both).
  private readonly deliveredPrompts = new Map<string, string>()

  override onRequest(handler: ServerRequestHandler): () => void {
    this.promptHandlers.add(handler)
    const off = super.onRequest(handler)

    return () => {
      this.promptHandlers.delete(handler)
      off()
    }
  }

  private deliverCanonicalPrompt(event: { type: string; session_id?: string; payload?: unknown }, replayed: boolean) {
    const kind = CANONICAL_PROMPT_EVENTS[event.type]
    const p = event.payload as Record<string, unknown> | undefined
    const sid = event.session_id

    if (!kind || !p || !sid || typeof p.prompt_id !== 'string') { return }
    const id = p.prompt_id

    if (this.deliveredPrompts.has(id)) { return }
    this.deliveredPrompts.set(id, sid)
    const choices = Array.isArray(p.choices) ? p.choices : []

    // The question card reads one `questions[]` shape. A canonical clarify prompt is one
    // question (the shared gateway asks a batch one card at a time), so its prompt id doubles
    // as the qid that `clarify.lock` answers with (see `request` below).
    const params: Record<string, unknown> = kind === 'clarify'
      ? { session_id: sid, answers: p.answers,
          questions: Array.isArray(p.questions) ? p.questions : [{ qid: id, question: p.question, choices, multi_select: p.multi_select }] }
      : { session_id: sid, request_id: id, command: p.command, description: p.description, choices,
          allow_permanent: choices.includes('always'), edit: p.edit }

    let settled = false

    const request: ServerRequest = {
      id,
      method: kind,
      params,
      replayed,
      respond: result => {
        if (settled) { return }
        settled = true
        // A clarify reply without an answer (skip, Stop, cancel) is the empty answer the
        // authority records as skipped; `clarify.respond` refuses a missing field.
        const answer = kind === 'clarify'
          ? { answer: typeof result.answer === 'string' ? result.answer : '' }
          : { choice: result.choice }
        void this.request(`${kind}.respond`, { session_id: sid, request_id: id, ...answer }).catch(() => undefined)
      },
      fail: () => { settled = true }
    }

    for (const handler of this.promptHandlers) {
      if (handler(request) !== false) { return }
    }
  }

  override on<K extends GatewayEventName>(type: K, handler: (event: GatewayEvent<K>) => void): () => void {
    return super.on<K>(type, event => {
      // Named listeners run before wildcard listeners in the shared client.
      // Normalize before either kind sees the prompt, including replay delivery.
      if (this.canonical) { this.protocol.event(event) }
      handler(event)
    })
  }

  override async connect(wsUrl: string): Promise<void> {
    this.canonical = new URL(wsUrl).searchParams.has('native_dial')
    this.attached.clear()

    return super.connect(wsUrl)
  }

  // Sessions attached over THIS socket. An authority subscription is per
  // transport: when routing moves a session to another socket (a profile split,
  // a redial) the new socket must resume before it may submit or respond.
  private attached = new Set<string>()

  override async request<T>(method: string, params: Record<string, unknown> = {}, timeoutMs?: number, signal?: AbortSignal): Promise<T> {
    if (!this.canonical) { return super.request<T>(method, legacyParams(method, params), timeoutMs, signal) }

    // The card locks each question with `clarify.lock`; a canonical prompt is a single
    // question, so its one lock is the generation-fenced `clarify.respond` (null = skipped).
    const promptSession = method === 'clarify.lock' ? this.deliveredPrompts.get(String(params.request_id ?? '')) : undefined

    if (promptSession) {
      const answer = typeof params.answer === 'string' ? params.answer : ''

      return this.request<T>('clarify.respond', { session_id: promptSession, request_id: params.request_id, answer }, timeoutMs, signal)
    }

    await this.attachForRequest(method, params)

    const prepared = this.protocol.prepare(method, params)
    const wireMethod = this.protocol.wire(method, prepared)

    try {
      const result = await super.request<T>(wireMethod, prepared, timeoutMs, signal)

      // A settle follow-up (compress -> resume) inherits the caller's deadline and cancellation.
      const followUp = (m: string, p: Record<string, unknown>) => this.request(m, p, timeoutMs, signal)
      const settled = this.protocol.settle(method, prepared, this.protocol.result(method, prepared, result), followUp) as T

      if (method === 'session.resume' || method === 'session.create' || method === 'session.activate') {
        this.adoptAttachedSnapshot(settled as { session_id?: string; prompts?: Array<Record<string, unknown>> })
      }

      return settled
    } catch (error) {
      this.protocol.failure(prepared, error)
      throw error
    }
  }

  // Resume the session (and, for branch-like methods, its parent) on THIS socket before a
  // method that needs an authority subscription here.
  private async attachForRequest(method: string, params: Record<string, unknown>): Promise<void> {
    const sid = typeof params.session_id === 'string' ? params.session_id : null

    if (sid && ATTACH_REQUIRED.has(method) && !this.attached.has(sid)) {
      await this.request('session.resume', { session_id: sid, defer_history: true, omit_messages: true, ...(params.profile ? { profile: params.profile } : {}) })
    }

    const parent = PARENT_ATTACH_REQUIRED.has(method) ? params.parent_session_id ?? params.session_id : null

    if (typeof parent === 'string' && parent && !this.attached.has(parent)) {
      await this.request('session.resume', { session_id: parent, defer_history: true, omit_messages: true, ...(params.profile ? { profile: params.profile } : {}) })
    }
  }

  private adoptAttachedSnapshot(settled: { session_id?: string; prompts?: Array<Record<string, unknown>> }): void {
    // Prompts still open on the authority re-deliver like `open_requests` after a reconnect.
    const sid = settled.session_id

    if (sid) { this.attached.add(sid) }

    for (const prompt of (settled.prompts ?? [])) {
      this.deliverCanonicalPrompt({ type: `${prompt.kind}.request`, session_id: sid, payload: prompt }, true)
    }
  }

  constructor() {
    super({
      closedErrorMessage: 'Hermes gateway connection closed',
      connectErrorMessage: 'Could not connect to Hermes gateway',
      createRequestId: nextId => nextId,
      notConnectedErrorMessage: GATEWAY_NOT_CONNECTED_MESSAGE,
      // The channel already answered -32603; surface the crash in devtools like the dial-failure sink.
      onRequestHandlerError: (error, request) =>
        console.error(`[gateway] server request handler crashed for ${request.method} (${request.id}):`, error),
      // The channel already answered -32601; note the missing registry in devtools.
      onUnhandledRequest: request =>
        console.warn(`[gateway] Hermes Desktop has no server-request registry for ${request.method} (${request.id})`),
      requestTimeoutMs: DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS,
      socketFactory: url => {
        const parsed = new URL(url)

        if (!parsed.searchParams.has('native_dial')) {return new WebSocket(url)}
        const ticket = parsed.searchParams.get('ticket')

        if (!ticket) {throw new Error('Native gateway requires a fresh private ticket')}
        parsed.searchParams.delete('ticket')

        return new WebSocket(parsed.toString(), [CANONICAL_GATEWAY_PROTOCOL, `hermes-gateway-ticket.${ticket}`])
      }
    })
    this.onEvent(event => {
      if (!this.canonical) { return }
      this.protocol.event(event)

      if (event.type in CANONICAL_PROMPT_EVENTS) {
        this.deliverCanonicalPrompt(event, false)
      } else if (event.type.endsWith('.settled')) {
        const id = (event.payload as { prompt_id?: unknown } | undefined)?.prompt_id

        if (typeof id === 'string') { this.deliveredPrompts.delete(id) }
      }
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
interface ApiRequestScope {
  profile: string | null
  connectionId: string | null
}

// This is the request authority, not a second copy in a presentation store.
export const $apiRequestScope: MapStore<ApiRequestScope> = map<ApiRequestScope>({ profile: null, connectionId: null })

export function setApiRequestProfile(profile: null | string): void {
  $apiRequestScope.setKey('profile', profile || null)
}

// An explicit scope (string or object, not `undefined`/`null`) is a user
// pointing a scope selector (Settings "Applies to", Capabilities, Messaging)
// at another profile — a visible action whose cold dial may take the pool's
// reserved foreground slot (#111651). The ambient path and the deliberate
// `null` → primary path stay untagged (main's background default) so
// hydration cannot consume that slot. The tag rides on the scope helper itself
// so no api/ helper can carry a scope without it.
export function profileScoped(profile?: null | string): { priority?: 'foreground'; profile?: string } {
  const selected = profile === undefined ? $apiRequestScope.get().profile : profile

  return {
    ...(selected ? { profile: selected } : {}),
    ...(profile == null ? {} : { priority: 'foreground' as const })
  }
}

/** A session's OWNER as a request scope: a profile belongs to ONE gateway, so
 *  a Bot on another connection is (its connection, its profile) — never the
 *  active connection with the Bot's profile name. Missing halves fall back to
 *  the ambient scope; an explicit connection — `'local'` included — overrides
 *  the ambient tag `hermesApi` spreads underneath (as capabilityScoped does). */
export interface OwnerScope {
  connectionId?: null | string
  profile?: null | string
}

export function ownerScoped(owner?: OwnerScope): { connectionId?: string; priority?: 'foreground'; profile?: string } {
  return {
    ...profileScoped(owner?.profile || undefined),
    ...(owner?.connectionId ? { connectionId: owner.connectionId } : {})
  }
}

/** An owner resolved ONCE for one operation (a recording, a voice conversation).
 *  `null` halves mean "untagged": the primary profile / the connection an
 *  untagged request lands on. They are never re-read from the ambient scope
 *  later, so a gateway/profile switch mid-operation cannot move its tail
 *  (release, transcription) to another backend. */
export interface ResolvedOwner {
  connectionId: null | string
  profile: null | string
}

/** Fill an owner's missing halves from the ambient scope, now. */
export function resolveOwnerNow(owner?: OwnerScope): ResolvedOwner {
  const ambient = $apiRequestScope.get()

  return {
    connectionId: owner?.connectionId || ambient.connectionId || null,
    profile: owner?.profile || ambient.profile || null
  }
}

/** `hermesApi` for a resolved owner: its tags are sent verbatim, with no
 *  ambient connection spread underneath. An untagged half stays untagged. A
 *  named profile is always explicit here, so it carries the foreground
 *  priority `profileScoped` gives explicit profiles (voice is user-driven). */
export function hermesApiAs<T>(owner: ResolvedOwner, request: HermesApiRequest): Promise<T> {
  return window.hermesDesktop.api<T>({
    ...(owner.connectionId ? { connectionId: owner.connectionId } : {}),
    ...(owner.profile ? { priority: 'foreground' as const, profile: owner.profile } : {}),
    ...request
  })
}

/** Profile that profile-scoped REST/WS calls should target (null → primary).
 *  Read-only twin of setApiRequestProfile for modules (e.g. voice playback)
 *  that build their own connection URLs and must stay on the same backend. */
export function getApiRequestProfile(): null | string {
  return $apiRequestScope.get().profile
}

// Registry connection serving the active gateway (null → the local pool).
// Pushed from store/gateway's setActive — the single seam BOTH
// ensureGatewayProfile and ensureGatewayAgent funnel through — so WS calls
// that dial their own backend (pluginSocket) resolve it through the SAME
// source of truth those paths maintain for $connection. That makes the plugin
// socket follow registry-agent activations too, not just profile switches.
// Same no-store-import contract as profile scope (avoids a cycle).

export function setApiRequestConnection(connectionId: null | string): void {
  $apiRequestScope.setKey('connectionId', connectionId || null)
}

// Registry connection scope for a REST request. A registered remote gateway
// owns its own state.db — cron jobs and their run sessions live THERE — so
// requests for gateway-owned data must carry the connection id for the main
// process to route them to that host (hermes:api's registry branch). Null
// resolves to no tag, keeping single-source users byte-identical; explicit
// 'local' must remain tagged when the legacy primary points elsewhere.
export function connectionScoped(): { connectionId?: string } {
  const connectionId: string | null = $apiRequestScope.get().connectionId

  return connectionId ? { connectionId } : {}
}

// Whether the window's primary connection is the local pool. Pushed from
// store/session's setConnection (same no-store-import contract as profile scope)
// so api/ helpers can name the backend an UNTAGGED request lands on without
// importing the heavy session store — which would close a module cycle
// through @/hermes.
let _apiLocalMode = false

export function setApiRequestLocalMode(local: boolean): void {
  _apiLocalMode = local
}

/** The connection an ambient (untagged) request is served by: the registry
 *  tag when one is active, else `'local'` for the local pool. Identity only —
 *  never send this as a request pin (an explicit `'local'` bypasses Electron's
 *  legacy per-profile remote overrides). */
export function ambientOwnerConnectionId(): string | undefined {
  return $apiRequestScope.get().connectionId ?? (_apiLocalMode ? 'local' : undefined)
}

/** Send a REST request to the renderer's active registry source. Request-level
 *  routing may override the active source for an explicitly-owned resource.
 *
 *  Helpers under `api/` go through here rather than calling the preload bridge
 *  directly, so the connection tag cannot be forgotten on a new one.
 *  capabilityScoped() now emits an explicit `connectionId` for EVERY object
 *  pin — `'local'` included — so a pin always overrides the ambient tag spread
 *  underneath it. (It used to omit the key for 'local', which made the pin
 *  unable to beat the ambient tag; helpers then had to bypass this wrapper.) */
export function hermesApi<T>(request: HermesApiRequest): Promise<T> {
  return window.hermesDesktop.api<T>({ ...connectionScoped(), ...request })
}

// ── Capability scope: (connection, profile) routing for the Capabilities
// surface (skills / toolsets / MCP / hub / env / toolset config) ────────────
//
// A profile is not a machine-global name — it belongs to ONE gateway. The
// Capabilities surface can be pointed at any (connection, profile) pair
// (CapabilitiesView's scope selector, Bot Mode's fixedProfile/fixedConnection), so
// its REST helpers accept either the legacy string form or an explicit scope
// object:
//
//   - `undefined` / string → the legacy profile path, PLUS the active registry
//     connection tag (connectionScoped, same contract the cron helpers adopted
//     in #87882). Without the tag, a window activated onto a registered remote
//     gateway read the LOCAL pool's skills/tools/MCP — the wrong machine.
//   - `{ connectionId, profile }` → explicit pin. A non-empty connection id —
//     `'local'` INCLUDED — is sent through so Electron's registry resolver
//     owns the routing. Dropping the `'local'` pin (the pre-#91564 behavior,
//     when an absent id always meant the local pool) silently re-routes a
//     "This device" pick to the registry PRIMARY once that primary is a
//     remote/cloud/ssh gateway: the v1 fallback route treats a remote registry
//     primary as global-remote, so the explicit pin is the ONLY way back to
//     this machine (see apiRequestRegistryConnectionId in Electron main).
export type ProfileScope = undefined | null | string | { connectionId?: null | string; profile?: null | string }

export function capabilityScoped(scope?: ProfileScope): {
  connectionId?: string
  priority?: 'foreground'
  profile?: string
} {
  if (scope && typeof scope === 'object') {
    const profile = (scope.profile ?? '').trim()
    const connectionId = (scope.connectionId ?? '').trim()

    return {
      ...(profile ? { profile } : {}),
      ...(connectionId ? { connectionId } : {}),
      priority: 'foreground'
    }
  }

  return { ...profileScoped(scope), ...connectionScoped() }
}

/** Stable cache-key for a capability scope: `profile` for the ambient/legacy
 *  path, `connectionId::profile` for ANY explicit pin — `local` included. An
 *  explicit "This device" pick and the ambient path are no longer guaranteed
 *  to hit the same backend (a remote registry PRIMARY makes the ambient path
 *  remote), so sharing the bare-profile cache row between them painted one
 *  machine's config under the other's scope (AGENTS.md scope-in-key rule). */
export function profileScopeKey(scope?: ProfileScope): string {
  if (scope && typeof scope === 'object') {
    const profile = (scope.profile ?? '').trim() || 'default'
    const connectionId = (scope.connectionId ?? '').trim()

    return connectionId ? `${connectionId}::${profile}` : profile
  }

  return (scope ?? '').trim() || 'default'
}

/** Registry connection id that connection-scoped WS calls should target
 *  (null → the local pool). Read-only twin of setApiRequestConnection. */
export function getApiRequestConnection(): null | string {
  return $apiRequestScope.get().connectionId
}

// ── Session-owner pin for session-scoped REST reads (#125372) ──────────────
//
// A read of /api/sessions/{id}[/messages|/timeline|/messages/around] only
// means anything on the backend that OWNS the row. The store already resolves
// that owner (knownOwnerForSession: tile route → persisted hint → tagged row)
// and pushes the resolver here — same no-store-import seam as
// setApiRequestProfile — so read helpers stop riding the WINDOW's ambient
// connection onto a host that answers 404 "Session not found".

export interface SessionReadOwnerRoute {
  connectionId?: null | string
  profile?: null | string
  targetProfile?: null | string
}

export type SessionOwnerResolver = (sessionId: string) => SessionReadOwnerRoute | null | string | undefined

let _sessionOwnerResolver: SessionOwnerResolver | null = null

export function setSessionOwnerResolver(resolver: SessionOwnerResolver | null): void {
  _sessionOwnerResolver = resolver
}

/** The connection pin a session-scoped READ must carry for `id`, or {} when
 *  the caller already pinned a connection, no exact owner is known, or the
 *  owner is the ambient connection. Pins 'local' too (the only way back to
 *  this device when the primary is a remote registry source). When the caller
 *  named no profile, the owner's backend-facing profile rides along: the
 *  answering host resolves ?profile= against ITS OWN profiles. */
export function sessionReadOwnerPin(id: string, profile?: ProfileScope): { connectionId?: string; profile?: string } {
  if (profile && typeof profile === 'object' && String(profile.connectionId ?? '').trim()) {
    return {}
  }

  const owner = _sessionOwnerResolver?.(id)

  if (!owner || typeof owner === 'string') {
    return {}
  }

  const connectionId = String(owner.connectionId ?? '').trim()

  if (!connectionId || connectionId === (ambientOwnerConnectionId() ?? 'local')) {
    return {}
  }

  const ownerProfile = String(owner.targetProfile || owner.profile || '').trim()

  return {
    connectionId,
    ...(profile == null && ownerProfile ? { profile: ownerProfile } : {})
  }
}
