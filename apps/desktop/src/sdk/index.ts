/**
 * @hermes/plugin-sdk — THE plugin language. The vscode-module model: plugin
 * authors import exactly one module and get everything — they never touch
 * `@/…` internals (lint-fenced) and never need codebase access.
 *
 * Two delivery modes, one surface:
 *  - bundled (`src/plugins/<name>/`): the import resolves here via alias;
 *  - runtime-fetched (plugin host, next phase): the loader injects this same
 *    object as `window.__HERMES_PLUGIN_SDK__` and maps the import to it, so a
 *    published plugin builds against the types with the SDK marked external.
 *
 * Capability tiers (WoW-style):
 *  - `host.state.*` — READONLY app state (nanostore atoms; `.get()` or
 *    subscribe; `useValue` in React).
 *  - `host.*` actions — curated, safe verbs (toast, haptic).
 *  - `host.request` — the gateway JSON-RPC door; the plugin's real power,
 *    and the future seam for per-plugin capability grants.
 *  - `ui.*` — the design language, so plugin UI looks native by default.
 */

import { atom, computed, type ReadableAtom } from 'nanostores'
import type { ReactNode } from 'react'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { openSession, type OpenSessionIntent } from '@/app/open-session'
import type { ClientSessionState } from '@/app/types'
import {
  $narrowViewport,
  $newSessionTabAction,
  $paneVisible,
  registerPaneCloser,
  removeTreePane,
  revealTreePane
} from '@/components/pane-shell/tree/store'
import {
  $workspaceMode,
  $workspaceOwnerKey,
  setWorkspaceScope as publishWorkspaceScope,
  type WorkspaceNewSessionTarget
} from '@/components/pane-shell/workspace-scope'
import { onGatewayEvent } from '@/contrib/events'
import { registry } from '@/contrib/registry'
import type { WorkspaceMode } from '@/contrib/types'
import { deleteProfile, getLogs, getStatus, type HermesGateway } from '@/hermes'
import {
  $gateway,
  activeGatewayConnectionId,
  openGatewayForAgent,
  openGatewayForProfile,
  requestGatewayForAgent,
  requestGatewayForProfile,
  retainGatewayForAgent,
  retainGatewayForRelay,
  retireLocalProfileGateways
} from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $gatewaySwapTarget,
  $profiles,
  ensureGatewayAgent,
  ensureGatewayProfile,
  newSessionInAgent,
  newSessionInProfile,
  normalizeProfileKey,
  refreshProfiles,
  selectProfile,
  setActiveProfile,
  setShowAllProfiles
} from '@/store/profile'
import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $currentModel,
  $gatewayState,
  $messages,
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHints,
  rememberedSessionProfile,
  requestSessionResume,
  sessionMatchesStoredId,
  setResumeExhaustedSessionId,
  setSessionOwnerHint
} from '@/store/session'
import {
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $sessionStates,
  $sessionTiles
} from '@/store/session-states'
import { runGatewayRestart } from '@/store/system-actions'
import type { UsageStats } from '@/types/hermes'

import { planPluginOpenSession } from './plugin-open-session-plan'

// -- state: readonly views over the app's live atoms -------------------------

const readonlyAtom = <T>(atomLike: ReadableAtom<T>): ReadableAtom<T> => atomLike

/**
 * Turn flag for the FOCUSED chat — same semantics as the statusbar's busy
 * pulse. While the focused surface is the primary workspace (or a draft with
 * no runtime slice yet) this reads the primary view, which itself falls back
 * to the global draft atoms. Once a session TILE holds focus, the tile's own
 * state slice is authoritative — a background session can never leak in.
 */
const focusedTurnFlag = (
  select: (state: ClientSessionState) => boolean,
  $primary: ReadableAtom<boolean>
): ReadableAtom<boolean> =>
  computed(
    [$focusedStoredSessionId, $selectedStoredSessionId, $focusedSessionState, $primary],
    (focused, selected, state, primary) =>
      !focused || focused === selected ? primary : Boolean(state && select(state))
  )

const $focusedBusy = focusedTurnFlag(state => state.busy, PRIMARY_SESSION_VIEW.$busy)

const $focusedAwaitingResponse = focusedTurnFlag(
  state => state.awaitingResponse,
  PRIMARY_SESSION_VIEW.$awaitingResponse
)

export interface PluginFocusedSessionOwner {
  connectionId: string
  profile: string
}

/**
 * Connection-qualified owner of the FOCUSED chat. The gateway-routing atom
 * (`$activeGatewayProfile`) answers "which backend is the live socket homed
 * on" — but tab/tile focus moves without swapping the socket, and a cold
 * start can restore a route into a session the booting gateway doesn't own.
 * Any per-bot readout must follow the chat the user is LOOKING AT, so this
 * resolves the focused stored session to a unique immutable owner hint or a
 * unique connection-qualified aggregated row. Ambiguous or unresolved focused
 * ids fail closed with null; only a draft/no focused id uses the active gateway
 * owner. `focusedSessionProfile` remains the profile-only compatibility ladder.
 */
const $focusedSessionOwner = computed(
  [$focusedStoredSessionId, $sessions, $activeGatewayProfile, $connection],
  (focused, sessions, activeProfile, connection): PluginFocusedSessionOwner | null => {
    const activeConnectionId = String(connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')).trim()

    const fallback = {
      connectionId: activeConnectionId,
      profile: normalizeProfileKey(activeProfile)
    }

    if (!focused) {
      return fallback
    }

    const hints = getSessionOwnerHints(focused)

    if (hints.length === 1) {
      return {
        connectionId: hints[0].connectionId,
        profile: normalizeProfileKey(hints[0].profile)
      }
    }

    if (hints.length > 1) {
      return null
    }

    const owners = new Map<string, PluginFocusedSessionOwner>()

    for (const row of sessions.filter(session => sessionMatchesStoredId(session, focused))) {
      const connectionId = String(row.connection_id || '').trim()
      const profile = normalizeProfileKey(row.profile)

      if (connectionId) {
        owners.set(`${connectionId}::${profile}`, { connectionId, profile })
      }
    }

    return owners.size === 1 ? [...owners.values()][0] : null
  }
)

const $focusedSessionProfile = computed(
  [$focusedSessionOwner, $focusedStoredSessionId, $sessions, $activeGatewayProfile],
  (owner, focused, sessions, activeProfile) =>
    owner?.profile || rememberedSessionProfile(sessions, focused, activeProfile)
)

export interface PluginProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  /** Desktop profile used to select the connection route. */
  profile: string
  /** Backend Hermes profile served by that route. */
  targetProfile: string
}

/** Window geometry + the app's responsive posture, one readonly rect. */
export interface ViewportRect {
  width: number
  height: number
  /** Below the app's sidebar-collapse breakpoint (rails become overlays). */
  narrow: boolean
}

const readViewport = (): ViewportRect => ({
  width: typeof window === 'undefined' ? 0 : window.innerWidth,
  height: typeof window === 'undefined' ? 0 : window.innerHeight,
  narrow: $narrowViewport.get()
})

/** Runtime session id → mid-turn. Not gateway socket state. */
const $busyBySession = computed($sessionStates, states => {
  const map: Record<string, boolean> = {}

  for (const [id, state] of Object.entries(states)) {
    map[id] = Boolean(state.busy)
  }

  return map
})

const $viewport = atom<ViewportRect>(readViewport())

async function requestPluginProfile<T>(
  route: PluginProfileRoute | string,
  method: string,
  params: Record<string, unknown>
): Promise<T> {
  if (typeof route !== 'string') {
    if (!route.connectionId.trim() || !route.profile.trim() || !route.targetProfile.trim()) {
      throw new Error('Profile route must include connectionId, profile, and targetProfile')
    }

    return requestGatewayForAgent<T>(route.connectionId, route.profile, method, params)
  }

  const getAgentRoster = window.hermesDesktop?.getAgentRoster

  if (!getAgentRoster) {
    return requestGatewayForProfile<T>(route, method, params)
  }

  const roster = await getAgentRoster()
  const profile = route.trim() || 'default'
  const soleLocalSource = roster.sources.length === 1 && roster.sources[0]?.kind === 'local'

  // The string overload is compatibility-only. A sole local registry is the
  // one topology where a profile name is intrinsically unambiguous, even when
  // its live enumeration transiently failed. Any additional source requires a
  // descriptor because an undialed/unreachable source may expose the same name.
  if (soleLocalSource) {
    return requestGatewayForProfile<T>(profile, method, params)
  }

  throw new Error(
    `Profile "${profile}" requires a route descriptor from host.profileRoutes(); profile-only routing is limited to legacy/local profiles.`
  )
}

/** Re-read Electron's current registry before retrying an exact-owner wake.
 *  A route that was removed or replaced while the first hydration wait ran is
 *  no longer authority to touch that backend, even when its labels still look
 *  identical. */
async function pluginRouteStillRegistered(route: PluginProfileRoute): Promise<boolean> {
  const getProfileRoutes = window.hermesDesktop?.getProfileRoutes

  if (!getProfileRoutes) {
    return false
  }

  try {
    const routes = await getProfileRoutes($profiles.get().map(profile => profile.name))

    return routes.some(
      candidate =>
        candidate.connectionId === route.connectionId &&
        candidate.profile === route.profile &&
        candidate.targetProfile === route.targetProfile
    )
  } catch {
    return false
  }
}

if (typeof window !== 'undefined') {
  const refresh = () => $viewport.set(readViewport())
  window.addEventListener('resize', refresh)
  $narrowViewport.listen(refresh)
}

/** Live usage of the FOCUSED session, projected out of the streamed session
 *  state — the same readout the core statusbar's context chip paints. */
const $focusedUsage = computed($focusedSessionState, state => state?.usage ?? null)

const $activeConnectionId = computed($connection, connection => {
  if (!connection) {
    return null
  }

  if (connection.connectionId) {
    return connection.connectionId
  }

  // mode:'local' used to report null, which made Bot Mode fall back to the
  // registry primary (often an SSH box) and treat Spark as the active source
  // while this window was actually local.
  return connection.mode === 'local' ? 'local' : null
})

const DEFAULT_SESSION_HYDRATION_TIMEOUT_MS = 20_000
let openSessionGeneration = 0

export interface PluginOpenSessionOptions {
  awaitHydration?: boolean
  expectHistory?: boolean
  hydrationTimeoutMs?: number
  intent?: OpenSessionIntent
  keepAllProfilesScope?: boolean
  profile?: null | string
  route?: PluginProfileRoute
  workspaceMode?: WorkspaceMode
  workspaceOwnerKey?: string
  /** A cold profile backend can lose the hydration-timeout race once and still
   *  be fine on a second try. When set, a hydration timeout is retried
   *  internally before it reaches the caller or arms the core stranded-session
   *  overlay ($resumeExhaustedSessionId) — a caller-side retry can't do this
   *  itself because only this SDK layer sees $resumeExhaustedSessionId. */
  retryHydrationTimeoutOnce?: boolean
  tabTitle?: string
}

export interface PluginNewChatOptions {
  workspaceMode?: WorkspaceMode
  workspaceOwnerKey?: string
}

function waitForFocusedSessionHydration({
  expectHistory,
  generation,
  isCurrent,
  profile,
  requireActiveProfile,
  storedSessionId,
  timeoutMs
}: {
  expectHistory: boolean
  generation: number
  isCurrent?: () => boolean
  profile: string
  requireActiveProfile: boolean
  storedSessionId: string
  timeoutMs: number
}): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false
    const unbinds: Array<() => void> = []
    let timer: number | undefined

    const finish = (error?: Error) => {
      if (settled) {
        return
      }

      settled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }

      for (const unbind of unbinds) {
        unbind()
      }

      if (error) {
        reject(error)
      } else {
        resolve()
      }
    }

    const check = () => {
      if (generation !== openSessionGeneration || (isCurrent && !isCurrent())) {
        finish(new Error('Session open was superseded by a newer selection.'))

        return
      }

      const profileMatches = !requireActiveProfile || normalizeProfileKey($activeGatewayProfile.get()) === profile
      const mainMatches = $selectedStoredSessionId.get() === storedSessionId
      const storedTile = $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)
      const tileMatches = $focusedStoredSessionId.get() === storedSessionId || Boolean(storedTile)
      const focusedTileMatches = $focusedStoredSessionId.get() === storedSessionId
      const tileRuntimeId = focusedTileMatches ? $focusedRuntimeId.get() : (storedTile?.runtimeId ?? null)

      const tileState = focusedTileMatches
        ? $focusedSessionState.get()
        : tileRuntimeId
          ? $sessionStates.get()[tileRuntimeId]
          : undefined

      const runtimeReady = mainMatches ? Boolean($activeSessionId.get()) : tileMatches ? Boolean(tileRuntimeId) : false

      const historyPainted = mainMatches
        ? Boolean($messages.get().length)
        : tileMatches
          ? Boolean(tileState?.messages.length)
          : false

      // Paint-first hydration: for a history-bearing chat, the wake is DONE
      // the moment the persisted transcript is painted on the right session —
      // the REST prefetch delivers it seconds after the profile backend's
      // HTTP comes up, while the full runtime resume (agent build, MCP
      // discovery, skill load) keeps warming in the background and binds the
      // composer when it lands. Gating on runtimeReady serialized the wake
      // behind that whole boot: on a cold multi-profile start the 20s budget
      // regularly lost the race on slower machines and surfaced as "errors
      // waking up bots" even though the transcript had been available almost
      // immediately. Only an expected-EMPTY chat still waits for the runtime
      // — with no transcript to paint, a bound runtime is the only proof the
      // surface is real rather than a stuck loader.
      const hydrated = expectHistory ? historyPainted : runtimeReady

      if (profileMatches && (mainMatches || tileMatches) && hydrated) {
        finish()
      }
    }

    unbinds.push($activeGatewayProfile.listen(check))
    unbinds.push($activeConnectionId.listen(check))
    unbinds.push($selectedStoredSessionId.listen(check))
    unbinds.push($activeSessionId.listen(check))
    unbinds.push($messages.listen(check))
    unbinds.push($focusedStoredSessionId.listen(check))
    unbinds.push($focusedRuntimeId.listen(check))
    unbinds.push($focusedSessionState.listen(check))
    unbinds.push($sessionTiles.listen(check))
    unbinds.push($sessionStates.listen(check))
    unbinds.push($workspaceMode.listen(check))
    unbinds.push($workspaceOwnerKey.listen(check))

    timer = window.setTimeout(() => {
      finish(new Error(`Timed out loading ${profile}'s session history.`))
    }, timeoutMs)

    check()
  })
}

// Wait for a profile switch, but never longer than the wake budget.
//
// ensureGatewayProfile awaits the store's dial, and HermesGateway.connect() has
// no dial timeout of its own: a backend that accepts the socket and then never
// completes the handshake leaves this promise pending for the life of the
// window. That is not merely a slow open. waitForFocusedSessionHydration arms
// the only timer on this path, and it is armed AFTER this await returns - so an
// unbounded activation means the wake never settles at all, and the pane wedges
// with no error, no Retry and no timeout (#89556: `ws accepted` in the gateway
// log with no matching `ws closed`).
//
// The activation gets its OWN budget rather than sharing the hydration one. A
// cold profile backend can legitimately spend most of the hydration budget
// painting a large transcript - that race is already tight enough to lose
// (#89617) - so charging activation to the same clock would turn a wedge into a
// regression. The trade is that a wake that is slow in BOTH phases can now take
// up to twice the budget before it surfaces; that is a maintainer call and is
// called out in the PR rather than buried here.
// The caller supplies the dial itself, because WHICH backend to open is a
// routing decision (a workspace switch moves chrome; a plain bot navigation
// only opens the gateway) while the deadline enforced here is the same either
// way.
async function awaitProfileActivation(
  dial: () => Promise<void>,
  targetProfile: string,
  timeoutMs: number
): Promise<void> {
  const activation = dial()
  let timer: number | undefined

  try {
    await Promise.race([
      activation,
      new Promise<never>((_resolve, reject) => {
        // Same message shape as the hydration timeout on purpose: openSession's
        // catch keys the core stranded-session surface off this prefix, and a
        // wedged dial wants exactly that surface. The phase is distinguished in
        // the [bot-wake] support log, not in the user-facing string.
        timer = window.setTimeout(
          () => reject(new Error(`Timed out loading ${targetProfile}'s session history.`)),
          timeoutMs
        )
      })
    ])
  } finally {
    if (timer !== undefined) {
      window.clearTimeout(timer)
    }
  }

  // No extra catch on the abandoned dial: an in-flight activation has no
  // cancellation handle and keeps running after the budget expires, but
  // Promise.race subscribes to every input, so a rejection that lands after the
  // race has settled is already handled and cannot escape as an unhandled
  // rejection. An explicit `activation.catch()` here was dead code - verified
  // by mutation: removing it changed no test outcome.
}

export const host = {
  state: {
    /** Runtime id of the active chat session (null on a fresh draft). */
    activeSessionId: readonlyAtom<null | string>($activeSessionId),
    /** True from send until the first assistant payload on the focused chat. */
    awaitingResponse: readonlyAtom<boolean>($focusedAwaitingResponse),
    /**
     * True while the focused chat is working after a send. Covers the wait
     * for the first token and the stream that follows. Follows tile focus —
     * same signal the statusbar's busy pulse reads. A draft with no runtime
     * id uses the global flag.
     */
    busy: readonlyAtom<boolean>($focusedBusy),
    /** Runtime session id → mid-turn. Not socket state; see `gateway`. */
    busyBySession: readonlyAtom<Record<string, boolean>>($busyBySession),
    /** Registry source that owns the active gateway, when source-scoped. */
    connectionId: readonlyAtom<null | string>($activeConnectionId),
    /** Active workspace cwd ('' when detached). */
    cwd: readonlyAtom<string>($currentCwd),
    /** Runtime id of the FOCUSED chat session — the interacted tile, else the
     *  primary. Prefer this over `activeSessionId` for any readout that
     *  should follow the user between tiles (context, tokens, cost). */
    focusedSessionId: readonlyAtom<null | string>($focusedRuntimeId),
    /** Connection-qualified owner of the focused chat. Prefer this for any
     *  readout or mutation where separate sources can share a profile name. */
    focusedSessionOwner: readonlyAtom<PluginFocusedSessionOwner | null>($focusedSessionOwner),
    /** Owner profile of the focused chat (session-row stamp, falling back to
     *  the gateway profile for drafts/uncached ids). Compatibility projection
     *  of `focusedSessionOwner`; use the complete owner for source routing. */
    focusedSessionProfile: readonlyAtom<string>($focusedSessionProfile),
    /** Stored (durable) id of the focused session — for navigation and
     *  session-list matching, where runtime ids don't survive reloads. */
    focusedStoredSessionId: readonlyAtom<null | string>($focusedStoredSessionId),
    /** Live usage snapshot of the focused session (`context_used` /
     *  `context_max` / `context_percent`, token counts, `cost_usd`) —
     *  streamed by the backend, no RPC needed. Null while unresolved.
     *  The UsageStats-optional fields (context_*, cost_usd) arrive as the
     *  backend reports them, so read them with a fallback. */
    focusedUsage: readonlyAtom<null | UsageStats>($focusedUsage),
    /** Gateway socket state: 'idle' | 'connecting' | 'open' | …. Not turn-busy. */
    gateway: readonlyAtom<string>($gatewayState),
    /** Current main model slug. */
    model: readonlyAtom<string>($currentModel),
    /** Profile the live gateway is routed to. */
    profile: readonlyAtom<string>($activeGatewayProfile),
    /** Window geometry ({ width, height, narrow }). */
    viewport: readonlyAtom<ViewportRect>($viewport)
  },

  /** Toast into the app's notification stack. */
  notify,
  notifyError,

  // NOTE: every host door is async-safe — wrapped so a sync throw from an
  // internal helper (e.g. no desktop bridge in a plain browser) becomes a
  // rejection a plugin's .catch() sees, never an error-boundary crash.

  /** Tail an app log file (`agent` / `errors` / `gateway` / `gui` / …). */
  logs: async (...args: Parameters<typeof getLogs>) => getLogs(...args),

  /** Navigate the app router (hash routes, e.g. '/command-center?section=system'). */
  navigate: (path: string) => {
    window.location.hash = path.startsWith('#') ? path : `#${path}`
  },

  /** Pre-dial a profile's gateway socket in the background — pool-only, no
   *  activation, no navigation, no scope change (openGatewayForProfile; it
   *  already no-ops for shared-remote routes and the primary). Roster UIs
   *  call this after mount so the FIRST click on an agent doesn't pay the
   *  whole backend spawn + socket dial latency. Fire-and-forget: failures
   *  are swallowed — the click path re-runs its own ensure and surfaces
   *  errors properly. */
  warmProfile: (profile: string): void => {
    const name = (profile ?? '').trim()

    if (!name || name === $activeGatewayProfile.get()) {
      return
    }

    void openGatewayForProfile(name).catch(() => undefined)
  },

  /** Delete a profile THROUGH the desktop's teardown-routed REST path — the
   *  same door core surfaces use (DeleteProfileDialog). Electron intercepts
   *  the DELETE, tears down that profile's pool/primary backend first, and
   *  routes the follow-up request away from it, so a live (or hover-warmed)
   *  backend can't hold the profile dir open or respawn mid-delete and
   *  resurrect the directory (issue #52279). Plugins must prefer this over
   *  `cli.exec ['profile','delete',…]`, which bypasses that interception
   *  entirely. When the deleted profile was the live gateway's, the app is
   *  re-homed to the default profile — same semantics as the core dialog.
   *  Rejects with the backend's error when the delete fails. */
  deleteProfile: async (profile: string | PluginProfileRoute): Promise<void> => {
    const route =
      typeof profile === 'string'
        ? null
        : {
            ...profile,
            connectionId: String(profile.connectionId || '').trim(),
            profile: String(profile.profile || '').trim(),
            targetProfile: String(profile.targetProfile || '').trim()
          }

    const name = typeof profile === 'string' ? profile.trim() : route?.profile || ''

    if (route && (!route.connectionId || !route.profile || !route.targetProfile)) {
      throw new Error('deleteProfile: route requires connectionId, profile, and targetProfile')
    }

    const targetProfile = route?.targetProfile || name

    if (!name) {
      throw new Error('deleteProfile: profile name required')
    }

    if (normalizeProfileKey(targetProfile) === 'default') {
      throw new Error('The default profile cannot be deleted.')
    }

    // Capture before the delete; re-home after so our write is the last one
    // (mirrors DeleteProfileDialog — a refreshActiveProfile racing the dying
    // backend can't clobber the pill back to the deleted profile).
    const wasActive = route
      ? route.connectionId === ($activeConnectionId.get() || '') &&
        normalizeProfileKey(route.profile) === normalizeProfileKey($activeGatewayProfile.get())
      : normalizeProfileKey(name) === normalizeProfileKey($activeGatewayProfile.get())

    // A hover-warmed Bot Mode row owns a retained renderer socket. Retire it
    // before Electron stops the profile backend so the socket closure cannot
    // schedule a reconnect that resurrects the deleted profile.
    if (!route || route.mode === 'local') {
      retireLocalProfileGateways(targetProfile)
    }

    await deleteProfile(targetProfile, route ? { connectionId: route.connectionId, profile: route.profile } : undefined)

    // The profile rail paints from the shared $profiles cache; without a
    // refresh the deleted profile's badge survives and clicking it starts a
    // doomed spawn-retry loop against Electron's deletion guard (#88769).
    // Best-effort: the delete itself already succeeded.
    await refreshProfiles().catch(() => undefined)

    if (wasActive) {
      selectProfile('default')
      setActiveProfile('default')
    }
  },

  // ── Multi-source agents (the Bot Mode door) ───────────────────────────────

  /** Registry connection id serving the gateway `host.request` currently hits
   *  — null for the local/legacy primary path. Roster UIs need this to tell
   *  "a row from the backend I'm already showing" apart from "a row from
   *  another source": two connections can both expose a 'default' profile,
   *  and matching by profile name alone duplicates every agent when the
   *  active gateway is a registered remote. Re-read per use — it changes on
   *  profile/agent swaps. */
  activeConnectionId: (): null | string => activeGatewayConnectionId(),

  /** The registered connection list (labels, kinds, primary) — token bytes
   *  never included. Rejects on Desktop builds without the registry. */
  connections: async () => {
    const bridge = window.hermesDesktop?.connections

    if (!bridge) {
      throw new Error('This Desktop build has no connection registry. Update Hermes Desktop.')
    }

    const registryPayload = await bridge.list()
    const rows = Array.isArray(registryPayload?.connections) ? registryPayload.connections : []

    return rows.map(connection => ({ ...connection, primary: connection.id === registryPayload.primary }))
  },

  /** The union agent roster across every registered connection: one row per
   *  (source, profile) with the pre-computed @name-device handle for
   *  duplicates. Sources that are unreachable (or ssh connect-on-demand)
   *  appear in `sources` with an error instead of failing the call. */
  agents: async () => {
    const roster = window.hermesDesktop?.getAgentRoster

    if (!roster) {
      throw new Error('This Desktop build cannot enumerate multi-source agents. Update Hermes Desktop.')
    }

    return roster()
  },

  /** Pre-dial an agent's socket on ITS source — the (connection, profile)
   *  analogue of warmProfile. Fire-and-forget, same semantics. */
  warmAgent: (connectionId: null | string, profile: string): void => {
    void openGatewayForAgent(connectionId, (profile ?? '').trim() || 'default').catch(() => undefined)
  },

  /** Activate an agent's gateway (dialing it if needed) so subsequent
   *  host.request calls hit that agent's backend. Goes through the store's
   *  serialized activation path so $connection / $activeGatewayProfile follow
   *  and rapid switches can't land out of order. The local source falls
   *  through to the profile path — single-source plugins keep working
   *  against older behavior unchanged. */
  ensureAgent: async (connectionId: null | string, profile: string): Promise<void> =>
    ensureGatewayAgent(connectionId, (profile ?? '').trim() || 'default'),

  /** Open a stored session the way core surfaces do. A plugin/Bot Mode open
   *  is navigation, not a workspace or chrome API-home switch —
   *  keepAllProfilesScope defaults true so `$activeGatewayProfile` /
   *  Sessions REST stay on the previous (usually launch) backend while the
   *  bot backend is dialed in the background. The bot forever-chat is hidden
   *  and would otherwise look like every session disappeared. Pass false to
   *  also scope chrome onto that profile and collapse the sidebar. */
  openSession: async (storedSessionId: string, options: PluginOpenSessionOptions = {}): Promise<void> => {
    const generation = ++openSessionGeneration
    const explicitRoute = options.route ? { ...options.route } : null
    const profile = (explicitRoute?.profile ?? options.profile ?? '').trim()
    const targetProfile = normalizeProfileKey(profile || $activeGatewayProfile.get())

    // A local bot open passes only `profile` (no cross-connection route), but
    // its RPCs STILL have to reach that profile's own local gateway while chrome
    // stays on the launch profile. Synthesize a local owner route from the
    // profile so the persisted tile carries it — the session-request router
    // reads the tile route to dispatch on the owning backend, and the canonical
    // Bot Chat is hidden (never in $sessions), so this is the only owner record
    // it can consult. Without it, submit falls back to the active profile and
    // 4001s / hangs against a backend that never owned the session.
    //
    // This is ROUTING metadata only (tile ownerRoute + owner hint); the dial
    // path below still keys off the explicit cross-connection route, so a plain
    // local open dials exactly as before (openGatewayForProfile), never the
    // registry-secondary path.
    const localConnectionId = activeGatewayConnectionId()

    const ownerRoute =
      explicitRoute ??
      (options.workspaceMode === 'bots' && profile && localConnectionId
        ? { connectionId: localConnectionId, mode: 'local' as const, profile: targetProfile }
        : null)

    const expectHistory = options.expectHistory ?? false

    if (options.workspaceMode === 'bots') {
      publishWorkspaceScope(
        'bots',
        options.workspaceOwnerKey ?? null,
        ownerRoute ? { kind: 'route', route: ownerRoute } : null
      )
    }

    const openingStillCurrent = () =>
      generation === openSessionGeneration &&
      (options.workspaceMode !== 'bots' ||
        ($workspaceMode.get() === 'bots' && $workspaceOwnerKey.get() === (options.workspaceOwnerKey ?? null)))

    const plan = planPluginOpenSession({
      activeProfile: $activeGatewayProfile.get(),
      keepAllProfilesScope: options.keepAllProfilesScope,
      profile
    })

    // Wake-path phase timings. Logged ONLY on a hydration timeout (bridged
    // into desktop.log via the renderer-console tap), so a support bundle
    // pinpoints WHERE the budget went — profile activation vs hydration —
    // instead of leaving us to infer it from process spawn timestamps.
    const wakeStartedAt = Date.now()
    let profileActiveAt = wakeStartedAt
    const hydrationTimeoutMs = Math.max(1, options.hydrationTimeoutMs ?? DEFAULT_SESSION_HYDRATION_TIMEOUT_MS)
    // Which half of the wake a timeout landed in. Only meaningful on the
    // failure path, where the two phases have different remedies: a stuck dial
    // is a gateway problem, a slow transcript is a backend-warmup one.
    let wakePhase: 'activation' | 'hydration' = 'activation'

    if (ownerRoute) {
      setSessionOwnerHint(storedSessionId, ownerRoute)
    } else if (profile) {
      // Local plugin-owned opens (Bot Mode without a cross-connection route)
      // still carry an explicit owning profile. Record it: hidden sessions
      // (canonical Bot Chats) have no sidebar row, so this hint is the only
      // durable owner record the session-RPC router can consult — without it
      // a later prompt.submit resolves to the ACTIVE profile's backend and
      // 4001s while the bot's own backend is healthy.
      const connectionId = activeGatewayConnectionId()

      if (connectionId) {
        setSessionOwnerHint(storedSessionId, { connectionId, mode: 'local', profile: targetProfile })
      }
    }

    // Bounded to 2 attempts (never more): a cold profile backend can lose the
    // hydration-timeout race once and still be fine moments later, but this is
    // a caller-opt-in retry of the SAME wait, not a backoff loop.
    const maxAttempts = options.awaitHydration && options.retryHydrationTimeoutOnce ? 2 : 1

    try {
      // WHICH backend to dial is the plan's call; HOW LONG to wait is the wake
      // budget's. A workspace switch moves $activeGatewayProfile / chrome REST;
      // a plain navigation only opens the bot's gateway so session.resume can
      // hydrate, leaving chrome on the launch backend.
      // Dial keys off the EXPLICIT cross-connection route only: a synthesized
      // local ownerRoute is routing metadata for the tile/hint, and a local
      // profile must dial through openGatewayForProfile (its established path),
      // not the registry-secondary path openGatewayForAgent takes for a 'local'
      // connection id. Behavior for a plain local open is unchanged.
      const dial = explicitRoute
        ? () => openGatewayForAgent(explicitRoute.connectionId, explicitRoute.profile)
        : plan.switchWorkspace
          ? () => ensureGatewayProfile(plan.switchWorkspace as string)
          : plan.dialWithoutSwitching
            ? () => openGatewayForProfile(plan.dialWithoutSwitching as string)
            : null

      if (dial) {
        // Bounded only on the hydration contract, which is where a budget and a
        // Retry surface both already exist. A plain open never asked for a
        // deadline and has nowhere to render one, so it keeps today's
        // behaviour rather than gaining a rejection its callers cannot handle.
        await (options.awaitHydration ? awaitProfileActivation(dial, targetProfile, hydrationTimeoutMs) : dial())
        profileActiveAt = Date.now()
      }

      if (!openingStillCurrent()) {
        throw new Error('Session open was superseded by a newer selection.')
      }

      // Only a cross-connection (explicit route) open forces the all-profiles
      // view; a local bot open keeps the planner's decision, unchanged from
      // before the synthesized-route addition.
      if (explicitRoute) {
        setShowAllProfiles(true)
      } else if (plan.showAllProfiles !== null) {
        setShowAllProfiles(plan.showAllProfiles)
      }

      wakePhase = 'hydration'

      if (!openingStillCurrent()) {
        throw new Error('Session open was superseded by a newer selection.')
      }

      if (options.awaitHydration) {
        // Keep the target-specific overlay visible through transcript hydration,
        // not merely through the gateway/profile activation that precedes it.
        $gatewaySwapTarget.set(targetProfile)
      }

      // Only the HYDRATION half retries. Activation already failed its own
      // bounded wait above, and a wedged dial does not get better by dialling
      // again inside the same wake — that is the Retry surface's job.
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          const navigate = (to: string, opts?: { replace?: boolean }) => {
            const target = to.startsWith('#') ? to : `#${to}`

            if (opts?.replace) {
              window.location.replace(target)
            } else {
              window.location.hash = target
            }
          }

          const intent = options.intent ?? 'in-place'

          if (options.workspaceMode === 'bots') {
            openSession(storedSessionId, navigate, intent, {
              ownerRoute: ownerRoute ?? undefined,
              workspaceMode: 'bots',
              workspaceOwnerKey: options.workspaceOwnerKey,
              ...(options.tabTitle ? { workspaceTabTitle: options.tabTitle } : {})
            })
          } else {
            openSession(storedSessionId, navigate, intent)
          }

          // Judge the main surface AFTER the open: on a cold start the persisted
          // route can already point at this session while selection has not
          // settled, so a pre-open "already selected" precondition skips the
          // resume exactly when it is needed (#89206 — blank Bot Chat with the
          // roster preview intact). The surface is healthy only when this stored
          // session is selected, a runtime is bound, and the expected transcript
          // is present; anything less gets an explicit sequenced resume request.
          // The route-resume effect only honors the request while the route
          // points at this session, and consumes it alongside any resume the
          // navigation itself triggers, so a redundant request is a no-op.
          const surfaceHealthy =
            $selectedStoredSessionId.get() === storedSessionId &&
            Boolean($activeSessionId.get()) &&
            (!expectHistory || $messages.get().length > 0)

          if (options.awaitHydration && !surfaceHealthy) {
            requestSessionResume(storedSessionId, ownerRoute || undefined)
          }

          if (options.awaitHydration) {
            await waitForFocusedSessionHydration({
              expectHistory,
              generation,
              isCurrent: openingStillCurrent,
              profile: targetProfile,
              // A background dial never moves $activeGatewayProfile, so gating
              // hydration on it would wait for something that is not coming.
              requireActiveProfile: ownerRoute ? false : plan.requireActiveProfileForHydration,
              storedSessionId,
              timeoutMs: hydrationTimeoutMs
            })
          }

          break
        } catch (error) {
          const retryable =
            options.awaitHydration &&
            generation === openSessionGeneration &&
            attempt < maxAttempts &&
            error instanceof Error &&
            error.message.startsWith('Timed out loading ')

          if (!retryable) {
            throw error
          }

          // The registry check applies only to a real cross-connection route
          // (explicit): a synthesized local route is never in getProfileRoutes,
          // so checking it would spuriously abort a local bot's hydration retry.
          if (explicitRoute && !(await pluginRouteStillRegistered(explicitRoute))) {
            throw new Error(`The ${targetProfile} gateway is no longer available.`)
          }

          // Logged per attempt so a support bundle shows the retry happened at
          // all; the terminal failure is reported once by the catch below.
          console.warn('[bot-wake] hydration timed out, retrying', {
            attempt,
            hydrationWaitMs: Date.now() - profileActiveAt,
            profile: targetProfile,
            storedSessionId
          })
        }
      }
    } catch (error) {
      if (
        options.awaitHydration &&
        openingStillCurrent() &&
        error instanceof Error &&
        error.message.startsWith('Timed out loading ')
      ) {
        const timedOutAt = Date.now()

        console.warn('[bot-wake] hydration timed out', {
          attempts: wakePhase === 'hydration' ? maxAttempts : 1,
          hydrationWaitMs: wakePhase === 'hydration' ? timedOutAt - profileActiveAt : 0,
          phase: wakePhase,
          profile: targetProfile,
          profileActivationMs: (wakePhase === 'activation' ? timedOutAt : profileActiveAt) - wakeStartedAt,
          runtimeBound: Boolean($activeSessionId.get()),
          selectionSettled: $selectedStoredSessionId.get() === storedSessionId,
          storedSessionId,
          transcriptPainted: $messages.get().length > 0
        })
        // Reuse the core stranded-session surface: it renders the explicit
        // error and Retry button, and the normal resume path clears the latch.
        setResumeExhaustedSessionId(storedSessionId)
      }

      throw error
    } finally {
      if (options.awaitHydration && generation === openSessionGeneration) {
        $gatewaySwapTarget.set(null)
      }
    }
  },

  /** Open (or re-front) a plugin-rendered MAIN-AREA workspace tile — the same
   *  surface a session tile or a preview occupies: a closeable tab docked
   *  beside the main workspace, taking over the chat area when active. This is
   *  the generic main-view door for plugins whose surface is not a stored
   *  session (`openSession` stays the door for those). Re-opening the same
   *  `id` refreshes `render`/`title` in place and fronts the existing tab
   *  instead of stacking a duplicate. Returns a disposer that closes the tab;
   *  the tab's own Close (⌘W / strip ✕) routes through the same teardown and
   *  fires `onClose`. Feature-detect on older desktops
   *  (`typeof host.openWorkspace === 'function'`) and keep an in-panel
   *  fallback. */
  openWorkspace: (
    id: string,
    options: {
      dock?: { before?: null | string; pane: string; pos: 'bottom' | 'center' | 'left' | 'right' | 'top' }
      headerVeto?: boolean
      minWidth?: string
      onClose?: () => void
      render: () => ReactNode
      title?: string
      uncloseable?: boolean
      workspaceMode?: WorkspaceMode
      workspaceOwnerKey?: string
    }
  ): (() => void) => {
    const key = (id ?? '').trim()

    if (!key || typeof options?.render !== 'function') {
      throw new Error('openWorkspace: an id and a render function are required')
    }

    const paneId = `plugin-workspace:${key}`

    const dispose = registry.register({
      area: 'panes',
      data: {
        // The session-tile shape: a full workspace surface docked beside main,
        // closeable so it keeps its tab when it lands in a zone of its own.
        dock: options.dock ?? { pane: 'workspace', pos: 'center' },
        headerVeto: options.headerVeto,
        minWidth: options.minWidth ?? '22rem',
        placement: 'main',
        uncloseable: options.uncloseable
      },
      id: paneId,
      render: options.render,
      title: options.title ?? key,
      workspaceMode: options.workspaceMode,
      workspaceOwnerKey: options.workspaceOwnerKey
    })

    const close = () => {
      registerPaneCloser(paneId)
      dispose()
      removeTreePane(paneId)
      options.onClose?.()
    }

    // Route the tab's Close through OUR teardown: without a closer, closing a
    // core-sourced contributed pane only dismisses it and the registration
    // would leak past the plugin surface that owns it.
    registerPaneCloser(paneId, close)
    revealTreePane(paneId)

    return close
  },

  /** Switch the visible main-pane workspace without unregistering retained panes. */
  setWorkspaceScope: (
    mode: WorkspaceMode,
    ownerKey: null | string = null,
    newSessionTarget: WorkspaceNewSessionTarget | null = null
  ): boolean => publishWorkspaceScope(mode, ownerKey, newSessionTarget),

  /** Start a fresh chat draft, optionally pointed at another profile (its
   *  backend spins up in the background — same door the sidebar's per-profile
   *  "+" uses). */
  newChat: (profile?: null | string | PluginProfileRoute, options: PluginNewChatOptions = {}): void => {
    if (options.workspaceMode === 'bots') {
      if (!profile || typeof profile === 'string' || !options.workspaceOwnerKey) {
        notify({ kind: 'error', message: 'Select a Bot before starting another chat.' })

        return
      }

      publishWorkspaceScope('bots', options.workspaceOwnerKey, { kind: 'route', route: { ...profile } })

      const openTab = $newSessionTabAction.get()

      if (!openTab) {
        notify({ kind: 'error', message: 'Update Hermes Desktop to open another Bot chat.' })

        return
      }

      openTab()

      return
    }

    if (profile && typeof profile !== 'string') {
      newSessionInAgent({ ...profile })
    } else {
      newSessionInProfile((profile ?? '').trim() || $activeGatewayProfile.get())
    }

    window.location.hash = '#/'
  },

  /** Reactive on-screen visibility of a contributed pane: true while it is in
   *  the layout tree, not dismissed/hidden, its zone un-minimized, AND holding
   *  its zone's active tab slot (a lone pane in its own zone counts). The
   *  contribution-scoped pane id is `<pluginId>:<paneId>`. Memoized per id —
   *  safe to call in render. Feature-detect on older desktops
   *  (`typeof host.paneVisibility === 'function'`). */
  paneVisibility: (paneId: string): ReadableAtom<boolean> => $paneVisible(paneId),

  /** HEAR the gateway stream (message deltas, session lifecycle, tool
   *  activity, …) by event type — `'*'` for everything. Returns a disposer.
   *  Listeners are isolated; a throw can't affect app dispatch. */
  onEvent: onGatewayEvent,

  /** Restart the backend gateway (progress surfaces in the core statusbar). */
  restartGateway: async () => runGatewayRestart(),

  /** One-shot system status snapshot (platforms, versions, …). */
  status: async () => getStatus(),

  /** Credential-free routes across every current registry source. Identity is
   *  the (connectionId, profile) pair; endpoint/auth details stay in Electron. */
  profileRoutes: async () => {
    const desktop = window.hermesDesktop
    const getProfileRoutes = desktop?.getProfileRoutes

    if (!getProfileRoutes) {
      throw new Error('Hermes Desktop connection routing unavailable')
    }

    let profiles = $profiles.get()

    try {
      profiles = await refreshProfiles()
    } catch {
      // Route inventory is a read: a transient backend failure falls back to
      // the last cache. Electron always adds the primary Desktop profile.
    }

    return getProfileRoutes(profiles.map(profile => profile.name))
  },

  /** Gateway JSON-RPC through a credential-free route descriptor without
   *  foregrounding it. Passing a bare profile is the v1/local compatibility
   *  overload; registry callers must pass the descriptor so duplicate names
   *  remain unambiguous. */
  requestProfile: async <T>(
    route: PluginProfileRoute | string,
    method: string,
    params: Record<string, unknown> = {}
  ): Promise<T> => requestPluginProfile<T>(route, method, params),

  /** Pin a route's pooled gateway socket open across repeated `requestProfile`
   *  calls (#93594: the bot-relay drain loop was dialing and tearing down a
   *  fresh WebSocket per registered connection per tick). Returns a once-only
   *  release. Local routes are exempt (no-op release) so the idle reaper can
   *  still reclaim spawned local backends. Feature-detect on older desktops
   *  (`typeof host.retainProfileSocket === 'function'`). */
  retainProfileSocket: (route: PluginProfileRoute | string): (() => void) => {
    if (typeof route === 'string' || !route) {
      // Bare-profile compatibility overload: local/legacy routing — exempt.
      return () => undefined
    }

    return retainGatewayForRelay(route.connectionId, route.profile)
  },

  /** Hold a route's pooled socket open across a multi-RPC, session-scoped
   *  sequence (#93602). Each requestProfile call is its own request lease, so
   *  a non-retained secondary socket closes at refcount 0 between calls — and
   *  the gateway reaps any runtime session that socket minted, failing the
   *  next RPC with 4001. Acquire before the first session-scoped RPC, release
   *  (idempotent) in a `finally`. Feature-detect: older hosts lack this. */
  retainProfile: async (route: PluginProfileRoute | string): Promise<() => void> => {
    if (typeof route !== 'string') {
      if (!route.connectionId.trim() || !route.profile.trim()) {
        throw new Error('Profile route must include connectionId and profile')
      }

      return retainGatewayForAgent(route.connectionId, route.profile)
    }

    return retainGatewayForAgent(null, route.trim() || 'default')
  },

  /** Gateway JSON-RPC — sessions, config, skills, cron, kanban, everything
   *  the app itself uses. Lazy: resolves the LIVE socket per call. */
  request: async <T>(method: string, params: Record<string, unknown> = {}): Promise<T> => {
    const gateway = $gateway.get()

    if (!gateway) {
      throw new Error('Hermes gateway unavailable')
    }

    return gateway.request<T>(method, params)
  },

  /** The LIVE gateway instance for the active profile (null before the first
   *  socket opens). Most plugins want `host.request`; this exists for SDK
   *  components that take a `HermesGateway` prop directly (e.g. `McpTab`),
   *  which need the instance, not just a JSON-RPC door. Re-read per use — the
   *  active instance changes on a profile swap. */
  getGateway: (): HermesGateway | null => $gateway.get()
}

// -- react bridge -------------------------------------------------------------

// Every contribution surface, plugin-reachable: register keybinds, palette
// commands, routes, themes, panes, composer extensions, and bar items with
// the same area ids + payload types core uses.
export {
  COMPOSER_AREAS,
  type ComposerAtCompletionItem,
  type ComposerAtCompletionSource,
  type ComposerAttachmentProvider,
  type ComposerMiddleware
} from '@/app/chat/composer/contrib'

// -- ui: the design language --------------------------------------------------

export { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
export { type RouteContribution, ROUTES_AREA, SIDEBAR_NAV_AREA, type SidebarNavContribution } from '@/app/routes'
/** THE full per-toolset config panel core Settings renders — provider picker,
 *  env vars / API keys, model catalog picker, and post-setup runners. Route-
 *  decoupled (the "manage keys" deep link is a no-op outside the router); pass
 *  `toolset`, optional `onConfiguredChange`, and an optional `profile`. */
export { ToolsetConfigPanel } from '@/app/settings/toolset-config-panel'
/** THE model catalog menu — the same searchable, provider-grouped, family-
 *  collapsing picker the chat composer uses, including the per-row
 *  thinking/effort/fast submenu. Drive it with a `ModelMenuController`: the
 *  menu renders and navigates, your controller decides what a selection MEANS
 *  (write to a session, hold a per-task override, …). Never fork it — a copy
 *  drifts from the composer the first time either side changes. */
export {
  ModelCatalogMenu,
  type ModelChoice,
  ModelMenuCloseContext,
  type ModelMenuController
} from '@/app/shell/model-catalog-menu'
export type { StatusbarItem } from '@/app/shell/statusbar-controls'
export type { TitlebarTool } from '@/app/shell/titlebar-controls'

/** THE whole Capabilities surface (Skills / Tools / MCP tabs, installed
 *  lists, full-skill detail pane, embedded hub picker with one-click
 *  installs). For plugin dialogs pass `embedded` (tab state stays local —
 *  never touches the page router) and `fixedProfile` to pin every tab to one
 *  bot's backend; the internal profile selector hides itself. Add
 *  `fixedConnection` (registry connection id) to pin a bot living on another
 *  registered gateway — probe `SkillsView.supportsFixedConnection` first;
 *  builds without it would route the pin to the ACTIVE gateway. Bot Mode's
 *  Advanced section is the reference consumer. */
export { SkillsView } from '@/app/skills'
/** THE full MCP tab core Settings renders — per-server enable + OAuth sign-in
 *  + API-key setup + live probes, not a checkbox list. Route-decoupled so it
 *  renders anywhere (a plugin dialog); pass a live `gateway` (see
 *  `host.getGateway()`) and an optional `profile` to scope it to one bot. */
export { McpTab } from '@/app/skills/mcp-tab'
/** Pane placement roles. `'floating'` is the one NON-tiling value: the pane is
 *  excluded from the layout tree and rendered as a fixed, draggable card above
 *  it — it takes no width from any zone, has no tab, and can't be docked.
 *  Pair it with `anchor` (spawn corner, default `'top-right'`) plus
 *  `width`/`height`. */
export type { FloatingAnchor } from '@/components/pane-shell/tree/renderer/floating-rect'
export { StatusDot, type StatusTone } from '@/components/status-dot'
export { Badge } from '@/components/ui/badge'
export { Button } from '@/components/ui/button'
export { Checkbox } from '@/components/ui/checkbox'
export { Codicon } from '@/components/ui/codicon'
export { ConfirmDialog } from '@/components/ui/confirm-dialog'
export {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
export { CopyButton } from '@/components/ui/copy-button'
export { DecodeText } from '@/components/ui/decode-text'
export {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
export {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
export { EmptyState } from '@/components/ui/empty-state'
export { ErrorState } from '@/components/ui/error-state'
export { FadeScroll } from '@/components/ui/fade-scroll'
export { GlyphSpinner } from '@/components/ui/glyph-spinner'
export { Input } from '@/components/ui/input'
export { Kbd, KbdGroup } from '@/components/ui/kbd'
/** The app's canonical loader (animated curves; `lemniscate-bloom` for long
 *  page loads) — the same one every core page uses. */
export { Loader, type LoaderType } from '@/components/ui/loader'
export { LogView } from '@/components/ui/log-view'
export { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
export { ScrollArea } from '@/components/ui/scroll-area'
export { SearchField } from '@/components/ui/search-field'
export { SegmentedControl } from '@/components/ui/segmented-control'
export { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
export { Separator } from '@/components/ui/separator'
export { Skeleton } from '@/components/ui/skeleton'
export { Switch } from '@/components/ui/switch'
export { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
export { Textarea } from '@/components/ui/textarea'
export { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
export type { GatewayEventListener } from '@/contrib/events'
export type {
  HermesPlugin,
  PluginContext,
  PluginContribution,
  PluginNativeNotificationInput,
  PluginNotificationAction,
  PluginOs,
  PluginRestOptions,
  PluginStorage
} from '@/contrib/plugin'
/** Mount-scoped contribution: while the rendering component is mounted, its
 *  children render in the target area's slot; unmount disposes it. Use for
 *  page-owned chrome (a page's titlebar control leaves with the page) —
 *  `ctx.register` stays the door for permanent contributions. Namespace the
 *  id with your plugin slug (`kanban:board-switcher`). */
export { Contribute, type ContributeProps } from '@/contrib/react/contribute'

// -- contracts ----------------------------------------------------------------

export type { Contribution } from '@/contrib/types'
/** The live gateway instance type — for typing the `gateway` prop `McpTab`
 *  takes; obtain the instance from `host.getGateway()`. */
export type { HermesGateway } from '@/hermes'
/** Grab-to-pan for overflow containers (boards, timelines, wide tables) —
 *  the shared scrub primitive; don't hand-roll drag-to-scroll. */
export { type GrabScroll, useGrabScroll } from '@/hooks/use-grab-scroll'
/** Localized copy. `useI18n` reuses the app's strings; `usePluginI18n(id)` +
 *  `ctx.i18n.register` let a plugin ship its OWN locale bundles, scoped like
 *  `ctx.storage` and resolved against the app's active locale — no core edit. */
export {
  type Locale,
  type PluginI18n,
  type PluginLocaleBundles,
  type PluginMessages,
  type PluginMessageValue,
  type PluginTranslate,
  useI18n,
  usePluginI18n
} from '@/i18n'
/** THE way to run a decorative rAF animation (avatars, shimmer, sprites):
 *  fps budget + hidden/minimized/unfocused pause + idle dormancy + teardown.
 *  Plugins must route animation clocks through this instead of raw rAF loops
 *  so a disabled plugin or an empty roster costs zero frames. */
export { type BudgetedLoop, type BudgetedLoopOptions, createBudgetedLoop } from '@/lib/budgeted-loop'
/** THE compact-number formatter — every user-facing count/token figure goes
 *  through here (1230 → "1.2k", 1_500_000 → "1.5M"). Don't hand-roll `/1000`. */
export { compactNumber } from '@/lib/format'
export { triggerHaptic as haptic } from '@/lib/haptics'
export type { HermesOpenTarget } from '@/lib/hermes-open-target'
/** The app's lucide icon set (RefreshCw, LayoutDashboard, Activity, …). */
export * as icons from '@/lib/icons'
export { type KeybindContribution, KEYBINDS_AREA } from '@/lib/keybinds/actions'
export { formatModifierToken } from '@/lib/keybinds/combo'
/** The app's deterministic identity color for a name (profiles, assignees,
 *  authors) + its translucent tag fill — so plugin-rendered identities read
 *  the same hue as everywhere else. */
export { profileColor, profileColorSoft } from '@/lib/profile-color'
/** The shared client itself, for invalidation OUTSIDE React (e.g. a
 *  `ctx.socket` frame invalidating a query). Inside components keep using
 *  `useQueryClient`. */
export { queryClient } from '@/lib/query-client'
/** Hermes' reasoning levels + their compact labels, so a plugin surfacing a
 *  thinking depth uses the same scale and spelling as the rest of the app. */
export {
  DEFAULT_REASONING_EFFORT,
  REASONING_EFFORT_VALUES,
  REASONING_EFFORTS,
  type ReasoningEffort,
  reasoningEffortLabel
} from '@/lib/reasoning-effort'

export const PANES_AREA = 'panes'
/** The app's own gateway-readiness evaluation (setup.status +
 *  setup.runtime_check, reconciled) — pass `host.request`. Don't hand-roll
 *  readiness from raw RPC shapes. */
export { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
export const STATUSBAR_AREAS = { left: 'statusBar.left', right: 'statusBar.right' } as const
export const TITLEBAR_AREAS = { center: 'titleBar.center', left: 'titleBar.left', right: 'titleBar.right' } as const

export { coarseElapsed, fmtDateTime, fmtDayTime, relativeTime } from '@/lib/time'
/** The transcript as a contribution area: register a named `::directive{...}`
 *  and the model can render your component inline in assistant messages. */
export {
  TRANSCRIPT_DIRECTIVE_AREA,
  type TranscriptDirectiveContribution,
  type TranscriptDirectiveProps
} from '@/lib/transcript-directives'
export { cn } from '@/lib/utils'
/** Live accent override — set a hex and the ACTIVE theme repaints with its
 *  accent family re-seeded from it (see `retintTheme`); `null` restores the
 *  authored palette. Deliberately not persisted: it is an authoring knob, not
 *  a setting, so a plugin that sets it must clear it on dispose. */
export { $accentOverride, setAccentOverride } from '@/themes/accent-override'
/** OKLCH colour maths, for anything deriving a palette rather than hardcoding
 *  one: perceptual conversion, the sRGB gamut boundary, WCAG contrast, and
 *  hue-stable blending. */
export {
  contrastRatio,
  hexToOklch,
  hueDelta,
  maxChroma,
  mixOklab,
  normalizeHex,
  type Oklch,
  oklchToHex,
  oklchToSrgb255,
  readableOn
} from '@/themes/color'
/** The painted theme, its name, and the appearance it resolved to — plus
 *  `setTheme` / `setMode` to change it from a component. */
export { useTheme } from '@/themes/context'
/** Switch the theme from outside React (a gateway event, a connection coming
 *  up, any callback with no component around it). Returns false and leaves the
 *  appearance alone when the name doesn't resolve, so it doubles as the "is
 *  this theme installed?" check. */
export { requestTheme } from '@/themes/request'
export { retintTheme, themeHue } from '@/themes/retint'
export type { DesktopTheme, DesktopThemeColors } from '@/themes/types'
export { THEMES_AREA } from '@/themes/user-themes'
export type { RpcEvent, StatusResponse } from '@/types/hermes'
/** Subscribe a component to a `host.state` atom. */
export { useStore as useValue } from '@nanostores/react'
/** The app's data-fetching layer. Plugins share the ONE QueryClient mounted at
 *  the app root, so their queries cache, dedupe, poll (`refetchInterval`), and
 *  invalidate exactly like core screens — no hand-rolled atoms or polls. */
export { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
/** Deterministic soft-body avatars from any string (name → face). String
 *  renderer for rasterization; React component for live rendering. */
export { blobatar as blobatarSvg } from 'blobatar/blob'
export { Blobatar } from 'blobatar/react'
/** Plugin-local reactive state (share between a trigger and its panel, poll
 *  loops, cross-component signals) — the same primitive `host.state` uses. */
export { atom, computed } from 'nanostores'
/** Markdown renderer (same pipeline core chat surfaces use) so plugins render
 *  message text as a preview instead of raw Markdown source. */
export { Streamdown } from 'streamdown'
