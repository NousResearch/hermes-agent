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
  $paneVisible,
  registerPaneCloser,
  removeTreePane,
  revealTreePane
} from '@/components/pane-shell/tree/store'
import { onGatewayEvent } from '@/contrib/events'
import { registry } from '@/contrib/registry'
import { deleteProfile, getLogs, getStatus, type HermesGateway } from '@/hermes'
import {
  $gateway,
  activeGatewayConnectionId,
  openGatewayForAgent,
  openGatewayForProfile,
  requestGatewayForAgent,
  requestGatewayForProfile,
  retireLocalProfileGateways
} from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $profiles,
  ensureGatewayAgent,
  ensureGatewayProfile,
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
  $selectedStoredSessionId
} from '@/store/session'
import {
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $sessionStates
} from '@/store/session-states'
import { runGatewayRestart } from '@/store/system-actions'
import type { UsageStats } from '@/types/hermes'

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

  /** Open a stored session the way core surfaces do (focus an existing
   *  tile/main, else load into main). When `profile` names a non-active
   *  profile, its backend is activated first so the resume routes to the
   *  right state.db — the same soft profile swap the unified sidebar does.
   *  `keepAllProfilesScope` (default true) keeps the Sessions sidebar in the
   *  unified all-profiles view instead of narrowing it to the target
   *  profile's sessions — a cross-profile open from a plugin surface is a
   *  navigation, not a scope choice; pass false to also scope the sidebar. */
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
  deleteProfile: async (profile: string): Promise<void> => {
    const name = (profile ?? '').trim()

    if (!name) {
      throw new Error('deleteProfile: profile name required')
    }

    if (normalizeProfileKey(name) === 'default') {
      throw new Error('The default profile cannot be deleted.')
    }

    // Capture before the delete; re-home after so our write is the last one
    // (mirrors DeleteProfileDialog — a refreshActiveProfile racing the dying
    // backend can't clobber the pill back to the deleted profile).
    const wasActive = normalizeProfileKey(name) === normalizeProfileKey($activeGatewayProfile.get())

    // A hover-warmed Bot Mode row owns a retained renderer socket. Retire it
    // before Electron stops the profile backend so the socket closure cannot
    // schedule a reconnect that resurrects the deleted profile.
    retireLocalProfileGateways(name)
    await deleteProfile(name)

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

    return bridge.list()
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

  openSession: async (
    storedSessionId: string,
    options: { intent?: OpenSessionIntent; keepAllProfilesScope?: boolean; profile?: null | string } = {}
  ): Promise<void> => {
    const profile = (options.profile ?? '').trim()

    if (profile && profile !== $activeGatewayProfile.get()) {
      await ensureGatewayProfile(profile)

      if (options.keepAllProfilesScope !== false) {
        setShowAllProfiles(true)
      }
    }

    openSession(
      storedSessionId,
      (to: string, opts?: { replace?: boolean }) => {
        const target = to.startsWith('#') ? to : `#${to}`

        if (opts?.replace) {
          window.location.replace(target)
        } else {
          window.location.hash = target
        }
      },
      options.intent ?? 'in-place'
    )
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
    options: { minWidth?: string; onClose?: () => void; render: () => ReactNode; title?: string }
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
        dock: { pane: 'workspace', pos: 'center' },
        minWidth: options.minWidth ?? '22rem',
        placement: 'main'
      },
      id: paneId,
      render: options.render,
      title: options.title ?? key
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

  /** Start a fresh chat draft, optionally pointed at another profile (its
   *  backend spins up in the background — same door the sidebar's per-profile
   *  "+" uses). */
  newChat: (profile?: null | string): void => {
    newSessionInProfile((profile ?? '').trim() || $activeGatewayProfile.get())
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
export { THEMES_AREA } from '@/themes/user-themes'
export type { RpcEvent, StatusResponse } from '@/types/hermes'
/** Subscribe a component to a `host.state` atom. */
export { useStore as useValue } from '@nanostores/react'
/** The app's data-fetching layer. Plugins share the ONE QueryClient mounted at
 *  the app root, so their queries cache, dedupe, poll (`refetchInterval`), and
 *  invalidate exactly like core screens — no hand-rolled atoms or polls. */
export { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
/** Plugin-local reactive state (share between a trigger and its panel, poll
 *  loops, cross-component signals) — the same primitive `host.state` uses. */
export { atom, computed } from 'nanostores'
/** Markdown renderer (same pipeline core chat surfaces use) so plugins render
 *  message text as a preview instead of raw Markdown source. */
export { Streamdown } from 'streamdown'
