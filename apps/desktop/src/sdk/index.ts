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

import { atom, type ReadableAtom } from 'nanostores'

import { openSession, type OpenSessionIntent } from '@/app/open-session'
import { $narrowViewport } from '@/components/pane-shell/tree/store'
import { onGatewayEvent } from '@/contrib/events'
import { getLogs, getStatus, type HermesGateway } from '@/hermes'
import { $gateway, ensureGatewayForAgent, openGatewayForAgent, openGatewayForProfile } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, ensureGatewayProfile, newSessionInProfile, setShowAllProfiles } from '@/store/profile'
import { $activeSessionId, $currentCwd, $currentModel, $gatewayState } from '@/store/session'
import { runGatewayRestart } from '@/store/system-actions'

// -- state: readonly views over the app's live atoms -------------------------

const readonlyAtom = <T>(atomLike: ReadableAtom<T>): ReadableAtom<T> => atomLike

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

const $viewport = atom<ViewportRect>(readViewport())

if (typeof window !== 'undefined') {
  const refresh = () => $viewport.set(readViewport())
  window.addEventListener('resize', refresh)
  $narrowViewport.listen(refresh)
}

export const host = {
  state: {
    /** Runtime id of the active chat session (null on a fresh draft). */
    activeSessionId: readonlyAtom<null | string>($activeSessionId),
    /** Active workspace cwd ('' when detached). */
    cwd: readonlyAtom<string>($currentCwd),
    /** Gateway socket state: 'idle' | 'connecting' | 'open' | …. */
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

  // ── Multi-source agents (the Bot Mode door) ───────────────────────────────

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
   *  host.request calls hit that agent's backend. The local source falls
   *  through to the profile path — single-source plugins keep working
   *  against older behavior unchanged. */
  ensureAgent: async (connectionId: null | string, profile: string): Promise<void> =>
    ensureGatewayForAgent(connectionId, (profile ?? '').trim() || 'default'),

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

  /** Start a fresh chat draft, optionally pointed at another profile (its
   *  backend spins up in the background — same door the sidebar's per-profile
   *  "+" uses). */
  newChat: (profile?: null | string): void => {
    newSessionInProfile((profile ?? '').trim() || $activeGatewayProfile.get())
    window.location.hash = '#/'
  },

  /** HEAR the gateway stream (message deltas, session lifecycle, tool
   *  activity, …) by event type — `'*'` for everything. Returns a disposer.
   *  Listeners are isolated; a throw can't affect app dispatch. */
  onEvent: onGatewayEvent,

  /** Restart the backend gateway (progress surfaces in the core statusbar). */
  restartGateway: async () => runGatewayRestart(),

  /** One-shot system status snapshot (platforms, versions, …). */
  status: async () => getStatus(),

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
export { COMPOSER_AREAS, type ComposerAttachmentProvider, type ComposerMiddleware } from '@/app/chat/composer/contrib'

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

// -- contracts ----------------------------------------------------------------

/** Mount-scoped contribution: while the rendering component is mounted, its
 *  children render in the target area's slot; unmount disposes it. Use for
 *  page-owned chrome (a page's titlebar control leaves with the page) —
 *  `ctx.register` stays the door for permanent contributions. Namespace the
 *  id with your plugin slug (`kanban:board-switcher`). */
export { Contribute, type ContributeProps } from '@/contrib/react/contribute'
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

export const PANES_AREA = 'panes'
/** Hermes' reasoning levels + their compact labels, so a plugin surfacing a
 *  thinking depth uses the same scale and spelling as the rest of the app. */
export {
  DEFAULT_REASONING_EFFORT,
  REASONING_EFFORT_VALUES,
  REASONING_EFFORTS,
  type ReasoningEffort,
  reasoningEffortLabel
} from '@/lib/reasoning-effort'
export const STATUSBAR_AREAS = { left: 'statusBar.left', right: 'statusBar.right' } as const
export const TITLEBAR_AREAS = { center: 'titleBar.center', left: 'titleBar.left', right: 'titleBar.right' } as const

/** The app's own gateway-readiness evaluation (setup.status +
 *  setup.runtime_check, reconciled) — pass `host.request`. Don't hand-roll
 *  readiness from raw RPC shapes. */
export { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
export { coarseElapsed, fmtDateTime, fmtDayTime, relativeTime } from '@/lib/time'
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
