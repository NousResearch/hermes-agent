/**
 * Desktop love/hate telemetry: which areas get used, which buttons get
 * pressed, what gets in the way, what gets turned off, and where first run
 * stops. No UI of its own — every fact is a counter wired into an existing
 * interaction, reported over the shared-metrics RPCs (`shared_metrics.desktop_*`).
 *
 * Opt-in only. Everything here is gated on `$desktopMetricsGate`, the focused
 * profile's `telemetry.shared_metrics.enabled` as last read from the backend
 * (the same switch the consent strip and Settings › Privacy write). While it is
 * not `on` no measurements are persisted or sent; turning it off purges payloads,
 * revokes sleeping peer collectors, and drops main's pending renderer-crash record. Binding
 * another (connection, profile) forgets the gate until that profile's switch is
 * read, so nothing of one profile is ever kept or sent under another's.
 *
 * Before the first-run consent answer, onboarding transitions are held in memory
 * only (never persisted or sent) and replayed if the user opts in during that
 * app session; a known "no" drops them.
 *
 * Every value is a code-defined id from the closed sets below (mirrored by
 * hermes_cli/observability/shared_metrics_contract.py, which collapses any
 * stranger to `other`). Never a session id, path, message, bot name or setting
 * value.
 *
 * Local state (localStorage `hermes.desktop.metrics.v1:<hash of connection|profile>`,
 * written only while on): the UTC day, areas already reported today, per-day
 * caps, today's aggregated action counts and per-mode activity, finished days
 * awaiting the backend's ack, and the onboarding latch (steps sent / still open).
 * Every change re-reads the record, so peer windows of one profile add to it
 * instead of overwriting each other; the backend latches each day and area once.
 * Frame histograms and visible time use single-writer journals under the same scope's `:frames`
 * family. A finished day waits for peer visible tails to checkpoint or exceed the suspend limit.
 * Off retains only a payload-free revocation token; no local owner id crosses the RPC boundary.
 */

import { randomUuid } from '@hermes/shared'
import { atom } from 'nanostores'

import { $workspaceMode } from '@/components/pane-shell/workspace-scope'
import { KEYBIND_ACTION_IDS, KEYBIND_READONLY } from '@/lib/keybinds/actions'
import { readJson, writeJson } from '@/lib/storage'

import { type DesktopFrameConsent, DesktopFrameMetrics } from './desktop-metrics-frames'
import type { SharedMetricsRequester } from './shared-metrics'

// ── closed vocabularies ───────────────────────────────────────────────────────

/** Settings views (SETTINGS_VIEWS in app/settings/index.tsx) → area. */
const SETTINGS_AREAS: Record<string, DesktopFeatureArea> = {
  about: 'settings_about',
  billing: 'settings_billing',
  'config:advanced': 'settings_config_advanced',
  'config:appearance': 'settings_config_appearance',
  'config:browser': 'settings_config_browser',
  'config:chat': 'settings_config_chat',
  'config:memory': 'settings_config_memory',
  'config:model': 'settings_config_model',
  'config:safety': 'settings_config_safety',
  'config:voice': 'settings_config_voice',
  'config:workspace': 'settings_config_workspace',
  // Legacy alias: the Connections tab redirects to Gateways.
  connections: 'settings_gateway',
  gateway: 'settings_gateway',
  keybinds: 'settings_keybinds',
  keys: 'settings_keys',
  notifications: 'settings_notifications',
  providers: 'settings_providers',
  sessions: 'settings_sessions',
  vault: 'settings_vault'
}

export type DesktopFeatureArea =
  | 'agents'
  | 'artifacts'
  | 'bot_mode'
  | 'browser_pane'
  | 'capabilities'
  | 'command_center'
  | 'command_palette'
  | 'cron'
  | 'extension_page'
  | 'file_pane'
  | 'find_in_page'
  | 'kanban'
  | 'messaging'
  | 'model_picker'
  | 'other'
  | 'profiles'
  | 'projects'
  | 'review_pane'
  | 'session_import'
  | 'session_picker'
  | 'session_search'
  | 'session_switcher'
  | 'settings_about'
  | 'settings_billing'
  | 'settings_config_advanced'
  | 'settings_config_appearance'
  | 'settings_config_browser'
  | 'settings_config_chat'
  | 'settings_config_memory'
  | 'settings_config_model'
  | 'settings_config_safety'
  | 'settings_config_voice'
  | 'settings_config_workspace'
  | 'settings_gateway'
  | 'settings_keybinds'
  | 'settings_keys'
  | 'settings_notifications'
  | 'settings_other'
  | 'settings_providers'
  | 'settings_sessions'
  | 'settings_vault'
  | 'skins'
  | 'starmap'
  | 'terminal_pane'
  | 'voice_conversation'
  | 'voice_dictation'
  | 'webhooks'

/** First path segment of a full-page route (APP_ROUTES) → area. Chat paths carry session ids and map to nothing. */
const ROUTE_AREAS: Record<string, DesktopFeatureArea> = {
  agents: 'agents',
  artifacts: 'artifacts',
  capabilities: 'capabilities',
  'command-center': 'command_center',
  cron: 'cron',
  kanban: 'kanban',
  messaging: 'messaging',
  profiles: 'profiles',
  'session-import': 'session_import',
  settings: 'settings_other',
  starmap: 'starmap',
  webhooks: 'webhooks'
}

export function settingsArea(view: string): DesktopFeatureArea {
  return SETTINGS_AREAS[view] ?? 'settings_other'
}

/** The area a full-page route opens, or null for chat/session routes. Contributed
 *  plugin pages (one-segment paths the core does not own) are `extension_page`. */
export function routeArea(pathname: string, contributedPaths: readonly string[] = []): DesktopFeatureArea | null {
  const segment = pathname.replace(/^\/+/, '').split(/[/?#]/)[0] ?? ''

  if (!segment || segment === 'settings') {
    return null
  }

  if (ROUTE_AREAS[segment]) {
    return ROUTE_AREAS[segment]
  }

  return contributedPaths.includes(`/${segment}`) ? 'extension_page' : null
}

export type DesktopFrictionKind =
  'backend_disconnect' | 'error_toast' | 'notice_dismissed' | 'renderer_crash' | 'slow_frame'

/** notifyError's summary rules (store/notifications.ts) → category. */
export type ErrorToastCategory =
  | 'api_key_missing'
  | 'api_key_rejected'
  | 'disk_full'
  | 'gateway_auth_failed'
  | 'method_not_allowed'
  | 'microphone_permission'
  | 'other'
  | 'pool_slot_timeout'
  | 'restart_required'
  | 'rpc_out_of_sync'
  | 'storage_failure'
  | 'timeout'
  | 'unclassified'

export type DesktopNoticeId =
  | 'artifacts_partial_load'
  | 'background_queue_stuck'
  | 'backend_skew'
  | 'billing_banner'
  | 'billing_block'
  | 'build_discontinued'
  | 'client_behind'
  | 'composer_queue_stuck'
  | 'credits'
  | 'free_tier_notice'
  | 'gateway_error'
  | 'gui_skew'
  | 'install_method'
  | 'mcp_health'
  | 'model_warning'
  | 'other'
  | 'restored_draft'
  | 'runtime_not_ready'
  | 'session_compress'
  | 'terminal_backend'
  | 'tip'
  | 'update_available'
  | 'voice_live_unavailable'
  | 'voice_stop_hint'

/** Stable toast ids the Desktop code passes to notify() → notice id. */
const NOTICE_IDS: Record<string, DesktopNoticeId> = {
  'artifacts-partial-load': 'artifacts_partial_load',
  'backend-contract-skew': 'backend_skew',
  'client-update-after-backend': 'client_behind',
  'composer-queue-stuck': 'composer_queue_stuck',
  'desktop-build-discontinued': 'build_discontinued',
  'desktop-update-available': 'update_available',
  'gui-contract-skew': 'gui_skew',
  'install-method-not-supported': 'install_method',
  'runtime-not-ready': 'runtime_not_ready',
  'terminal-backend-unavailable': 'terminal_backend',
  'voice-live-unavailable': 'voice_live_unavailable',
  'voice-stop-hint': 'voice_stop_hint'
}

/** Toast ids with a data suffix (session id, provider, server key): only the prefix counts. */
const NOTICE_PREFIXES: readonly (readonly [string, DesktopNoticeId])[] = [
  ['billing-block:', 'billing_block'],
  ['composer-background-queue-stuck-', 'background_queue_stuck'],
  ['credits.', 'credits'],
  ['gateway-error:', 'gateway_error'],
  ['mcp-health-', 'mcp_health'],
  ['model-warning-confirm-', 'model_warning'],
  ['session-compress:', 'session_compress']
]

export function noticeIdForToast(id: string): DesktopNoticeId {
  return NOTICE_IDS[id] ?? NOTICE_PREFIXES.find(([prefix]) => id.startsWith(prefix))?.[1] ?? 'other'
}

export type RendererCrashDetail = 'crash' | 'killed' | 'oom' | 'other'
export type BackendDisconnectDetail = 'backend_exit' | 'network' | 'other' | 'timeout'
export type SlowFrameBucket = '100ms_to_250ms' | '1s_to_5s' | '250ms_to_1s' | 'gte_5s'

export function slowFrameBucket(durationMs: number): SlowFrameBucket | null {
  if (!(durationMs >= 100)) {
    return null
  }

  return durationMs < 250
    ? '100ms_to_250ms'
    : durationMs < 1000
      ? '250ms_to_1s'
      : durationMs < 5000
        ? '1s_to_5s'
        : 'gte_5s'
}

/** Long-animation-frame histogram buckets: the browser reports frames of 50 ms and up, so the first
 *  band sits below the 100 ms slow_frame threshold; the rest match slowFrameBucket. */
export type LongFrameBucket = '50ms_to_100ms' | SlowFrameBucket

export function longFrameBucket(durationMs: number): LongFrameBucket | null {
  return durationMs >= 50 && durationMs < 100 ? '50ms_to_100ms' : slowFrameBucket(durationMs)
}

export type DesktopFrictionDetail =
  BackendDisconnectDetail | DesktopNoticeId | ErrorToastCategory | RendererCrashDetail | SlowFrameBucket

export type DesktopOnboardingStep =
  | 'choose_later'
  | 'consent'
  | 'first_message'
  | 'free_tier_ready'
  | 'guide'
  | 'guide_skip'
  | 'model_pick'
  | 'provider_api_key'
  | 'provider_local'
  | 'provider_oauth'
  | 'provider_setup'
  | 'sign_in'

export type DesktopOnboardingEvent = 'abandoned' | 'completed' | 'reached'

/** Button presses that dispatch no keybinding action get a stable id here (and
 *  in DESKTOP_BUTTON_ACTION_IDS on the backend). Everything else uses its
 *  KEYBIND_ACTIONS id (lib/keybinds/actions.ts). */
export const DESKTOP_BUTTON_ACTIONS = {
  composerAttach: 'composer.attach',
  messageCopy: 'message.copy',
  messageRetry: 'message.retry'
} as const

export type DesktopActionVia = 'click' | 'menu' | 'palette' | 'shortcut'

/** Built-in action ids (DESKTOP_ACTION_IDS on the backend); anything else — a plugin's palette
 *  command or keybinding, a numbered slot (`profile.switch.3`) — is `other`. */
const ACTION_IDS: ReadonlySet<string> = new Set(
  [
    ...KEYBIND_ACTION_IDS,
    ...KEYBIND_READONLY.map(action => action.id),
    ...Object.values(DESKTOP_BUTTON_ACTIONS)
  ].filter(id => !/\.\d+$/.test(id))
)

export type DesktopDislikeSignal =
  'cancelled' | 'feature_disabled' | 'quick_close' | 'rage_click' | 'setting_off_default' | 'undo'

export type DesktopFlowId =
  | 'command_palette'
  | 'free_tier_sign_in'
  | 'keybind_capture'
  | 'model_picker'
  | 'project_create'
  | 'provider_oauth'
  | 'session_picker'
  | 'session_switcher'

export type DesktopFeatureToggle =
  | 'backdrop'
  | 'bot_activity_toasts'
  | 'composer_popout_gestures'
  | 'intro_splash'
  | 'native_notifications'
  | 'notification_kind'
  | 'reactions'
  | 'thread_timeline'
  | 'tips'
  | 'tours'
  | 'vibe_hearts'

export type DesktopUndoTarget = 'closed_tab' | 'restored_draft'

export type DesktopMode = 'bots' | 'sessions'

// ── limits ────────────────────────────────────────────────────────────────────

const STATE_KEY = 'hermes.desktop.metrics.v1'
/** Friction per (kind, detail) per day; slow frames per bucket per day. */
const FRICTION_DAILY_CAP = 20
const SLOW_FRAME_DAILY_CAP = 5
const DISLIKE_DAILY_CAP = 20
const QUICK_CLOSE_MS = 5000
const RAGE_CLICK_WINDOW_MS = 1000
const RAGE_CLICK_COUNT = 3
/** A gap between interactions longer than this is idle, not active time. */
export const ACTIVE_IDLE_GAP_MS = 5 * 60_000
const PENDING_DAYS_MAX = 7

const LONG_FRAME_BUCKETS: readonly LongFrameBucket[] = [
  '50ms_to_100ms',
  '100ms_to_250ms',
  '250ms_to_1s',
  '1s_to_5s',
  'gte_5s'
]

const QUEUE_MAX = 50
const PRE_CONSENT_MAX = 100
const DROP_SETTLE_MS = 3000

// ── state ─────────────────────────────────────────────────────────────────────

interface ModeDay {
  activeMs: number
  messages: number
}

interface DayAggregate {
  actions: Record<string, number>
  botCount: number
  day: string
  /** Pre-journal payload, retained only until its deterministic copy is durably observed. */
  frames?: Partial<Record<LongFrameBucket, number>>
  visibleMs?: number
  modes: Partial<Record<DesktopMode, ModeDay>>
}

interface MetricsState {
  areas: string[]
  caps: Record<string, number>
  onboarding: { open: Record<string, string>; sent: string[] }
  pending: DayAggregate[]
  today: DayAggregate
  v: 1
}

/** `on`/`off` = the focused profile's collection switch as last read; `null` = not known yet. */
export type DesktopMetricsGate = 'off' | 'on' | null

export const $desktopMetricsGate = atom<DesktopMetricsGate>(null)

/** One id per renderer load: an onboarding step still open from another load was abandoned. */
const LAUNCH_ID = randomUuid()

/** Hash of the bound `connection|profile`: names the local record and main's crash consent. */
let scope = scopeKey('|default')
let request: SharedMetricsRequester | null = null
let queue: [string, Record<string, unknown>][] = []
let flushingDays = false
/** Onboarding transitions before the consent answer (memory only); null once the answer is known. */
let preConsent: (() => void)[] | null = []
let botCount = 0
let lastInteractionAt = 0
const areaOpenedAt = new Map<string, number>()
const openFlows = new Map<DesktopFlowId, boolean>()
let pendingDrop: { detail: BackendDisconnectDetail; timer: ReturnType<typeof setTimeout> } | null = null
let lastBackendExitAt = 0
const recentClicks = new Map<string, number[]>()
let visibleMaxSpanMs = 0
let frameMetrics: DesktopFrameMetrics | null = null
let gateOwner = 0

/** FNV-1a: the local key never carries a profile name. */
function scopeKey(value: string): string {
  let hash = 0x811c9dc5

  for (let i = 0; i < value.length; i++) {
    hash = Math.imul(hash ^ value.charCodeAt(i), 0x01000193)
  }

  return (hash >>> 0).toString(16).padStart(8, '0')
}

function stateKey(): string {
  return `${STATE_KEY}:${scope}`
}

function utcDay(now = Date.now()): string {
  return new Date(now).toISOString().slice(0, 10)
}

function emptyDay(day: string, botCount: number): DayAggregate {
  return { actions: {}, botCount, day, modes: {} }
}

/** Anything for `shared_metrics.desktop_daily` (button presses, mode use). */
function dayHasDailyUse(day: DayAggregate): boolean {
  return (
    Object.values(day.actions).some(n => n > 0) ||
    Object.values(day.modes).some(mode => (mode?.activeMs ?? 0) > 0 || (mode?.messages ?? 0) > 0)
  )
}

function finite(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0
}

function readDay(raw: unknown, collector: DesktopFrameMetrics): DayAggregate | null {
  if (!raw || typeof raw !== 'object' || typeof (raw as DayAggregate).day !== 'string') {
    return null
  }

  const source = raw as Partial<DayAggregate>
  const actions: Record<string, number> = {}

  for (const [key, count] of Object.entries(source.actions ?? {})) {
    actions[key] = finite(count)
  }

  const modes: DayAggregate['modes'] = {}

  for (const mode of ['bots', 'sessions'] as const) {
    const entry = source.modes?.[mode]

    if (entry) {
      modes[mode] = { activeMs: finite(entry.activeMs), messages: finite(entry.messages) }
    }
  }

  const frames: Partial<Record<LongFrameBucket, number>> = {}

  for (const bucket of LONG_FRAME_BUCKETS) {
    const count = finite(source.frames?.[bucket])

    if (count > 0) {
      frames[bucket] = count
    }
  }

  const visibleMs = finite(source.visibleMs)

  const legacy = visibleMs > 0 || Object.keys(frames).length > 0
    ? { frames, visibleMs }
    : null

  const adopted = legacy ? collector.adopt(source.day as string, legacy) : true

  return {
    actions,
    botCount: finite(source.botCount),
    day: source.day as string,
    ...(!adopted && legacy ? legacy : {}),
    modes
  }
}

function readState(key: string, collector: DesktopFrameMetrics, observedBotCount: number): MetricsState {
  const raw = readJson<Partial<MetricsState>>(key)
  const today = (raw && readDay(raw.today, collector)) || emptyDay(utcDay(), observedBotCount)

  return {
    areas: Array.isArray(raw?.areas) ? raw.areas.filter(a => typeof a === 'string') : [],
    caps: raw?.caps && typeof raw.caps === 'object' ? { ...raw.caps } : {},
    onboarding: {
      open: raw?.onboarding?.open && typeof raw.onboarding.open === 'object' ? { ...raw.onboarding.open } : {},
      sent: Array.isArray(raw?.onboarding?.sent) ? raw.onboarding.sent.filter(s => typeof s === 'string') : []
    },
    pending: Array.isArray(raw?.pending)
      ? raw.pending.map(day => readDay(day, collector)).filter((day): day is DayAggregate => day !== null)
      : [],
    today,
    v: 1
  }
}

/** Close out a finished day (into `pending`) when the UTC date moved; true when it did. */
function rollDay(current: MetricsState, now: number, observedBotCount: number): boolean {
  const day = utcDay(now)

  if (current.today.day === day) {
    return false
  }

  if (dayHasDailyUse(current.today) || current.today.frames || current.today.visibleMs) {
    current.pending = [...current.pending, current.today].slice(-PENDING_DAYS_MAX)
  }

  current.today = emptyDay(day, observedBotCount)
  current.areas = []
  current.caps = {}

  return true
}

/** Serialize the complete read/modify/write with profile retirement. Callbacks are synchronous;
 * effects run after ownership is released. Network receipts must pass their original collector. */
function withState<T = void>(
  fn?: (current: MetricsState) => T,
  after?: (result: T) => void,
  collector = frameMetrics
): Promise<T | undefined> | undefined {
  if ($desktopMetricsGate.get() !== 'on' || !collector?.isCurrent() || frameMetrics !== collector) {
    return undefined
  }

  const owner = scope
  const key = stateKey()
  const observedBotCount = botCount
  const currentOwner = () => scope === owner && frameMetrics === collector && $desktopMetricsGate.get() === 'on'

  // One phase publishes both the observed visible-time boundary and its aggregate mutation.
  // A later view switch does not discard already queued observations from this valid cohort.
  return collector.checkpoint(Date.now(), () => {
    const current = readState(key, collector, observedBotCount)

    rollDay(current, Date.now(), observedBotCount)
    const result = fn?.(current)

    writeJson(key, current)

    return result
  })?.then(result => {
    if (!currentOwner() || !collector.isCurrent()) {
      return undefined
    }

    if (result !== undefined) {
      after?.(result)
    }

    return result
  })
}

function send(method: string, params: Record<string, unknown>): void {
  if ($desktopMetricsGate.get() !== 'on') {
    return
  }

  if (!request) {
    queue = [...queue, [method, params] as [string, Record<string, unknown>]].slice(-QUEUE_MAX)

    return
  }

  void request(method, params).catch(() => undefined)
}

/** Take one slot of a per-day cap; false once it is spent. */
function takeCap(current: MetricsState, key: string, cap: number): boolean {
  const used = current.caps[key] ?? 0

  if (used >= cap) {
    return false
  }

  current.caps[key] = used + 1

  return true
}

// ── recording API (every entry point is a silent no-op unless on) ───────────

/** An area opened. Reported once per UTC day; the open time also arms quick_close. */
export function recordFeatureUse(area: DesktopFeatureArea, now = Date.now()): void {
  if ($desktopMetricsGate.get() !== 'on' || !frameMetrics?.isCurrent()) {
    return
  }

  areaOpenedAt.set(area, now)
  void withState(current => {
    if (current.areas.includes(area)) {
      return false
    }

    current.areas = [...current.areas, area]

    return true
  }, fresh => {
    if (fresh) {
      send('shared_metrics.desktop_feature_use', { area })
    }
  })
}

/** An area closed: within QUICK_CLOSE_MS of opening it is a quick_close dislike. */
export function noteAreaClosed(area: DesktopFeatureArea, now = Date.now()): void {
  const openedAt = areaOpenedAt.get(area)

  areaOpenedAt.delete(area)

  if (openedAt !== undefined && now - openedAt < QUICK_CLOSE_MS) {
    recordDislike('quick_close', area)
  }
}

/** Track a boolean open-state for an area: false→true counts use, true→false may be a quick close. */
export function trackArea(area: DesktopFeatureArea, open: boolean, now = Date.now()): void {
  if (open) {
    recordFeatureUse(area, now)
  } else {
    noteAreaClosed(area, now)
  }
}

/** A flow/dialog opened (true) or closed (false). Closing one that never
 *  reached `completeFlow` is a `cancelled` dislike. */
export function trackFlow(flow: DesktopFlowId, open: boolean): void {
  if (open) {
    openFlows.set(flow, false)

    return
  }

  const completed = openFlows.get(flow)

  openFlows.delete(flow)

  if (completed === false) {
    recordDislike('cancelled', flow)
  }
}

/** The open flow did what it was opened for (a palette command ran, a model was picked). */
export function completeFlow(flow: DesktopFlowId): void {
  if (openFlows.has(flow)) {
    openFlows.set(flow, true)
  }
}

/** The primary gateway socket dropped after a healthy boot. Classified a few
 *  seconds later: a backend process exit around it makes it `backend_exit`, a
 *  gateway switch starting in the window (Restart Hermes recycles the backend
 *  before the switch flag rises) cancels it. */
export function noteBackendDrop(reason: 'timeout' | null, now = Date.now()): void {
  if (pendingDrop || $desktopMetricsGate.get() !== 'on') {
    return
  }

  const detail: BackendDisconnectDetail =
    now - lastBackendExitAt < DROP_SETTLE_MS ? 'backend_exit' : (reason ?? 'network')

  pendingDrop = {
    detail,
    timer: setTimeout(() => {
      const settled = pendingDrop

      pendingDrop = null

      if (settled) {
        recordFriction('backend_disconnect', settled.detail)
      }
    }, DROP_SETTLE_MS)
  }
}

/** Main reported the local backend process exited (not during a switch). */
export function noteBackendExited(now = Date.now()): void {
  lastBackendExitAt = now

  if (pendingDrop) {
    pendingDrop.detail = 'backend_exit'
  }
}

/** A deliberate gateway switch began: a drop still settling was part of it. */
export function cancelPendingBackendDrop(): void {
  if (pendingDrop) {
    clearTimeout(pendingDrop.timer)
    pendingDrop = null
  }
}

export function recordFriction(kind: DesktopFrictionKind, detail: DesktopFrictionDetail): void {
  void withState(
    current => takeCap(current, `friction:${kind}:${detail}`, kind === 'slow_frame' ? SLOW_FRAME_DAILY_CAP : FRICTION_DAILY_CAP),
    fresh => {
      if (fresh) {
        send('shared_metrics.desktop_friction', { detail, kind })
      }
    }
  )
}

export function recordDislike(
  signal: DesktopDislikeSignal,
  target: DesktopFeatureArea | DesktopFeatureToggle | DesktopFlowId | DesktopUndoTarget | string,
  setting?: string,
  profile?: null | string
): void {
  void withState(current => takeCap(current, `dislike:${signal}`, DISLIKE_DAILY_CAP), fresh => {
    if (fresh) {
      send('shared_metrics.desktop_dislike', {
        ...(setting ? { setting, signal, target: 'setting' } : { signal, target }),
        ...(profile ? { profile } : {})
      })
    }
  })
}

const MAX_SETTING_KEYS_PER_SAVE = 10

/** Dotted leaf keys of a config patch (`display.show_reasoning`). Arrays are leaves. */
export function configPatchKeys(patch: unknown, prefix = ''): string[] {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) {
    return prefix ? [prefix] : []
  }

  return Object.entries(patch as Record<string, unknown>).flatMap(([key, value]) =>
    configPatchKeys(value, prefix ? `${prefix}.${key}` : key)
  )
}

/** A Settings autosave landed. Only keys the config schema publishes go out (never a key below a
 *  user-named container such as `providers.<name>`); the backend reads the saved values itself and
 *  records whether each moved to or away from its default. */
export function recordSettingsSaved(
  patch: unknown,
  published: Readonly<Record<string, unknown>>,
  profile?: null | string
): void {
  const keys = configPatchKeys(patch).filter(key => Object.hasOwn(published, key))

  for (const key of keys.slice(0, MAX_SETTING_KEYS_PER_SAVE)) {
    recordDislike('setting_off_default', 'setting', key, profile)
  }
}

/** A shipped feature toggled. Only the on→off edge is a signal. */
export function recordFeatureToggle(toggle: DesktopFeatureToggle, wasOn: boolean, isOn: boolean): void {
  if (wasOn && !isOn) {
    recordDislike('feature_disabled', toggle)
  }
}

/** A button/shortcut/palette/menu press, aggregated per day and reported once in the daily report. */
export function recordAction(rawAction: string, via: DesktopActionVia, now = Date.now()): void {
  const action = ACTION_IDS.has(rawAction) ? rawAction : 'other'

  void withState(current => {
    const key = `${action}|${via}`

    current.today.actions[key] = (current.today.actions[key] ?? 0) + 1
  })

  if (via !== 'click' || $desktopMetricsGate.get() !== 'on') {
    return
  }

  const clicks = [...(recentClicks.get(action) ?? []).filter(at => now - at < RAGE_CLICK_WINDOW_MS), now]

  if (clicks.length >= RAGE_CLICK_COUNT) {
    recentClicks.delete(action)
    recordDislike('rage_click', action)
  } else {
    recentClicks.set(action, clicks)
  }
}

/** A user interaction (pointer/key/wheel): active time accrues to the current
 *  Desktop mode for every gap up to ACTIVE_IDLE_GAP_MS; a longer gap was idle. */
export function noteInteraction(now = Date.now(), mode: DesktopMode = $workspaceMode.get()): void {
  const gap = lastInteractionAt ? now - lastInteractionAt : 0

  lastInteractionAt = now

  void withState(current => {
    const entry = (current.today.modes[mode] ??= { activeMs: 0, messages: 0 })

    if (gap > 0 && gap <= ACTIVE_IDLE_GAP_MS) {
      entry.activeMs += gap
    }
  })
}

/** The user sent a message from a session in `mode` (a bot-owned tile is `bots`). */
export function noteMessageSent(mode: DesktopMode = $workspaceMode.get()): void {
  void withState(current => {
    const entry = (current.today.modes[mode] ??= { activeMs: 0, messages: 0 })

    entry.messages += 1

    return current.onboarding.sent.some(key => !key.startsWith('first_message:'))
  }, onboarded => {
    if (onboarded) {
      recordOnboarding('first_message', 'completed')
    }
  })
}

/** A batch of long animation frames seen while the window was visible (durations in ms). */
export function recordLongFrames(durations: readonly number[]): void {
  if ($desktopMetricsGate.get() === 'on') {
    frameMetrics?.record(durations.map(longFrameBucket).filter((bucket): bucket is LongFrameBucket => bucket !== null))
  }
}

/** Main-window lifetime; spans longer than the tick period plus slack are uncredited suspend gaps. */
export function trackDesktopVisibleTime(maxSpanMs: number): () => void {
  visibleMaxSpanMs = maxSpanMs
  frameMetrics?.track(maxSpanMs)

  return () => {
    frameMetrics?.track(0)
    visibleMaxSpanMs = 0
  }
}

/** Configured bots (Bot Mode roster = Hermes profiles) — a count, never names. */
export function setDesktopBotCount(count: number): void {
  const observed = Math.max(0, Math.floor(finite(count)))

  botCount = observed
  void withState(current => {
    current.today.botCount = Math.max(current.today.botCount, observed)
  })
}

/** Hold a first-run transition until the consent answer: true when held (or dropped: no is known). */
function heldForConsent(replay: () => void): boolean {
  if ($desktopMetricsGate.get() === 'on') {
    return false
  }

  if (preConsent && preConsent.length < PRE_CONSENT_MAX) {
    preConsent.push(replay)
  }

  return true
}

function updateOnboarding(current: MetricsState, step: DesktopOnboardingStep, event: DesktopOnboardingEvent): boolean {
  if (event === 'reached') {
    current.onboarding.open[step] = LAUNCH_ID
  } else {
    delete current.onboarding.open[step]
  }

  const key = `${step}:${event}`
  const fresh = !current.onboarding.sent.includes(key)

  if (fresh) {
    current.onboarding.sent.push(key)
  }

  return fresh
}

/** One first-run step transition, once per (step, event) per profile. A step
 *  reached but not completed is remembered; if it is still open on a later
 *  load, it was abandoned. */
export function recordOnboarding(step: DesktopOnboardingStep, event: DesktopOnboardingEvent): void {
  if (heldForConsent(() => recordOnboarding(step, event))) {
    return
  }

  void withState(current => updateOnboarding(current, step, event), fresh => {
    if (fresh) {
      send('shared_metrics.desktop_onboarding', { event, step })
    }
  })
}

/** A step ended without an event of its own (skipped past, cancelled): no
 *  longer open, so the next launch does not call it abandoned. */
export function closeOnboardingStep(step: DesktopOnboardingStep): void {
  if (heldForConsent(() => closeOnboardingStep(step))) {
    return
  }

  void withState(current => {
    const open = { ...current.onboarding.open }

    delete open[step]
    current.onboarding.open = open
  })
}

// ── lifecycle ─────────────────────────────────────────────────────────────────

function reportAbandonedSteps(): void {
  void withState(current => {
    const abandoned: DesktopOnboardingStep[] = []

    for (const [step, launch] of Object.entries(current.onboarding.open)) {
      if (launch !== LAUNCH_ID && updateOnboarding(current, step as DesktopOnboardingStep, 'abandoned')) {
        abandoned.push(step as DesktopOnboardingStep)
      }
    }

    return abandoned
  }, abandoned => {
    for (const step of abandoned) {
      send('shared_metrics.desktop_onboarding', { event: 'abandoned', step })
    }
  })
}

function flushFrameDays(owner: string, requester: SharedMetricsRequester, collector: DesktopFrameMetrics | null): void {
  if (collector && scope === owner && request === requester && $desktopMetricsGate.get() === 'on') {
    void collector.flush(requester, () =>
      frameMetrics === collector && scope === owner && request === requester && $desktopMetricsGate.get() === 'on')
  }
}

async function flushPendingDays(): Promise<void> {
  if (flushingDays || !request) {
    return
  }

  flushingDays = true

  try {
    // Pinned to the profile and gateway that started the flush: a switch mid-await stops it.
    const [owner, requester, collector] = [scope, request, frameMetrics]

    await collector?.ready

    if (scope !== owner || request !== requester || frameMetrics !== collector) {
      return
    }

    for (
      let day = await withState(current => current.pending[0], undefined, collector);
      day;
      day = await withState(current => current.pending[0], undefined, collector)
    ) {
      const reported = day.day

      const settled = () => scope === owner && request === requester && frameMetrics === collector &&
        $desktopMetricsGate.get() === 'on' && collector?.isCurrent() === true

      if (!settled()) {
        return
      }

      if (dayHasDailyUse(day)) {
        if (!(await sendDaily(requester, day)) || !settled()) {
          return
        }

        // A consent withdrawal purges the entry, including an off→on flip during the await.
        const kept = await withState(current => {
          const entry = current.pending.find(pending => pending.day === reported)

          if (entry) {
            entry.actions = {}
            entry.modes = {}
          }

          return entry !== undefined
        }, undefined, collector)

        if (!kept) {
          return
        }
      }

      if (day.frames || day.visibleMs) {
        const adopted = collector?.adopt(day.day, { frames: day.frames ?? {}, visibleMs: day.visibleMs ?? 0 })

        if (!adopted) {
          break
        }
      }

      await withState(current => {
        current.pending = current.pending.filter(entry => entry.day !== reported)
      }, undefined, collector)
    }

    flushFrameDays(owner, requester, collector)
  } catch {
    // A flap: the day stays pending for the next attach.
  } finally {
    flushingDays = false
  }
}

async function sendDaily(requester: SharedMetricsRequester, day: DayAggregate): Promise<boolean> {
  const result = await requester<{ recorded?: boolean }>('shared_metrics.desktop_daily', {
    actions: Object.entries(day.actions)
      .filter(([, count]) => count > 0)
      .map(([key, count]) => {
        const at = key.lastIndexOf('|')

        return { action: key.slice(0, at), count, via: key.slice(at + 1) }
      }),
    bot_count: day.botCount,
    day: day.day,
    modes: (['bots', 'sessions'] as const)
      .filter(mode => day.modes[mode])
      .map(mode => ({ active_ms: day.modes[mode]!.activeMs, messages_sent: day.modes[mode]!.messages, mode }))
  })

  return result?.recorded === true
}

/** Renderer crashes main persisted while this renderer (or its predecessor) was gone. */
async function drainRendererCrashes(): Promise<void> {
  const bridge = window.hermesDesktop?.desktopMetrics

  try {
    const pending = await bridge?.takeRendererCrashes?.()

    if (!pending) {
      return
    }

    let sent = false

    try {
      for (const reason of pending.reasons) {
        recordFriction('renderer_crash', reason)
      }

      sent = $desktopMetricsGate.get() === 'on'
    } finally {
      await bridge?.ackRendererCrashes?.(sent)
    }
  } catch {
    // Telemetry never surfaces.
  }
}

/** Send everything waiting: queued events, finished days, abandoned steps, crashes. */
export function flushDesktopMetrics(): void {
  if ($desktopMetricsGate.get() !== 'on' || !request || !frameMetrics?.isCurrent()) {
    return
  }

  const waiting = queue

  queue = []

  for (const [method, params] of waiting) {
    send(method, params)
  }

  reportAbandonedSteps()
  void flushPendingDays()
  void drainRendererCrashes()
}

/** Main records this window's renderer crashes only while its profile is on; off drops that profile's. */
function tellMain(on: boolean): void {
  try {
    Promise.resolve(window.hermesDesktop?.desktopMetrics?.setEnabled?.(on, scope)).catch(() => undefined)
  } catch {
    // An older shell without the bridge.
  }
}

function forgetInMemory(): void {
  gateOwner += 1
  queue = []
  areaOpenedAt.clear()
  openFlows.clear()
  recentClicks.clear()
  cancelPendingBackendDrop()

  frameMetrics?.stop()
  frameMetrics = null
}

function closeDesktopMetricsGate(decided: boolean): void {
  if (decided) {
    preConsent = null
  }

  $desktopMetricsGate.set('off')
  forgetInMemory()
  tellMain(false)
}

/** Capture before an asynchronous consent read/write. A withdrawal owns its original permit
 * even after a rename or focus change; a positive result never authorizes a newer bound owner. */
export async function captureDesktopMetricsGate(profileScope?: string): Promise<(
  next: DesktopMetricsGate,
  decided?: boolean
) => Promise<boolean>> {
  const ownerScope = profileScope === undefined ? scope : scopeKey(profileScope)
  const key = `${STATE_KEY}:${ownerScope}`
  const collector = ownerScope === scope ? frameMetrics : null
  const owner = gateOwner
  const captured = await DesktopFrameMetrics.captureCoordinated(key)
  const ownsView = () => ownerScope === scope && owner === gateOwner && collector === frameMetrics
  const sameCohort = (value: DesktopFrameConsent) => value.epoch === captured.epoch && value.consent === captured.consent

  async function withdrawCapturedConsent(decided: boolean): Promise<boolean> {
    const currentOwner = ownsView() && sameCohort(DesktopFrameMetrics.capture(key))

    if (currentOwner) {
      closeDesktopMetricsGate(decided)
    }

    const closedOwner = gateOwner

    const revoked = await DesktopFrameMetrics.purge(key, captured)
    const current = DesktopFrameMetrics.capture(key)
    const after = revoked.after ?? captured

    // The receipt distinguishes our own epoch retirement from another renderer replacing
    // this same scope while we awaited ownership. A stale Off still revokes its old permit.
    return currentOwner && ownerScope === scope && gateOwner === closedOwner && frameMetrics === null &&
      (!revoked.before || sameCohort(revoked.before)) &&
      current.epoch === after.epoch && current.consent === after.consent &&
      (!revoked.after || current.active === revoked.after.active)
  }

  return async (next, decided = true) => {
    if (next === 'off') {
      return withdrawCapturedConsent(decided)
    }

    // Validation must not mint a replacement cohort for an old asynchronous result.
    const current = await DesktopFrameMetrics.captureCoordinated(key, false)

    if (!ownsView() || current.epoch !== captured.epoch || current.consent !== captured.consent ||
      (captured.active && !current.active)) {
      return false
    }

    setDesktopMetricsGate(next, captured)
    const acceptedOwner = gateOwner
    const acceptedCollector = frameMetrics

    await acceptedCollector?.ready

    if (ownerScope !== scope || gateOwner !== acceptedOwner || frameMetrics !== acceptedCollector) {
      return false
    }

    // A failed telemetry initializer is not a failed consent save. Only a newer persisted
    // owner/withdrawal invalidates the accepted view, not storage or coordinator availability.
    if (next === 'on' && !acceptedCollector?.isCurrent()) {
      const settled = DesktopFrameMetrics.capture(key)

      return settled.epoch === captured.epoch && settled.consent === captured.consent &&
        (!captured.active || settled.active)
    }

    return true
  }
}

/** Only an accepted, captured consent result may create a collector. */
function setDesktopMetricsGate(next: Exclude<DesktopMetricsGate, 'off'>, captured: DesktopFrameConsent): void {
  if (next === $desktopMetricsGate.get()) {
    if (next !== 'on' || frameMetrics?.isCurrent()) {
      return
    }

    forgetInMemory()
  }

  $desktopMetricsGate.set(next)

  if (next !== 'on') {
    frameMetrics?.stop()
    frameMetrics = null
  }

  if (next === 'on') {
    const held = preConsent ?? []

    preConsent = null
    gateOwner += 1
    frameMetrics = new DesktopFrameMetrics(stateKey(), captured)
    frameMetrics.track(visibleMaxSpanMs)
    tellMain(true)
    flushDesktopMetrics()
    held.forEach(replay => replay())
  }
}

/** The requester for the focused profile's gateway (null while it is down: events queue in memory).
 *  `profileScope` (`connection|profile`) other than the bound one forgets the gate until that
 *  profile's switch is read. */
export function bindDesktopMetrics(next: SharedMetricsRequester | null, profileScope?: string): void {
  const nextScope = profileScope === undefined ? scope : scopeKey(profileScope)

  if (nextScope !== scope) {
    forgetInMemory()
    scope = nextScope
    preConsent = []
    $desktopMetricsGate.set(null)
  }

  request = next
  flushDesktopMetrics()
}

/** Profile deletion also revokes sleeping windows before their next frame checkpoint. */
export function dropDesktopMetricsForProfile(profile: string, connectionId = 'local'): void {
  for (const connection of connectionId === 'local' ? ['', 'local'] : [connectionId]) {
    const key = `${STATE_KEY}:${scopeKey(`${connection}|${profile}`)}`

    if (key === stateKey()) {
      forgetInMemory()
      $desktopMetricsGate.set(null)
    }

    void DesktopFrameMetrics.purge(key)
  }
}

/** Renames are local, like migrateTilesForProfile. Move the whole family without retaining names. */
export async function migrateDesktopMetricsForProfile(from: string, to: string): Promise<void> {
  if (from === to) {
    return
  }

  for (const connection of ['', 'local']) {
    const oldKey = `${STATE_KEY}:${scopeKey(`${connection}|${from}`)}`
    const newKey = `${STATE_KEY}:${scopeKey(`${connection}|${to}`)}`

    if (oldKey === stateKey() || newKey === stateKey()) {
      forgetInMemory()
      $desktopMetricsGate.set(null)
    }

    await DesktopFrameMetrics.migrate(oldKey, newKey)
  }
}

/** Periodic/visibility tick: roll the day over and send what finished. */
export function tickDesktopMetrics(): void {
  void withState()
  void flushPendingDays()
}

/** Test-only: forget in-memory state (localStorage is the test's to clear). */
export function resetDesktopMetricsForTests(): void {
  scope = scopeKey('|default')
  preConsent = []
  request = null
  queue = []
  flushingDays = false
  botCount = 0
  lastInteractionAt = 0
  frameMetrics?.stop()
  frameMetrics = null
  visibleMaxSpanMs = 0
  areaOpenedAt.clear()
  openFlows.clear()
  recentClicks.clear()
  cancelPendingBackendDrop()
  lastBackendExitAt = 0
  $desktopMetricsGate.set(null)
}

/** Prefix of the per-profile local record (`<prefix>:<scope hash>`). */
export const DESKTOP_METRICS_STATE_KEY = STATE_KEY
