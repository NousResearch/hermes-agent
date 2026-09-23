import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import { isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import { isLocalBackendSlotWaitTimeout, requestPoolLimitsSettings } from '@/store/pool-limits'
import { requestBackendRestart, requestRoute } from '@/store/recovery-requests'

export type NotificationKind = 'error' | 'warning' | 'info' | 'success'

export interface NotificationAction {
  label: string
  onClick: () => void
}

export type NotificationPlacement = 'default' | 'bottom-right'

export interface AppNotification {
  id: string
  kind: NotificationKind
  /** When set, renders this codicon instead of the default kind icon. */
  icon?: string
  /** When set, tints the icon and message with this CSS color (severity ramp). */
  accentColor?: string
  /** Secondary detail line rendered below the message, muted (e.g. "$220.00 cap"). */
  meta?: string
  title?: string
  message: string
  detail?: string
  action?: NotificationAction
  /** Second, quieter button beside `action` (e.g. "Disable" next to "Sign in"). */
  secondaryAction?: NotificationAction
  onDismiss?: () => void
  createdAt: number
  placement?: NotificationPlacement
}

export interface NotificationInput {
  id?: string
  kind?: NotificationKind
  icon?: string
  accentColor?: string
  meta?: string
  title?: string
  message: string
  detail?: string
  action?: NotificationAction
  secondaryAction?: NotificationAction
  onDismiss?: () => void
  durationMs?: number
  placement?: NotificationPlacement
}

let notificationCounter = 0
const timers = new Map<string, number>()

export const $notifications = atom<AppNotification[]>([])

function defaultDuration(kind: NotificationKind) {
  if (kind === 'error' || kind === 'warning') {
    return 0
  }

  return 5_000
}

// Only interruptions worth a top-center toast: errors, warnings, and anything
// with an action button the user needs to notice and click (restart gateway,
// update available, sign-in prompts). Everything else — the bulk of routine
// "saved"/"enabled"/"archived" confirmations across settings, MCP, cron,
// profiles, messaging — is ambient feedback and defaults to a quiet
// bottom-right toast instead. Callers can still force `placement: 'default'`
// for a specific case.
function defaultPlacement(kind: NotificationKind, action?: NotificationAction): NotificationPlacement {
  if (kind === 'error' || kind === 'warning' || action) {
    return 'default'
  }

  return 'bottom-right'
}

function cleanErrorText(value: string) {
  return value.replace(/^Error:\s*/, '').trim()
}

/** True when an error string is a disk-full / ENOSPC / SQLITE_FULL failure. */
export function isDiskFullErrorMessage(message: string): boolean {
  return (
    /no space left on device/i.test(message) ||
    /not enough space/i.test(message) ||
    /database or disk is full/i.test(message) ||
    /\bENOSPC\b/i.test(message) ||
    /disk full/i.test(message) ||
    /full disk/i.test(message)
  )
}

/** Settings deep links the summariser can attach to a toast. */
const KEYS_ROUTE = (envKey: string) => `/settings?tab=keys&key=${encodeURIComponent(envKey)}`
const GATEWAY_SETTINGS_ROUTE = '/settings?tab=gateway'
const MAINTENANCE_ROUTE = '/command-center?section=maintenance'

/** One-click recoveries reused by several rules. */
export const RECOVERY_ACTIONS = {
  openUpdates: (): NotificationAction => ({
    label: translateNow('notifications.updateHermes'),
    onClick: () => void import('@/store/updates').then(({ openUpdatesWindow }) => openUpdatesWindow())
  }),
  restartHermes: (): NotificationAction => ({
    label: translateNow('notifications.actions.restartHermes'),
    onClick: requestBackendRestart
  }),
  openKeys: (envKey: string): NotificationAction => ({
    label: translateNow('notifications.actions.openKeys'),
    onClick: () => requestRoute(KEYS_ROUTE(envKey))
  }),
  openGateways: (): NotificationAction => ({
    label: translateNow('notifications.actions.openGateways'),
    onClick: () => requestRoute(GATEWAY_SETTINGS_ROUTE)
  }),
  openMaintenance: (): NotificationAction => ({
    label: translateNow('notifications.actions.openMaintenance'),
    onClick: () => requestRoute(MAINTENANCE_ROUTE)
  })
}

/** Structured storage failure codes the backend puts in RPC/HTTP error data
 *  (`hermes_state_errors.classify_persistence_error`). */
const STORAGE_CODE_RE = /['"]code['"]\s*:\s*['"](storage_[a-z_]+|disk_full)['"]/i

interface ErrorSummaryRule {
  test: (msg: string) => boolean
  summarize: (msg: string) => string
  /** Recovery button attached to the toast when this rule matches. */
  action?: (msg: string) => NotificationAction
}

const ERROR_SUMMARIES: ErrorSummaryRule[] = [
  {
    // Disk full / ENOSPC — session DB write, backend crash, or any path that
    // bubbles "no space left" / SQLITE_FULL through notifyError. Match before
    // generic length truncation so the user gets a clear "free space" toast
    // instead of a silent send or a raw errno dump.
    test: isDiskFullErrorMessage,
    summarize: () => translateNow('notifications.errors.diskFull'),
    action: () => RECOVERY_ACTIONS.openMaintenance()
  },
  {
    // Any other classified storage failure (locked, corrupt, read-only …):
    // the Maintenance panel runs the doctor that names the fix.
    test: msg => STORAGE_CODE_RE.test(msg),
    summarize: () => translateNow('notifications.errors.storageFailure'),
    action: () => RECOVERY_ACTIONS.openMaintenance()
  },
  {
    test: msg => /['"]code['"]\s*:\s*['"]gateway_auth_failed['"]/i.test(msg),
    summarize: () => translateNow('notifications.errors.gatewayAuthFailed'),
    action: () => RECOVERY_ACTIONS.openGateways()
  },
  {
    test: msg => /incorrect api key provided/i.test(msg) || /['"]code['"]\s*:\s*['"]invalid_api_key['"]/i.test(msg),
    summarize: () => translateNow('notifications.errors.openaiRejectedApiKey'),
    action: () => RECOVERY_ACTIONS.openKeys('OPENAI_API_KEY')
  },
  {
    test: msg => /neither voice_tools_openai_key nor openai_api_key is set/i.test(msg),
    summarize: () => translateNow('notifications.errors.openaiTtsNeedsKey'),
    action: () => RECOVERY_ACTIONS.openKeys('OPENAI_API_KEY')
  },
  {
    test: msg => /ELEVENLABS_API_KEY not set/i.test(msg) || /ElevenLabs STT API error \(HTTP 401\)/i.test(msg),
    summarize: msg =>
      /ELEVENLABS_API_KEY not set/i.test(msg)
        ? translateNow('notifications.errors.elevenLabsNeedsKey')
        : translateNow('notifications.errors.elevenLabsRejectedKey'),
    action: () => RECOVERY_ACTIONS.openKeys('ELEVENLABS_API_KEY')
  },
  {
    test: msg => /method not allowed/i.test(msg),
    summarize: () => translateNow('notifications.errors.methodNotAllowed'),
    action: () => RECOVERY_ACTIONS.restartHermes()
  },
  {
    test: msg => /microphone permission/i.test(msg),
    summarize: () => translateNow('notifications.errors.microphonePermission')
  },
  {
    test: msg => isOutOfSyncRpcParams(msg),
    summarize: () => translateNow('notifications.errors.rpcOutOfSync'),
    action: () => RECOVERY_ACTIONS.openUpdates()
  },
  {
    test: msg => /Restart required:/i.test(msg),
    summarize: () => translateNow('notifications.errors.codeSkewRestartRequired'),
    action: () => RECOVERY_ACTIONS.restartHermes()
  }
]

function summarizeErrorMessage(message: string, fallback: string) {
  const rule = ERROR_SUMMARIES.find(r => r.test(message))

  if (rule) {
    return { action: rule.action?.(message), message: rule.summarize(message) }
  }

  return { action: undefined, message: message.length > 180 ? fallback : message || fallback }
}

// Exported so flows that surface errors inline (e.g. ConfirmDialog's onConfirm
// rethrow) can reuse the same IPC-unwrapping + summarizing as notifyError.
export function readableError(
  error: unknown,
  fallback: string
): { message: string; detail?: string; action?: NotificationAction } {
  const raw = error instanceof Error ? error.message : typeof error === 'string' ? error : fallback
  const unwrapped = raw.match(/Error invoking remote method '[^']+': Error: (.+)$/)?.[1] ?? raw
  const cleaned = cleanErrorText(unwrapped)
  const detail = cleaned.match(/"detail"\s*:\s*"([^"]+)"/)?.[1] ?? cleaned
  const summary = summarizeErrorMessage(detail, fallback)

  return { message: summary.message, detail: detail === summary.message ? undefined : detail, action: summary.action }
}

export function notify(input: NotificationInput): string {
  const kind = input.kind ?? 'info'
  const id = input.id ?? `${Date.now()}-${notificationCounter++}`

  const notification: AppNotification = {
    id,
    kind,
    icon: input.icon,
    accentColor: input.accentColor,
    meta: input.meta,
    title: input.title,
    message: input.message,
    detail: input.detail,
    action: input.action,
    secondaryAction: input.secondaryAction,
    onDismiss: input.onDismiss,
    createdAt: Date.now(),
    placement: input.placement ?? defaultPlacement(kind, input.action)
  }

  window.clearTimeout(timers.get(id))
  timers.delete(id)
  // Visual depth is capped by CardStack, not by discarding queued notifications.
  $notifications.set([notification, ...$notifications.get().filter(item => item.id !== id)])

  const duration = input.durationMs ?? defaultDuration(kind)

  if (duration > 0) {
    timers.set(
      id,
      window.setTimeout(() => dismissNotification(id), duration)
    )
  }

  return id
}

export function notifyError(
  error: unknown,
  fallback: string,
  options: { action?: NotificationAction; id?: string } = {}
): string {
  const readable = readableError(error, fallback)
  const poolSlotTimeout = isLocalBackendSlotWaitTimeout(error)

  return notify({
    action: poolSlotTimeout
      ? {
          label: translateNow('desktop.poolSlotTimeoutOpenSettings'),
          onClick: requestPoolLimitsSettings
        }
      : (options.action ?? readable.action),
    // A caller that can fire again for the same cause names its toast, so the repeat replaces it.
    id: options.id,
    kind: 'error',
    title: fallback,
    message: poolSlotTimeout ? translateNow('desktop.poolSlotTimeoutBody') : readable.message,
    detail: poolSlotTimeout ? readable.message : readable.detail
  })
}

export function dismissNotification(id: string) {
  window.clearTimeout(timers.get(id))
  timers.delete(id)
  const dismissed = $notifications.get().find(item => item.id === id)
  $notifications.set($notifications.get().filter(item => item.id !== id))
  dismissed?.onDismiss?.()
}

export function clearNotifications() {
  for (const timer of timers.values()) {
    window.clearTimeout(timer)
  }

  timers.clear()
  const all = $notifications.get()
  $notifications.set([])

  for (const item of all) {
    item.onDismiss?.()
  }
}
