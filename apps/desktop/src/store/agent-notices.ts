import type { NativeNotificationInput } from '@/store/native-notifications'
import { dismissNotification, type NotificationInput, type NotificationKind, notify } from '@/store/notifications'

/**
 * Wire shape of a `notification.show` payload — the driver-agnostic
 * `AgentNotice` spine (`agent/credits_tracker.py`) as forwarded by
 * `tui_gateway/server.py`. Snake_case to match the wire; the `text` already
 * carries its own leading glyph (• ⚠ ✕ ✓) from the Python policy, so a toast
 * NEVER adds another icon on top.
 *
 * - `level` is severity: info | warn | error | success.
 * - `kind` is lifetime: `sticky` (stays until an explicit clear) or `ttl`
 *   (self-expires after `ttl_ms`).
 */
export interface AgentNoticePayload {
  text?: string
  level?: string
  kind?: string
  ttl_ms?: null | number
  key?: string
  id?: string
}

const LEVEL_TO_TOAST_KIND: Record<string, NotificationKind> = {
  error: 'error',
  info: 'info',
  success: 'success',
  warn: 'warning'
}

/**
 * Map an agent notice to a toast input, or `null` when it carries no text.
 *
 * Pure and side-effect free so it can be unit-tested directly. The mapping is
 * the whole contract:
 * - `level` → toast kind (info/warn/error/success, warn→warning).
 * - `sticky` → `durationMs: 0` (persists); `ttl` → `durationMs: ttl_ms`.
 * - the notice `key` doubles as the toast `id`, so re-emitting the same key
 *   REPLACES the prior toast — the credits 50→75→90 line escalates in place
 *   instead of stacking, and a key-matched `notification.clear` can dismiss it.
 */
export function noticeToToast(payload: AgentNoticePayload | undefined): NotificationInput | null {
  const text = payload?.text?.trim()

  if (!text) {
    return null
  }

  const isTtl = payload?.kind === 'ttl'
  const ttl = isTtl && typeof payload?.ttl_ms === 'number' && payload.ttl_ms > 0 ? payload.ttl_ms : undefined

  return {
    // sticky → 0 (never auto-dismiss); ttl with a ttl_ms → that value; a ttl
    // without a usable ttl_ms falls back to notify()'s per-kind default.
    durationMs: isTtl ? ttl : 0,
    id: payload?.key || payload?.id,
    kind: LEVEL_TO_TOAST_KIND[payload?.level ?? 'info'] ?? 'info',
    message: text
  }
}

/** Render a `notification.show` notice as a toast (no-op when it has no text). */
export function showAgentNotice(payload: AgentNoticePayload | undefined): void {
  const toast = noticeToToast(payload)

  if (toast) {
    notify(toast)
  }
}

/**
 * Dismiss the toast a `notification.clear` targets. The clear only ever names a
 * `key`, which we used as the toast id, so this is a key-matched dismissal.
 */
export function clearAgentNotice(key: string | undefined): void {
  if (key) {
    dismissNotification(key)
  }
}

// Only these two credit notices are urgent enough to break through as a native
// OS notification (when Hermes is backgrounded). The escalating usage line
// (`credits.usage`) and the grant-spent notice stay in-app toasts only — they
// aren't worth interrupting the user's OS for.
const NATIVE_NOTICE_KEYS = new Set(['credits.depleted', 'credits.restored'])

/**
 * Map a notice to a native OS notification input, or `null` when it isn't one of
 * the urgent credit notices. Pure — the caller passes the localized `title` and
 * decides whether to dispatch. `global: true` because credit state is
 * account-wide, not tied to a chat session, so it should fire whenever the user
 * is away regardless of which session (if any) is focused. The notice `text`
 * already carries its glyph and is passed through as the raw body.
 */
export function nativeNoticeInput(
  payload: AgentNoticePayload | undefined,
  title: string
): NativeNotificationInput | null {
  const text = payload?.text?.trim()

  if (!text || !payload?.key || !NATIVE_NOTICE_KEYS.has(payload.key)) {
    return null
  }

  return {
    body: text,
    global: true,
    kind: 'credits',
    title
  }
}
