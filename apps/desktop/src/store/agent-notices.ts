import type { NativeNotificationInput } from '@/store/native-notifications'
import { dismissNotification, type NotificationInput, type NotificationKind, notify } from '@/store/notifications'

/**
 * Wire shape of a `notification.show` payload — the driver-agnostic
 * `AgentNotice` spine (`agent/credits_tracker.py`) as forwarded by
 * `tui_gateway/server.py`. Snake_case to match the wire.
 *
 * The `text` carries its own leading severity glyph (• ⚠ ✕ ✓) from the Python
 * policy — that's how the CLI/TUI render it (glyph in a status line, no separate
 * icon). The desktop toast is different: every toast renders a kind icon, so we
 * strip the leading glyph and let that icon carry severity (see `stripGlyph`),
 * otherwise the toast shows two markers. The native OS notification keeps the
 * glyph (it has no icon of ours).
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

// The severity glyphs the Python notice policy prefixes (`•` `⚠` `✕`/`✗` `✓`),
// optionally with a variation selector, plus trailing space. Stripped for the
// desktop toast because the toast already renders a kind icon.
const LEADING_GLYPH = /^[•⚠✕✗✓]\uFE0F?\s*/u

/** Drop a single leading severity glyph so the toast doesn't double up on it. */
export function stripGlyph(text: string): string {
  return text.replace(LEADING_GLYPH, '')
}

/** A `$12.34` money token, as the Nous notice policy formats amounts. */
const MONEY = /\$(\d+(?:\.\d{2})?)/g

/**
 * Used fraction of the cap derived from a "You've used $X of your $Y cap" notice
 * — the first two money tokens are (used, cap). Returns a value in [0, 1], or
 * `null` when the line has no usable pair.
 */
export function usageFraction(text: string | undefined): null | number {
  const amounts = [...(text ?? '').matchAll(MONEY)].map(m => Number(m[1]))

  if (amounts.length < 2 || !(amounts[1] > 0)) {
    return null
  }

  return amounts[0] / amounts[1]
}

/**
 * Accent color for a credit notice, as a range/severity ramp. The usage gauge
 * keys off how much of the cap is spent ($used / $cap): it stays on the toast's
 * default muted color under 75%, then escalates to orange, then red, as the cap
 * nears. `credits.depleted` is red (paused) and `credits.restored` green.
 *
 * Returns a CSS color token or `undefined` (= keep the default muted color).
 * These reuse the app's existing usage palette (`--ui-*`; see the
 * `--context-usage-*` block in styles.css) — no new colors are introduced.
 */
export function noticeAccent(payload: AgentNoticePayload | undefined): string | undefined {
  if (payload?.key === 'credits.depleted') {
    return 'var(--ui-red)'
  }

  if (payload?.key === 'credits.restored') {
    return 'var(--ui-green)'
  }

  const frac = usageFraction(payload?.text)

  if (frac === null) {
    return undefined
  }

  if (frac >= 0.9) {
    return 'var(--ui-red)'
  }

  if (frac >= 0.75) {
    return 'var(--ui-orange)'
  }

  return undefined
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
  const ttl = typeof payload?.ttl_ms === 'number' && payload.ttl_ms > 0 ? payload.ttl_ms : undefined

  // The Python notice text packs a trailing detail after a middot
  // (`… used · $220.00 cap`, `Grant spent · $12.00 top-up left`). On one CLI/TUI
  // status line that reads fine, but the toast follows the title-plus-description
  // convention (Sonner/shadcn): the primary status is the message and the detail
  // drops to a muted second line, instead of inlining a `·` separator.
  const [primary, meta] = splitMeta(stripGlyph(text))

  return {
    // Icon + text tint by usage band (muted → orange → red); undefined keeps
    // the default muted color.
    accentColor: noticeAccent(payload),
    // sticky → 0 (never auto-dismiss); ttl with a ttl_ms → that value; a ttl
    // without a usable ttl_ms falls back to notify()'s per-kind default.
    durationMs: isTtl ? ttl : 0,
    id: payload?.key || payload?.id,
    kind: LEVEL_TO_TOAST_KIND[payload?.level ?? 'info'] ?? 'info',
    message: primary,
    meta
  }
}

/**
 * Split a notice line into its primary status and a trailing detail on the first
 * ` · ` (space-middot-space). No middot → the whole line is the primary and
 * there's no meta.
 */
export function splitMeta(text: string): [primary: string, meta: string | undefined] {
  const at = text.indexOf(' · ')

  if (at === -1) {
    return [text, undefined]
  }

  return [text.slice(0, at), text.slice(at + 3) || undefined]
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
