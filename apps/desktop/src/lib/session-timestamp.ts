import type { SessionInfo } from '@/types/hermes'

export const JS_DATE_MAX_UNIX_SECONDS = 8_640_000_000_000

/** Admit only finite Unix-second values that ECMAScript Date can render. */
export function dateRepresentableUnixSeconds(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= JS_DATE_MAX_UNIX_SECONDS ? value : null
}

/**
 * Shared session-list recency contract. A zero last_active remains unset and
 * falls back to a safe started_at, matching the sidebar row's display rule.
 */
export function sessionListRecencySeconds(session: Pick<SessionInfo, 'last_active' | 'started_at'>): number {
  const active = dateRepresentableUnixSeconds(session.last_active)
  if (active !== null && active !== 0) {
    return active
  }
  return dateRepresentableUnixSeconds(session.started_at) ?? 0
}
