import { atom } from 'nanostores'

import type { GatewayEventPayload } from '@/lib/chat-messages'

/**
 * Notification mode for background process completions, mirroring the
 * gateway's display.background_process_notifications semantics
 * (gateway/run.py): `false` → off, unknown values → all.
 */
export type ProcessNotificationMode = 'all' | 'error' | 'off' | 'result'

const MODES = new Set<ProcessNotificationMode>(['all', 'error', 'off', 'result'])

export function normalizeProcessNotificationMode(raw: unknown): ProcessNotificationMode {
  if (raw === false) {
    return 'off'
  }

  const mode = typeof raw === 'string' ? raw.trim().toLowerCase() : ''

  return MODES.has(mode as ProcessNotificationMode) ? (mode as ProcessNotificationMode) : 'all'
}

/** Synced from hermes config (display.background_process_notifications). */
export const $processNotificationsMode = atom<ProcessNotificationMode>('all')

export const setProcessNotificationsMode = (raw: unknown) =>
  $processNotificationsMode.set(normalizeProcessNotificationMode(raw))

const BODY_MAX_CHARS = 140

/**
 * Native-toast content for a status.update (kind=process) gateway event, or
 * null when the mode/event combination shouldn't notify. Only completion
 * events surface as OS notifications — watch matches stay in-chat (they can
 * fire many times per process and would spam the notification center).
 */
export function processCompletionToast(
  payload: GatewayEventPayload,
  mode: ProcessNotificationMode
): { body: string; title: string } | null {
  if (mode === 'off' || (payload.event_type ?? 'completion') !== 'completion') {
    return null
  }

  const failed = typeof payload.exit_code === 'number' && payload.exit_code !== 0

  if (mode === 'error' && !failed) {
    return null
  }

  const command = (payload.command ?? '').trim()

  return {
    title: failed ? `Background process failed (exit ${payload.exit_code})` : 'Background process finished',
    body: (command || payload.text || '').slice(0, BODY_MAX_CHARS)
  }
}
