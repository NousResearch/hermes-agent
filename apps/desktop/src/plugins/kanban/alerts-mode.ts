/**
 * Kanban alerts mode (#123596): how a terminal worker event reaches the user.
 *
 *  - `toast` (default): today's in-app toast + native OS notification when away.
 *  - `quiet`: bottom-right auto-dismissing toast + the completion sound; the
 *    OS notification still fires when away.
 *  - `badge`: nothing interrupts; only the unseen count on the Kanban nav row
 *    and in the board switcher.
 *
 * A leaf module (SDK imports only) so both the delivery path and the UI read
 * it without an import cycle through `api.ts`, which hydrates it.
 */

import { atom } from '@hermes/plugin-sdk'

export type KanbanAlertsMode = 'badge' | 'quiet' | 'toast'

export const ALERTS_MODES: readonly KanbanAlertsMode[] = ['toast', 'quiet', 'badge']

/** `ctx.storage` key. Deliberately unsuffixed — DEVICE-GLOBAL, like the
 *  native-notification toggles and the completion-sound choice: it is
 *  presentation, and the board it describes is shared across profiles. */
export const ALERTS_MODE_KEY = 'alertsMode'

/** Anything that is not a known mode (missing, corrupt, a future value) is `toast`. */
export function parseAlertsMode(value: unknown): KanbanAlertsMode {
  return ALERTS_MODES.includes(value as KanbanAlertsMode) ? (value as KanbanAlertsMode) : 'toast'
}

export const $alertsMode = atom<KanbanAlertsMode>('toast')
