import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'

// Renderer-only preference: the composer's hands-free auto-send. The backend
// never reads it, so it lives with the renderer (localStorage) rather than in
// config.yaml.
const ENABLED_KEY = 'hermes.desktop.autoSendIdle'
const DELAY_KEY = 'hermes.desktop.autoSendIdleDelay'

/** Delay choices offered in Settings. Stored as a string id; ms at the edge. */
export const AUTO_SEND_DELAY_IDS = ['1500', '2000', '3000', '5000'] as const
export type AutoSendDelayId = (typeof AUTO_SEND_DELAY_IDS)[number]
export const AUTO_SEND_DEFAULT_DELAY_ID: AutoSendDelayId = '2000'

/** Off by default: an idle auto-send is destructive when the user did not ask for it. */
export const $autoSendIdleEnabled = atom<boolean>(storedBoolean(ENABLED_KEY, false))

/** Validate the stored id against the current choices, so a stale/foreign value falls back. */
const readDelayId = (): AutoSendDelayId => {
  const raw = storedString(DELAY_KEY)
  return AUTO_SEND_DELAY_IDS.find(id => id === raw) ?? AUTO_SEND_DEFAULT_DELAY_ID
}

export const $autoSendIdleDelayMs = atom<number>(Number(readDelayId()))

export function setAutoSendIdleEnabled(enabled: boolean): void {
  persistBoolean(ENABLED_KEY, enabled)
  $autoSendIdleEnabled.set(enabled)
}

export function setAutoSendIdleDelayId(id: AutoSendDelayId): void {
  persistString(DELAY_KEY, id)
  $autoSendIdleDelayMs.set(Number(id))
}
