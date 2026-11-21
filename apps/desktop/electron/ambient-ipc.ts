import { ipcMain } from 'electron'

import { isDoNotDisturbActive } from './do-not-disturb'
import { createAmbientClaimArbiter } from './event-dedupe'

// Ambient cues (turn-end chime, spoken replies): which window plays one, and
// whether the OS is in Do Not Disturb so the renderer stays quiet.
export function registerAmbientCueIpc(): void {
  // Speech claims outlive instant cues so throttled peer windows cannot replay a reply.
  const ownsAmbientCue = createAmbientClaimArbiter()
  ipcMain.handle('hermes:ambient:claim', (_event, key: unknown): boolean => ownsAmbientCue(String(key ?? '')))
  ipcMain.handle('hermes:do-not-disturb:active', (): Promise<boolean> => isDoNotDisturbActive())
}
