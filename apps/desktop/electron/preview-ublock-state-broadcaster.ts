import type { PreviewUblockState } from './preview-ublock'

export interface PreviewUblockStateWindow {
  isDestroyed(): boolean
  webContents: {
    isDestroyed(): boolean
    send(channel: string, state: PreviewUblockState): void
  }
}

export type PreviewUblockStateLog = (message: string) => void

export function broadcastPreviewUblockStateToWindows(
  windows: readonly PreviewUblockStateWindow[],
  state: PreviewUblockState,
  log: PreviewUblockStateLog
): void {
  for (const window of windows) {
    try {
      if (window.isDestroyed() || window.webContents.isDestroyed()) {
        continue
      }

      window.webContents.send('hermes:preview-ublock:state', state)
    } catch {
      // Broadcasting is presentation-only. A window can be destroyed between
      // the checks and send(), and other windows must still receive the state.
      log('[preview] uBlock state broadcast failed for one window')
    }
  }
}
