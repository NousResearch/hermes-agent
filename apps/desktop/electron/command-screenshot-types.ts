export interface ScreenshotStatus {
  enabled: boolean
  state: 'disabled' | 'starting' | 'ready' | 'input-permission' | 'screen-permission' | 'unavailable'
}

export interface ScreenshotWindow {
  windowId: number
  width: number
  height: number
}

export type ScreenshotResult =
  | { ok: true; png: Uint8Array }
  | { ok: false; reason: 'expired' | 'screen-permission' | 'unavailable' }

export interface ScreenshotApi {
  getSettings(): Promise<ScreenshotStatus>
  setEnabled(enabled: boolean): Promise<ScreenshotStatus>
  openPermissionSettings(kind: 'input' | 'screen'): Promise<void>
  onStatus(callback: (status: ScreenshotStatus) => void): () => void
  onRequest(callback: (requestId: string) => void): () => void
  capture(requestId: string): Promise<ScreenshotResult>
}
