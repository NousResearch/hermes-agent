/** Where a command-key capture lands. 'current-draft' keeps today's behavior. */
export type ScreenshotDestination = 'current-draft' | 'new-session'

export interface ScreenshotStatus {
  enabled: boolean
  destination: ScreenshotDestination
  bringToFront: boolean
  state: 'disabled' | 'starting' | 'ready' | 'input-permission' | 'screen-permission' | 'unavailable'
}

/** A settings patch: every field optional, applied over the persisted file. */
export interface ScreenshotSettingsPatch {
  enabled?: boolean
  destination?: ScreenshotDestination
  bringToFront?: boolean
}

export interface ScreenshotWindow {
  windowId: number
  width: number
  height: number
}

export type ScreenshotResult =
  { ok: true; png: Uint8Array } | { ok: false; reason: 'expired' | 'screen-permission' | 'unavailable' }

export interface ScreenshotApi {
  getSettings(): Promise<ScreenshotStatus>
  /** Patches the persisted settings (enabled/destination/bringToFront). */
  updateSettings(patch: ScreenshotSettingsPatch): Promise<ScreenshotStatus>
  openPermissionSettings(kind: 'input' | 'screen'): Promise<void>
  onStatus(callback: (status: ScreenshotStatus) => void): () => void
  onRequest(callback: (requestId: string, destination: ScreenshotDestination) => void): () => void
  capture(requestId: string): Promise<ScreenshotResult>
}
