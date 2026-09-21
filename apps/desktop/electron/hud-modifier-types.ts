export interface HudModifierStatus {
  enabled: boolean
  state: 'disabled' | 'starting' | 'ready' | 'input-permission' | 'unavailable'
}

export interface HudModifierApi {
  getSettings: () => Promise<HudModifierStatus>
  setEnabled: (enabled: boolean) => Promise<HudModifierStatus>
  openPermissionSettings: () => Promise<void>
  onStatus: (callback: (status: HudModifierStatus) => void) => () => void
}
