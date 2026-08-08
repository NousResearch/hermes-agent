/**
 * Windows tray preferences — main-process owned, mirrored for Settings.
 *
 * Values live in `userData/tray-preferences.json` (see electron/tray-preferences.ts).
 * Launch-at-login is also applied via Electron `setLoginItemSettings`.
 */
import { atom } from 'nanostores'

export type TrayPreferences = {
  enabled: boolean
  closeToTray: boolean
  startInTray: boolean
  popOutPetOnStartup: boolean
  launchAtLogin: boolean
}

export type TrayPreferencesSnapshot = {
  preferences: TrayPreferences
  trayAvailable: boolean
  platform: string
  launchAtLoginSupported: boolean
}

const DEFAULTS: TrayPreferences = {
  enabled: true,
  closeToTray: true,
  startInTray: false,
  popOutPetOnStartup: false,
  launchAtLogin: false
}

export const $trayPreferences = atom<TrayPreferences>({ ...DEFAULTS })
export const $trayAvailable = atom(false)
export const $trayLaunchAtLoginSupported = atom(false)
export const $trayPlatform = atom('')
export const $trayPrefsStatus = atom<'idle' | 'loading' | 'ready' | 'error'>('idle')

export async function loadTrayPreferences(): Promise<boolean> {
  const bridge = window.hermesDesktop

  if (!bridge?.getTrayPreferences) {
    $trayPrefsStatus.set('error')

    return false
  }

  $trayPrefsStatus.set('loading')

  try {
    const snap = await bridge.getTrayPreferences()
    applySnapshot(snap)
    $trayPrefsStatus.set('ready')

    return true
  } catch {
    $trayPrefsStatus.set('error')

    return false
  }
}

export async function setTrayPreference<K extends keyof TrayPreferences>(
  key: K,
  value: TrayPreferences[K]
): Promise<boolean> {
  const bridge = window.hermesDesktop

  if (!bridge?.setTrayPreferences) {
    return false
  }

  const previous = $trayPreferences.get()
  $trayPreferences.set({ ...previous, [key]: value })

  try {
    const snap = await bridge.setTrayPreferences({ [key]: value })
    applySnapshot(snap)

    return true
  } catch {
    $trayPreferences.set(previous)

    return false
  }
}

function applySnapshot(snap: TrayPreferencesSnapshot) {
  if (snap?.preferences) {
    $trayPreferences.set({ ...DEFAULTS, ...snap.preferences })
  }

  $trayAvailable.set(Boolean(snap?.trayAvailable))
  $trayLaunchAtLoginSupported.set(Boolean(snap?.launchAtLoginSupported))
  $trayPlatform.set(String(snap?.platform || ''))
}

if (typeof window !== 'undefined' && window.hermesDesktop?.getTrayPreferences) {
  void loadTrayPreferences()
}
