import fs from 'node:fs'
import path from 'node:path'

export const LINUX_WINDOW_CONTROLS_MODES = ['auto', 'show', 'hide'] as const

export type LinuxWindowControlsMode = (typeof LINUX_WINDOW_CONTROLS_MODES)[number]

const DEFAULT_MODE: LinuxWindowControlsMode = 'auto'

// Tiling compositors own window management, so the in-window buttons only take
// up titlebar space there. Anything unlisted keeps its buttons: a missing entry
// costs a few pixels, a wrong entry strands a user without a close button.
const TILING_DESKTOPS = new Set([
  'awesome',
  'bspwm',
  'dwl',
  'dwm',
  'herbstluftwm',
  'hyprland',
  'i3',
  'leftwm',
  'niri',
  'qtile',
  'river',
  'spectrwm',
  'sway',
  'swayfx',
  'xmonad'
])

// Set by the compositor itself, so they survive launchers that drop
// XDG_CURRENT_DESKTOP from the app's environment.
const TILING_SESSION_VARS = ['HYPRLAND_INSTANCE_SIGNATURE', 'I3SOCK', 'NIRI_SOCKET', 'SWAYSOCK'] as const

export function isTilingSession(env: NodeJS.ProcessEnv): boolean {
  if (TILING_SESSION_VARS.some(name => env[name])) {
    return true
  }

  // XDG_CURRENT_DESKTOP is a colon-separated list, e.g. `niri:GNOME`.
  return [env.XDG_CURRENT_DESKTOP, env.XDG_SESSION_DESKTOP]
    .flatMap(value => (value ?? '').split(':'))
    .some(name => TILING_DESKTOPS.has(name.trim().toLowerCase()))
}

interface LinuxWindowControlsHiddenInput {
  env: NodeJS.ProcessEnv
  isWsl: boolean
  mode: LinuxWindowControlsMode
  platform: NodeJS.Platform
}

/** WSLg is excluded: its host draws no frame, so these are its only controls. */
export function linuxWindowControlsHidden({ env, isWsl, mode, platform }: LinuxWindowControlsHiddenInput): boolean {
  if (platform !== 'linux' || isWsl) {
    return false
  }

  return mode === 'hide' || (mode === 'auto' && isTilingSession(env))
}

function isMode(value: unknown): value is LinuxWindowControlsMode {
  return LINUX_WINDOW_CONTROLS_MODES.includes(value as LinuxWindowControlsMode)
}

export function readLinuxWindowControlsMode(configPath: string): LinuxWindowControlsMode {
  try {
    const { mode } = JSON.parse(fs.readFileSync(configPath, 'utf8'))

    return isMode(mode) ? mode : DEFAULT_MODE
  } catch {
    // Missing or malformed: behave like a fresh install.
    return DEFAULT_MODE
  }
}

function writeLinuxWindowControlsMode(configPath: string, mode: LinuxWindowControlsMode): void {
  fs.mkdirSync(path.dirname(configPath), { recursive: true })
  fs.writeFileSync(configPath, JSON.stringify({ mode }, null, 2), 'utf8')
}

export interface LinuxWindowControlsStatus {
  mode: LinuxWindowControlsMode
  restartNeeded: boolean
}

interface LinuxWindowControlsIpc {
  handle(channel: string, listener: (event: unknown, mode: unknown) => unknown): unknown
}

// titleBarOverlay is fixed when a window is created, so the setting is stored
// here and only read at launch. The file stays the single source of truth; the
// renderer asks for it instead of keeping a second copy in localStorage.
export function registerLinuxWindowControlsIpc(
  ipcMain: LinuxWindowControlsIpc,
  configPath: string,
  status: (mode: LinuxWindowControlsMode) => LinuxWindowControlsStatus
): void {
  ipcMain.handle('hermes:window-controls:mode:get', () => status(readLinuxWindowControlsMode(configPath)))

  ipcMain.handle('hermes:window-controls:mode:set', (_event, mode) => {
    if (!isMode(mode)) {
      throw new Error(`Unknown window controls mode: ${String(mode)}`)
    }

    writeLinuxWindowControlsMode(configPath, mode)

    return status(mode)
  })
}

export function setupLinuxWindowControls(ipcMain: LinuxWindowControlsIpc, userData: string, isWsl: boolean): boolean {
  const configPath = path.join(userData, 'window-controls.json')

  const hiddenWith = (mode: LinuxWindowControlsMode) =>
    linuxWindowControlsHidden({ env: process.env, isWsl, mode, platform: process.platform })

  const hiddenAtLaunch = hiddenWith(readLinuxWindowControlsMode(configPath))

  // Compares outcomes, not names: Auto to Show on KDE changes nothing on screen.
  registerLinuxWindowControlsIpc(ipcMain, configPath, mode => ({
    mode,
    restartNeeded: hiddenWith(mode) !== hiddenAtLaunch
  }))

  return hiddenAtLaunch
}
