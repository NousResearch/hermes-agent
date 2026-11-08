import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { NativeGestureMonitor, type NativeGestureStatus } from './native-gesture-monitor'

export type HudModifierStatus = NativeGestureStatus

/** A session capability, not a claim that Xwayland can observe Wayland apps. */
export function hudModifierMonitorSupported(
  platform: NodeJS.Platform = process.platform,
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return (
    platform === 'darwin' ||
    platform === 'win32' ||
    (platform === 'linux' && !!env.DISPLAY && !env.WAYLAND_DISPLAY && env.XDG_SESSION_TYPE?.toLowerCase() !== 'wayland')
  )
}

export function resolveHudModifierMonitorPath(
  appPath = resolve(dirname(fileURLToPath(import.meta.url)), '..')
): string {
  const binary = process.platform === 'win32' ? 'hud-modifier-monitor.exe' : 'hud-modifier-monitor'
  const target = `${process.platform}-${process.platform === 'darwin' ? 'universal' : process.arch}`

  return resolve(appPath, 'dist/native', target, binary).replace(/\.asar(?=[/\\])/g, '.asar.unpacked')
}

/** Input remains inside the native process; only readiness/errors and summon cross the pipe. */
export class HudModifierMonitor {
  private readonly monitor: NativeGestureMonitor<true>

  constructor({ appPath }: { appPath?: string } = {}) {
    this.monitor = new NativeGestureMonitor({
      path: resolveHudModifierMonitorPath(appPath),
      parseGesture: value => (value.type === 'summon' ? true : null)
    })
  }

  /** Request macOS Input Monitoring permission only after an explicit opt-in/retry. */
  start(onSummon: () => void, onStatus: (status: HudModifierStatus) => void, requestPermission = false): void {
    this.stop()

    if (!hudModifierMonitorSupported()) {
      onStatus({ type: 'error', code: 'unavailable', reason: 'unsupported-session' })

      return
    }

    this.monitor.start(onSummon, onStatus, requestPermission)
  }

  stop(): void {
    this.monitor.stop()
  }
}
