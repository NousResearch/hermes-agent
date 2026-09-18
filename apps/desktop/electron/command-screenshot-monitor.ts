import { spawn as nodeSpawn, type SpawnOptions } from 'node:child_process'
import type { EventEmitter } from 'node:events'
import { dirname, resolve } from 'node:path'
import type { Readable, Writable } from 'node:stream'
import { fileURLToPath } from 'node:url'

import type { ScreenshotWindow } from './command-screenshot-types'

export interface CommandScreenshotCapture extends ScreenshotWindow {
  type: 'capture'
}

export type CommandScreenshotStatus =
  | { type: 'starting' | 'ready' | 'stopped' }
  | { type: 'error'; code: 'permission-required' | 'unavailable' }

interface MonitorChild extends EventEmitter {
  stdin: Writable | null
  stdout: Readable | null
  kill(signal?: NodeJS.Signals): boolean
}

interface MonitorOptions {
  appPath?: string
  platform?: NodeJS.Platform
  spawn?: (command: string, args: string[], options: SpawnOptions) => MonitorChild
  startupTimeoutMs?: number
  stopTimeoutMs?: number
}

export function resolveCommandScreenshotMonitorPath(
  appPath = resolve(dirname(fileURLToPath(import.meta.url)), '..'),
): string {
  // Executables cannot run inside ASAR. dist/** is already explicitly unpacked.
  return resolve(appPath, 'dist/native/command-screenshot-monitor')
    .replace(/\.asar(?=[/\\])/g, '.asar.unpacked')
}

function parseMessage(line: string): CommandScreenshotCapture | CommandScreenshotStatus | null {
  let value: unknown
  try {
    value = JSON.parse(line)
  } catch {
    return null
  }
  if (!value || typeof value !== 'object') {
    return null
  }
  const message = value as Record<string, unknown>
  if (message.type === 'ready') {
    return { type: 'ready' }
  }
  if (message.type === 'error' && (message.code === 'permission-required' || message.code === 'unavailable')) {
    return { type: 'error', code: message.code }
  }
  if (message.type !== 'capture') {
    return null
  }
  const { windowId, width, height } = message
  if (typeof windowId !== 'number' || !Number.isInteger(windowId) || windowId <= 0 || windowId > 0xffffffff
    || typeof width !== 'number' || !Number.isFinite(width) || width <= 0
    || typeof height !== 'number' || !Number.isFinite(height) || height <= 0) {
    return null
  }
  return { type: 'capture', windowId, width, height }
}

/** Own one passive native monitor. Call stop() before app quit or disabling the gesture. */
export class CommandScreenshotMonitor {
  private cleanup: (() => void) | undefined

  constructor(private readonly options: MonitorOptions = {}) {}

  /** Only explicit user intent may pass requestPermission=true (opens the macOS prompt). */
  start(
    onCapture: (capture: CommandScreenshotCapture) => void,
    onStatus: (status: CommandScreenshotStatus) => void,
    requestPermission = false,
  ): void {
    this.stop()
    if ((this.options.platform ?? process.platform) !== 'darwin') {
      onStatus({ type: 'error', code: 'unavailable' })
      return
    }
    let child: MonitorChild
    try {
      child = (this.options.spawn ?? nodeSpawn)(resolveCommandScreenshotMonitorPath(this.options.appPath),
        requestPermission ? ['--request-permission'] : [],
        { stdio: ['pipe', 'pipe', 'ignore'], shell: false, detached: false, windowsHide: true })
    } catch {
      onStatus({ type: 'error', code: 'unavailable' })
      return
    }
    let ready = false
    let active = true
    let pending = ''
    let killTimer: ReturnType<typeof setTimeout> | undefined
    const startupTimer = setTimeout(() => fail(), this.options.startupTimeoutMs ?? (requestPermission ? 60_000 : 5_000))
    startupTimer.unref()
    const dispose = () => {
      active = false
      pending = ''
      clearTimeout(startupTimer)
      child.stdout?.removeListener('data', onData)
      if (this.cleanup === stop) {
        this.cleanup = undefined
      }
    }
    const terminate = (status: CommandScreenshotStatus) => {
      if (!active) {
        return
      }
      dispose()
      child.stdin?.end() // EOF also stops the helper if the parent exits unexpectedly.
      child.kill('SIGTERM')
      killTimer = setTimeout(() => child.kill('SIGKILL'), this.options.stopTimeoutMs ?? 1_000)
      killTimer.unref()
      onStatus(status)
    }
    const fail = () => terminate({ type: 'error', code: 'unavailable' })
    const onData = (chunk: Buffer | string) => {
      if (!active) {
        return
      }
      if (chunk.length > 65_536) {
        fail()
        return
      }
      pending += chunk.toString()
      const lines = pending.split('\n')
      pending = lines.pop() ?? ''
      if (pending.length > 4_096) {
        fail()
        return
      }
      for (const line of lines) {
        if (!active) {
          break
        }
        if (line.length > 4_096) {
          fail()
          break
        }
        const message = parseMessage(line)
        if (!message) {
          continue
        }
        if (message.type === 'capture') {
          if (ready) {
            onCapture(message)
          }
        } else if (message.type === 'ready' && !ready) {
          ready = true
          clearTimeout(startupTimer)
          onStatus(message)
        } else if (message.type === 'error') {
          terminate(message)
        }
      }
    }
    const onClose = () => {
      const unexpected = active
      dispose()
      clearTimeout(killTimer)
      child.removeListener('close', onClose)
      child.removeListener('error', fail)
      child.stdin?.removeListener('error', fail)
      child.stdout?.removeListener('error', fail)
      if (unexpected) {
        onStatus({ type: 'error', code: 'unavailable' })
      }
    }
    const stop = () => terminate({ type: 'stopped' })
    this.cleanup = stop
    child.stdout?.on('data', onData)
    child.stdout?.on('error', fail)
    child.stdin?.on('error', fail)
    child.once('close', onClose)
    child.on('error', fail)
    onStatus({ type: 'starting' })
    if (!child.stdout || !child.stdin) {
      fail()
    }
  }

  stop(): void {
    this.cleanup?.()
  }
}
