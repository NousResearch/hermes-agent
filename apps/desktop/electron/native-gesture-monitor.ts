import { spawn as nodeSpawn, type SpawnOptions } from 'node:child_process'
import type { EventEmitter } from 'node:events'
import type { Readable, Writable } from 'node:stream'

export type NativeGestureStatus =
  { type: 'starting' | 'ready' | 'stopped' } | { type: 'error'; code: 'permission-required' | 'unavailable' }

interface MonitorChild extends EventEmitter {
  stdin: Writable | null
  stdout: Readable | null
  kill(signal?: NodeJS.Signals): boolean
}

export interface NativeGestureOptions {
  spawn?: (command: string, args: string[], options: SpawnOptions) => MonitorChild
  startupTimeoutMs?: number
  stopTimeoutMs?: number
}

interface MonitorOptions<T> extends NativeGestureOptions {
  path: string
  parseGesture: (value: Record<string, unknown>) => T | null
}

/** Bounded JSON-lines lifecycle shared by the passive screenshot and HUD helpers. */
export class NativeGestureMonitor<T> {
  private cleanup: (() => void) | undefined

  constructor(private readonly options: MonitorOptions<T>) {}

  start(
    onGesture: (gesture: T) => void,
    onStatus: (status: NativeGestureStatus) => void,
    requestPermission = false
  ): void {
    this.stop()
    let child: MonitorChild

    try {
      child = (this.options.spawn ?? nodeSpawn)(this.options.path, requestPermission ? ['--request-permission'] : [], {
        stdio: ['pipe', 'pipe', 'ignore'],
        shell: false,
        detached: false,
        windowsHide: true
      })
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

    const terminate = (status: NativeGestureStatus) => {
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

        let value: unknown

        try {
          value = JSON.parse(line)
        } catch {
          continue
        }

        if (!value || typeof value !== 'object') {
          continue
        }

        const message = value as Record<string, unknown>

        if (message.type === 'ready') {
          if (!ready) {
            ready = true
            clearTimeout(startupTimer)
            onStatus({ type: 'ready' })
          }
        } else if (
          message.type === 'error' &&
          (message.code === 'permission-required' || message.code === 'unavailable')
        ) {
          terminate({ type: 'error', code: message.code })
        } else if (ready) {
          const gesture = this.options.parseGesture(message)

          if (gesture !== null) {
            onGesture(gesture)
          }
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
