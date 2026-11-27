import type { DesktopBootstrapEvent, DesktopBootstrapState } from '@/global'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS, TimeoutError } from '@/lib/with-timeout'

export const INITIAL_BOOT_MAX_WAIT_MS = 30 * 60_000

interface BootstrapBridge {
  getBootstrapState?: () => Promise<DesktopBootstrapState>
  onBootstrapEvent?: (callback: (event: DesktopBootstrapEvent) => void) => () => void
}

export function waitForBootConnection<T>(work: Promise<T>, desktop: BootstrapBridge, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let active = false
    let choosing = false
    let revision = 0
    let settled = false
    let idleTimer: ReturnType<typeof setTimeout> | undefined
    let off: (() => void) | undefined
    const stages = new Map<string, string>()
    const message = 'Timed out connecting to Hermes backend'
    const fail = () => finish(() => reject(new TimeoutError(message)))
    const absoluteTimer = setTimeout(fail, INITIAL_BOOT_MAX_WAIT_MS)

    function finish(action: () => void) {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(absoluteTimer)
      clearTimeout(idleTimer)
      off?.()
      signal.removeEventListener('abort', abort)
      action()
    }

    function abort() {
      finish(() => reject(new Error('Boot connection wait cancelled')))
    }

    function resetIdle() {
      clearTimeout(idleTimer)

      if (!choosing) {
        idleTimer = setTimeout(fail, BACKEND_BOOT_WAIT_TIMEOUT_MS)
      }
    }

    resetIdle()

    if (signal.aborted) {
      abort()

      return
    }

    signal.addEventListener('abort', abort, { once: true })
    off = desktop.onBootstrapEvent?.(event => {
      if (settled) {
        return
      }

      if (event.type !== 'log') {
        revision++
      }

      if (event.type === 'setup-choice') {
        active = false

        if (choosing !== event.active) {
          choosing = event.active
          resetIdle()
        }
      } else if (event.type === 'manifest') {
        if (!active) {
          active = true
          choosing = false
          stages.clear()
          resetIdle()
        }
      } else if (event.type === 'stage' && !choosing) {
        if (!active && event.state === 'running') {
          active = true
        }

        if (active && stages.get(event.name) !== event.state) {
          stages.set(event.name, event.state)
          resetIdle()
        }
      } else if (event.type === 'log' && active && event.line.trim()) {
        resetIdle()
      } else if (
        event.type === 'complete' ||
        event.type === 'failed' ||
        event.type === 'dismissed' ||
        event.type === 'unsupported-platform'
      ) {
        if (active || choosing) {
          active = false
          choosing = false
          resetIdle()
        }
      }
    })
    const snapshotRevision = revision
    desktop
      .getBootstrapState?.()
      .then(snapshot => {
        if (settled || revision !== snapshotRevision || !snapshot) {
          return
        }

        active = snapshot.active
        choosing = Boolean(snapshot.setupChoice)

        for (const [name, stage] of Object.entries(snapshot.stages ?? {})) {
          stages.set(name, stage.state)
        }

        resetIdle()
      })
      .catch(() => undefined)
    Promise.resolve(work).then(
      value => finish(() => resolve(value)),
      error => finish(() => reject(error))
    )
  })
}
