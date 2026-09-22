import type { DesktopBootProgress } from '@/global'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS, TimeoutError } from '@/lib/with-timeout'

import { UPDATE_WAIT_TIMEOUT_MS } from '../../electron/update-gate'

/** A live updater is not a hung backend. Only its explicit progress renews
 * the spawn watchdog; ordinary progress cannot keep a broken boot alive.
 * The update gate's absolute ceiling still applies, even with endless ticks. */
export function withBackendBootTimeout<T>(
  promise: Promise<T>,
  message: string,
  desktop: Pick<Window['hermesDesktop'], 'getBootProgress' | 'onBootProgress'> | undefined = window.hermesDesktop
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false
    let updateWaiting = false
    let updateDeadline: number | null = null
    let snapshotSuperseded = false
    let timer: ReturnType<typeof setTimeout>
    let unlisten: (() => void) | undefined

    const cleanup = () => {
      settled = true
      clearTimeout(timer)
      unlisten?.()
    }

    const arm = () => {
      clearTimeout(timer)
      const remaining = updateDeadline === null ? Infinity : updateDeadline - Date.now()
      timer = setTimeout(
        () => {
          cleanup()
          reject(new TimeoutError(message))
        },
        Math.max(0, Math.min(BACKEND_BOOT_WAIT_TIMEOUT_MS, remaining))
      )
    }

    const observe = (progress: DesktopBootProgress) => {
      if (settled) {
        return
      }

      const waiting = progress.running && !progress.error && progress.phase === 'backend.update-wait'

      if (waiting) {
        // Allow main's bounded gate to finish, then one ordinary cold spawn.
        updateDeadline ??= Date.now() + UPDATE_WAIT_TIMEOUT_MS + BACKEND_BOOT_WAIT_TIMEOUT_MS
      }

      if (waiting || updateWaiting) {
        arm()
      }

      updateWaiting = waiting
    }

    arm()
    unlisten = desktop?.onBootProgress?.(progress => {
      snapshotSuperseded = true
      observe(progress)
    })
    void desktop
      ?.getBootProgress?.()
      .then(progress => {
        if (!snapshotSuperseded) {
          observe(progress)
        }
      })
      .catch(() => undefined)

    Promise.resolve(promise).then(
      value => {
        cleanup()
        resolve(value)
      },
      error => {
        cleanup()
        reject(error)
      }
    )
  })
}
