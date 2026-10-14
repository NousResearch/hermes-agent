import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $desktopBoot } from '@/store/boot'
import { initializeConnectionsRegistry, refreshConnectionsRegistry } from '@/store/connections'
import { isAuxiliaryWindow, isPeerInstanceWindow } from '@/store/windows'

const RETRY_DELAYS_MS = [1_000, 2_000]

/** Gateway discovery belongs to the window, not its optional navigation chrome. */
export function useConnectionsRegistry(): void {
  const boot = useStore($desktopBoot)

  useEffect(() => {
    let disposed = false
    let failed = false
    let pending = false
    let requested = false
    let retries = 0
    let timer: ReturnType<typeof setTimeout> | undefined

    const refresh = async () => {
      if (pending) {
        requested = true

        return
      }

      clearTimeout(timer)
      pending = true

      try {
        await refreshConnectionsRegistry()
        failed = false
        retries = 0
      } catch (error) {
        failed = true

        if (!disposed) {
          if (retries < RETRY_DELAYS_MS.length) {
            timer = setTimeout(() => void refresh(), RETRY_DELAYS_MS[retries++])
          } else {
            console.warn('[connections] Registry read failed; retrying on focus or registry change', error)
          }
        }
      } finally {
        pending = false

        // A save may have landed while the previous snapshot was in flight.
        if (!disposed && requested) {
          requested = false
          void refresh()
        }
      }
    }

    const onFocus = () => {
      if (failed && !pending) {
        retries = 0
        void refresh()
      }
    }

    const off = window.hermesDesktop?.connections?.onChanged?.(() => void refresh())
    window.addEventListener('focus', onFocus)
    void refresh()

    return () => {
      disposed = true
      clearTimeout(timer)
      window.removeEventListener('focus', onFocus)
      off?.()
    }
  }, [])

  useEffect(() => {
    // Finish the primary config/session reads before restoring another source,
    // or their late responses could repaint the new workspace. Peer and
    // auxiliary windows already have an explicit destination of their own.
    if (!boot.running && !isAuxiliaryWindow() && !isPeerInstanceWindow()) {
      void initializeConnectionsRegistry().catch(() => undefined)
    }
  }, [boot.running])
}
