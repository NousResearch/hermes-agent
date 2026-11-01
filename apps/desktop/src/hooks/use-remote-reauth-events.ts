import { useEffect } from 'react'

import { clearRemoteReauthExhausted, latchRemoteReauthExhausted } from '@/store/remote-reauth'

/**
 * Mirrors the Electron main process's silent-sign-in exhaustion ledger into
 * renderer state. Both pushes are optional: an older Electron main without
 * them simply never latches the banner — the bounded-retry guard itself still
 * lives main-side, this only carries its verdict to the surface.
 */
export function useRemoteReauthEvents(): void {
  useEffect(() => {
    const desktop = window.hermesDesktop
    const offExhausted = desktop?.onOauthLoginExhausted?.(payload => latchRemoteReauthExhausted(payload.baseUrl))
    const offRestored = desktop?.onOauthLoginRestored?.(payload => clearRemoteReauthExhausted(payload.baseUrl))

    return () => {
      offExhausted?.()
      offRestored?.()
    }
  }, [])
}
