import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import type { IxAgencyVpnStatus } from '@/global'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { titlebarButtonClass } from './titlebar'

// Company-VPN pill pinned in the titlebar's top-right cluster — visible on
// every view, not just the IX Agency tab. Click toggles the WireGuard tunnel
// (hermes:ix-agency:vpn:* IPC; elevation prompts are the honest macOS
// admin-password / Windows UAC dialogs).
//
// Auto-connect: the pill also watches the portal OTP session (main-process
// authStatus probe). The moment a login lands — fresh OTP or a restored
// cookie session at startup — it brings the tunnel up automatically, once per
// login. A dismissed elevation prompt is NOT retried until the user signs out
// and back in (or clicks the pill), so nobody gets nagged in a loop.

const VPN_POLL_MS = 5_000
const AUTH_POLL_MS = 10_000

const DOT: Record<IxAgencyVpnStatus['state'], string> = {
  connected: 'bg-emerald-500',
  connecting: 'bg-amber-400 animate-pulse',
  disconnected: 'bg-neutral-400',
  unavailable: 'bg-neutral-400',
  unknown: 'bg-amber-400'
}

// Module-scoped so remounts (overlay open/close) don't re-prompt elevation.
let autoConnectAttempted = false

function ixApi() {
  return window.hermesDesktop?.ixAgency ?? null
}

export function IxVpnTitlebarPill() {
  const [status, setStatus] = useState<IxAgencyVpnStatus>({ state: 'unknown', detail: 'Checking VPN status…' })
  const [busy, setBusy] = useState(false)
  const statusRef = useRef(status)
  const busyRef = useRef(busy)

  statusRef.current = status
  busyRef.current = busy

  const refresh = useCallback(async () => {
    const api = ixApi()

    if (!api) {
      return
    }

    try {
      setStatus(await api.vpnStatus())
    } catch {
      // Best-effort poll; the last known state stays on screen.
    }
  }, [])

  const connect = useCallback(
    async (auto: boolean) => {
      const api = ixApi()

      if (!api || busyRef.current) {
        return
      }

      setBusy(true)

      try {
        const next = await api.vpnConnect()
        setStatus(next)
        notify({ message: auto ? 'VPN connected (auto, signed-in session)' : 'VPN connected', detail: next.detail })
      } catch (error) {
        notifyError(error, auto ? 'VPN auto-connect failed' : 'VPN connect failed')
        await refresh()
      } finally {
        setBusy(false)
      }
    },
    [refresh]
  )

  const toggle = useCallback(async () => {
    const api = ixApi()

    if (!api || busyRef.current) {
      return
    }

    if (statusRef.current.state !== 'connected') {
      // A manual connect counts as this login's attempt.
      autoConnectAttempted = true
      await connect(false)

      return
    }

    setBusy(true)

    try {
      const next = await api.vpnDisconnect()
      setStatus(next)
      notify({ message: 'VPN disconnected', detail: next.detail })
    } catch (error) {
      notifyError(error, 'VPN disconnect failed')
      await refresh()
    } finally {
      setBusy(false)
    }
  }, [connect, refresh])

  // Live tunnel state for the pill.
  useEffect(() => {
    void refresh()

    const timer = window.setInterval(() => void refresh(), VPN_POLL_MS)

    return () => window.clearInterval(timer)
  }, [refresh])

  // Auto-connect on successful portal login. Two triggers share one tick:
  //  - the periodic auth poll (catches restored cookie sessions at startup),
  //  - the portal webview's navigation event (fires the moment an OTP login
  //    lands on a signed-in route; force-refreshes past the auth cache).
  useEffect(() => {
    const api = ixApi()

    if (!api?.authStatus) {
      return
    }

    const tick = async (force = false) => {
      try {
        const auth = await api.authStatus(force)

        if (!auth.authenticated) {
          // Signed out — arm the next login for a fresh auto-connect.
          autoConnectAttempted = false

          return
        }

        if (autoConnectAttempted || busyRef.current) {
          return
        }

        // Only when a profile is configured and the tunnel is genuinely down:
        // 'unavailable' means no conf imported (nothing to connect),
        // 'connected'/'connecting' need no action, 'unknown' is not trusted.
        if (statusRef.current.state === 'unknown') {
          try {
            statusRef.current = await api.vpnStatus()
            setStatus(statusRef.current)
          } catch {
            return
          }
        }

        if (statusRef.current.state !== 'disconnected') {
          return
        }

        autoConnectAttempted = true
        await connect(true)
      } catch {
        // Auth probe unreachable — try again next tick.
      }
    }

    const onPortalNavigated = () => void tick(true)

    window.addEventListener('ix-agency:portal-navigated', onPortalNavigated)
    void tick()

    const timer = window.setInterval(() => void tick(), AUTH_POLL_MS)

    return () => {
      window.removeEventListener('ix-agency:portal-navigated', onPortalNavigated)
      window.clearInterval(timer)
    }
  }, [connect])

  if (!ixApi()) {
    return null
  }

  const connected = status.state === 'connected'
  const inFlight = busy || status.state === 'connecting'
  const label = inFlight ? 'VPN…' : connected ? 'VPN on' : 'VPN off'
  const action = connected ? 'Click to disconnect the company VPN' : 'Click to connect the company VPN'

  return (
    <Tip label={`${status.detail} — ${action}`}>
      <Button
        aria-label={`VPN: ${status.detail}`}
        className={cn(titlebarButtonClass, 'h-(--titlebar-control-height) gap-1.5 bg-transparent px-2 select-none')}
        disabled={inFlight || status.state === 'unavailable'}
        onClick={() => void toggle()}
        onPointerDown={event => event.stopPropagation()}
        size="xs"
        type="button"
        variant="ghost"
      >
        <span aria-hidden className={cn('size-2 shrink-0 rounded-full', DOT[status.state])} />
        <span className="text-[0.6875rem] tabular-nums">{label}</span>
      </Button>
    </Tip>
  )
}
