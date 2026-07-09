import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { IxStatusSummary } from '@/global'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

// Slim native strip above the IX Agency surfaces: VPN lamp + connect toggle,
// admin-mcp gateway lamp, and the NON-BLOCKING update button. The same lamps
// live in the system tray (fed by the same main-process pollers).

const POLL_MS = 30_000

function Lamp({ color, label, title }: { color: string; label: string; title: string }) {
  return (
    <Tip label={title} side="bottom">
      <span className="flex min-w-0 items-center gap-1.5 text-[0.7rem] text-muted-foreground">
        <span aria-hidden="true" className={cn('size-2 shrink-0 rounded-full', color)} />
        <span className="truncate">{label}</span>
      </span>
    </Tip>
  )
}

const VPN_LAMP_COLOR: Record<string, string> = {
  connected: 'bg-emerald-500',
  connecting: 'bg-amber-500 animate-pulse',
  degraded: 'bg-amber-500',
  disconnected: 'bg-neutral-400',
  error: 'bg-red-500',
  unavailable: 'bg-neutral-400'
}

const MCP_LAMP_COLOR: Record<string, string> = {
  green: 'bg-emerald-500',
  grey: 'bg-neutral-400',
  red: 'bg-red-500'
}

export function IxStatusStrip() {
  const bridge = window.hermesDesktop?.ixAgency
  const [summary, setSummary] = useState<IxStatusSummary | null>(null)
  const [vpnBusy, setVpnBusy] = useState(false)

  const refresh = useCallback(
    async (force = false) => {
      if (!bridge?.statusSummary) {
        return
      }

      try {
        setSummary(await bridge.statusSummary(force))
      } catch {
        // Best-effort; the last known lamps stay on screen.
      }
    },
    [bridge]
  )

  useEffect(() => {
    void refresh()

    const timer = setInterval(() => void refresh(), POLL_MS)
    const onFocus = () => void refresh()

    window.addEventListener('focus', onFocus)

    return () => {
      clearInterval(timer)
      window.removeEventListener('focus', onFocus)
    }
  }, [refresh])

  if (!bridge) {
    return null
  }

  const vpn = summary?.vpn
  const mcp = summary?.mcp
  const update = summary?.update

  const toggleVpn = async () => {
    setVpnBusy(true)

    try {
      if (vpn?.tunnelUp) {
        await bridge.vpnDisconnect()
      } else {
        await bridge.vpnConnect()
      }
    } catch (error) {
      notifyError(error, 'VPN action failed')
    } finally {
      setVpnBusy(false)
      void refresh(true)
    }
  }

  return (
    <div className="flex h-9 w-full shrink-0 items-center gap-3 rounded-md border border-(--ui-border-primary) bg-(--ui-bg-quinary) px-3">
      <Lamp
        color={VPN_LAMP_COLOR[vpn?.state ?? 'unavailable']}
        label={`VPN ${vpn?.state ?? '…'}`}
        title={vpn?.detail ?? 'Checking VPN…'}
      />
      <Button disabled={vpnBusy} onClick={() => void toggleVpn()} size="xs" variant="ghost">
        {vpnBusy ? <Codicon name="loading~spin" size="0.75rem" /> : null}
        {vpn?.tunnelUp ? 'Disconnect' : 'Connect'}
      </Button>
      <span className="h-4 w-px bg-(--ui-border-primary)" />
      <Lamp
        color={MCP_LAMP_COLOR[mcp?.state ?? 'grey']}
        label={`MCP${mcp?.toolCount ? ` (${mcp.toolCount})` : ''}`}
        title={mcp?.detail ?? 'Checking admin-mcp gateway…'}
      />
      <div className="min-w-0 flex-1" />
      {update?.updateAvailable && (
        <Tip
          label={`${update.currentVersion} → ${update.latestVersion}${update.notes ? ` — ${update.notes}` : ''}`}
          side="bottom"
        >
          <Button
            onClick={() => void bridge.updateApply().catch((error: unknown) => notifyError(error, 'Update failed'))}
            size="xs"
            variant="secondary"
          >
            <Codicon name="arrow-circle-up" size="0.75rem" />
            Update available — Restart to update
          </Button>
        </Tip>
      )}
    </div>
  )
}
