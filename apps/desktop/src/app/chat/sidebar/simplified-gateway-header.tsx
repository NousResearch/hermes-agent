import { useStore } from '@nanostores/react'
import { useId } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { useI18n } from '@/i18n'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $gatewayState } from '@/store/session'

type MachineState = 'open' | 'connecting' | 'idle' | 'closed' | 'error'

/** Figma's three machine tones: green online, orange in transition, grey offline. */
const TONE: Record<MachineState, 'online' | 'pending' | 'offline'> = {
  open: 'online',
  connecting: 'pending',
  idle: 'pending',
  closed: 'offline',
  error: 'offline'
}

interface SimplifiedGatewayHeaderProps {
  connectionId: string | null
  /** The machine's display label, already resolved by `useSimplifiedAgents`. */
  label: string
  selected: boolean
  open: boolean
  onToggle: () => void
}

/** A flat machine row (status dot, name, status) that discloses the machine's agents. */
export function SimplifiedGatewayHeader({
  connectionId,
  label,
  selected,
  open,
  onToggle
}: SimplifiedGatewayHeaderProps) {
  const { t } = useI18n()
  const statusId = useId()
  const registry = useStore($connectionsRegistry)
  const roster = useStore($fleetRoster)
  const gatewayState = useStore($gatewayState)
  const source = roster?.sources.find(source => source.connectionId === connectionId)
  const registered = registry?.connections.find(connection => connection.id === connectionId)

  const address =
    registered?.url && URL.canParse(registered.url)
      ? new URL(registered.url).host
      : registered?.host || (selected && isBrowserHostedDesktop() ? window.location.host : '')

  const toggleLabel = `${open ? t.common.collapse : t.common.expand}: ${label}`
  const state: MachineState = selected ? gatewayState : source ? (source.reachable ? 'open' : 'closed') : 'idle'

  const status = {
    open: t.shell.gatewayMenu.connected,
    connecting: t.shell.gatewayMenu.connecting,
    idle: t.common.loading,
    closed: t.shell.gatewayMenu.offline,
    error: t.shell.gatewayMenu.offline
  }[state]

  return (
    <div
      aria-current={selected ? 'true' : undefined}
      aria-label={label}
      data-slot="gateway-heading"
      data-tone={TONE[state]}
      role="group"
    >
      <RowButton
        aria-describedby={statusId}
        aria-expanded={open}
        aria-label={toggleLabel}
        className="portal-machine-row"
        onClick={onToggle}
        title={address ? `${label} · ${address}` : label}
      >
        <span aria-hidden className="portal-status-dot" />
        <span className="portal-machine-name">{label}</span>
        <span className="portal-machine-status" id={statusId}>
          {state === 'open' ? registered?.org || status : status}
        </span>
      </RowButton>
    </div>
  )
}
