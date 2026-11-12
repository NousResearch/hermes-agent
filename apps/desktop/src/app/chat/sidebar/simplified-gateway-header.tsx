import { useStore } from '@nanostores/react'
import { type PointerEvent, type ReactNode, useEffect, useId, useRef, useState } from 'react'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { RowButton } from '@/components/ui/row-button'
import { useI18n } from '@/i18n'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $gatewayState } from '@/store/session'

import { ActiveMachineStatusPanel, MachineStatusPanel } from '../../redesign/portal-gateway-menu-panel'

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

/** A machine row (Figma 147:7470): name with its organisation as a quiet inline tag, the status at
 *  the end behind its dot; it discloses the machine's agents. */
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
        <span className="portal-machine-name">{label}</span>
        {registered?.org && <span className="portal-machine-tag">{registered.org}</span>}
      </RowButton>
      <MachineStatusPopover
        label={t.shell.gatewayMenu.machineStatus(label, status)}
        panel={close =>
          selected ? (
            <ActiveMachineStatusPanel name={label} onClose={close} />
          ) : (
            <MachineStatusPanel connectionId={connectionId} label={label} onClose={close} />
          )
        }
      >
        <span aria-hidden className="portal-status-dot" />
        <span id={statusId}>{status}</span>
      </MachineStatusPopover>
    </div>
  )
}

/** How long a resting pointer takes to open the status popover, and how long it may travel between
 *  the status and the popover before it closes. */
const HOVER_OPEN_MS = 300
const HOVER_CLOSE_MS = 200

interface MachineStatusPopoverProps {
  children: ReactNode
  label: string
  panel: (close: () => void) => ReactNode
}

/** The machine's status as its own button (IA quick win 619:1788): a click, Enter or Space opens the
 *  machine's health popover with focus in it; a resting mouse opens it too, without taking focus,
 *  and it stays while the pointer is over the status or the popover. */
function MachineStatusPopover({ children, label, panel }: MachineStatusPopoverProps) {
  // 'hover' = opened by a resting pointer, which must not move focus or keep it open on leave.
  const [open, setOpen] = useState<'click' | 'hover' | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const schedule = (next: 'hover' | null, delay: number) => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setOpen(current => (current === 'click' ? current : next)), delay)
  }

  const cancel = () => window.clearTimeout(timer.current)

  useEffect(() => cancel, [])

  const close = () => {
    cancel()
    setOpen(null)
  }

  const hoverHandlers = {
    onPointerEnter: (event: PointerEvent<HTMLElement>) => {
      if (event.pointerType === 'mouse') {
        schedule('hover', open ? 0 : HOVER_OPEN_MS)
      }
    },
    onPointerLeave: (event: PointerEvent<HTMLElement>) => {
      if (event.pointerType === 'mouse') {
        schedule(null, HOVER_CLOSE_MS)
      }
    }
  }

  return (
    <Popover onOpenChange={next => (next ? setOpen('click') : close())} open={open !== null}>
      <PopoverTrigger asChild>
        <button
          aria-label={label}
          className="portal-machine-status"
          onClick={event => {
            // A click on a popover the pointer already opened keeps it, now as a deliberate open.
            if (open === 'hover') {
              event.preventDefault()
              cancel()
              setOpen('click')
            }
          }}
          type="button"
          {...hoverHandlers}
        >
          {children}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        aria-label={label}
        className="portal-machine-popover w-[300px] max-w-[calc(100vw-1rem)] max-h-[calc(100dvh-2rem)] overflow-y-auto p-0"
        onCloseAutoFocus={event => open === 'hover' && event.preventDefault()}
        onOpenAutoFocus={event => open === 'hover' && event.preventDefault()}
        side="bottom"
        {...hoverHandlers}
      >
        {panel(close)}
      </PopoverContent>
    </Popover>
  )
}
