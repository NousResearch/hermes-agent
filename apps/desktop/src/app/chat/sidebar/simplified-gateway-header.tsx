import { useStore } from '@nanostores/react'
import { type ReactNode, useId } from 'react'

import type { AgentSettingsOwner } from '@/app/agent-settings/data'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Plus } from '@/lib/icons'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { $connectionsRegistry } from '@/store/connection-registry-state'

import { ActiveMachineStatusPanel, MachineStatusPanel } from '../../redesign/portal-gateway-menu-panel'

import { type MachineState, type NewAgentOffer, useMachineStatusText } from './portal/new-agent-offers'
import { useRestingHover } from './use-resting-hover'

/** Figma's three machine tones: green online, orange in transition, grey offline. */
const TONE: Record<MachineState, 'online' | 'pending' | 'offline'> = {
  open: 'online',
  connecting: 'pending',
  idle: 'pending',
  closed: 'offline',
  error: 'offline'
}

interface SimplifiedGatewayHeaderProps {
  /** The machine, and where New agent stands for it. */
  machine: NewAgentOffer
  open: boolean
  /** Absent while a search holds the machine open: the row then stands as a disabled disclosure. */
  onToggle?: () => void
  onNewAgent: (owner: AgentSettingsOwner) => void
}

/** A machine row (Figma 147:7470): name with its organisation as a quiet inline tag, the status at
 *  the end behind its dot; it discloses the machine's agents. */
export function SimplifiedGatewayHeader({ machine, open, onToggle, onNewAgent }: SimplifiedGatewayHeaderProps) {
  const { connectionId, label, selected, state } = machine
  const { t } = useI18n()
  const statusId = useId()
  const registry = useStore($connectionsRegistry)
  const registered = registry?.connections.find(connection => connection.id === connectionId)

  const address =
    registered?.url && URL.canParse(registered.url)
      ? new URL(registered.url).host
      : registered?.host || (selected && isBrowserHostedDesktop() ? window.location.host : '')

  const toggleLabel = `${open ? t.common.collapse : t.common.expand}: ${label}`
  const status = useMachineStatusText()[state]

  return (
    <div
      aria-current={selected ? 'true' : undefined}
      aria-label={label}
      data-slot="gateway-heading"
      data-tone={TONE[state]}
      role="group"
    >
      {/* Only what the row does not already say: a tip repeating the name it sits on is noise. */}
      <Tip label={address ? `${label} · ${address}` : undefined}>
        <RowButton
          aria-describedby={statusId}
          // aria-disabled, not disabled: the row stays on the arrow-key walk while it cannot fold.
          aria-disabled={onToggle ? undefined : true}
          aria-expanded={open}
          aria-label={toggleLabel}
          className="portal-machine-row"
          onClick={onToggle}
        >
          <span className="portal-machine-name">{label}</span>
          {registered?.org && <span className="portal-machine-tag">{registered.org}</span>}
        </RowButton>
      </Tip>
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
      {/* A machine that cannot take an agent keeps its button, which says what it waits for. */}
      <Tip label={machine.tip}>
        <Button
          aria-disabled={machine.owner ? undefined : true}
          // The tip says what the row does not; the name also says which machine, as its neighbours do.
          aria-label={`${machine.tip}: ${label}`}
          className="portal-section-action"
          onClick={() => machine.owner && onNewAgent(machine.owner)}
          size="icon-xs"
          variant="ghost"
        >
          <Plus aria-hidden />
        </Button>
      </Tip>
    </div>
  )
}

interface MachineStatusPopoverProps {
  children: ReactNode
  label: string
  panel: (close: () => void) => ReactNode
}

/** The machine's status as its own button (IA quick win 619:1788): a click, Enter or Space opens the
 *  machine's health popover with focus in it; a resting mouse opens it too, without taking focus,
 *  and it stays while the pointer is over the status or the popover. */
function MachineStatusPopover({ children, label, panel }: MachineStatusPopoverProps) {
  const { cancel, close, hoverHandlers, open, setOpen } = useRestingHover()

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
