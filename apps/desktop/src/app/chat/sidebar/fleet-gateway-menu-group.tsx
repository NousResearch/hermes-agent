import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { StatusDot } from '@/components/status-dot'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenuItem,
  DropdownMenuLabel,
  dropdownMenuSectionLabel,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $profileColors } from '@/store/profile'

import { ConnectionGlyph } from './connection-glyph'
import type { FleetAgent, FleetGroup } from './fleet-rail'

interface FleetGatewayMenuGroupProps {
  group: FleetGroup
  onSelect: (agent: FleetAgent) => void
  /** `data-slot` on the group wrapper; each menu keeps its own for tests. */
  slot: string
  /** Optional wrapper around each row (the rail menu adds a launch context menu). */
  wrapRow?: (row: ReactNode, agent: FleetAgent, label: string) => ReactNode
}

/** One remote gateway's section in a profile menu: separator, gateway label
 *  (amber dot when unreachable), then a row per profile on that gateway.
 *  Shared by the rail's overflow dropdown and the compact profile dropdown. */
export function FleetGatewayMenuGroup({ group, onSelect, slot, wrapRow }: FleetGatewayMenuGroupProps) {
  const { t } = useI18n()
  const p = t.profiles
  const colors = useStore($profileColors)

  return (
    <div data-connection-id={group.connectionId} data-slot={slot}>
      <DropdownMenuSeparator />
      <DropdownMenuLabel className={cn(dropdownMenuSectionLabel, 'flex items-center gap-1.5')}>
        <ConnectionGlyph connection={group} />
        <span className="truncate">{group.label}</span>
        {!group.reachable && <StatusDot tone="warn" />}
      </DropdownMenuLabel>
      {[group.defaultAgent, ...group.named].map(agent => {
        const localDefault = agent.connectionKind === 'local' && agent.isDefault
        const label = localDefault ? p.fleet.localDevice : p.fleet.onGateway(agent.profile, group.label)

        const row = (
          <DropdownMenuItem aria-label={label} className="min-w-0" key={agent.profile} onSelect={() => onSelect(agent)}>
            <span className="flex min-w-0 items-center gap-1.5">
              {localDefault ? (
                <Codicon aria-hidden="true" name="device-desktop" size="0.875rem" />
              ) : (
                <ProfileGlyph
                  aria-hidden="true"
                  color={resolveProfileColor(agent.profile, colors)}
                  isDefault={agent.isDefault}
                  name={agent.profile}
                />
              )}
              <span className="truncate">{agent.profile}</span>
            </span>
          </DropdownMenuItem>
        )

        return wrapRow ? wrapRow(row, agent, label) : row
      })}
    </div>
  )
}
