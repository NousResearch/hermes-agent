import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  dropdownMenuRow,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { AgentPluginRow } from '@/store/agent-plugins'
import { $pluginFleetByPlugin, fleetPluginKey, openFleetUpdate } from '@/store/plugin-fleet-updates'

/** The row's "Update to x" button. When the same plugin is also out of date in
 *  other profiles or on other gateways, a caret beside it offers the fleet-wide
 *  update; otherwise it is the plain single-profile button. `onUpdate` is the
 *  tab's own single-profile update (consent + scan override). */
export function UpdateButton({
  agent,
  busy,
  label,
  onUpdate,
  scopeLabel
}: {
  agent: AgentPluginRow
  busy: boolean
  label: string
  onUpdate: () => void
  scopeLabel: string
}) {
  const { t } = useI18n()
  const f = t.skills.plugins.fleet
  const fleetInstalls = useStore($pluginFleetByPlugin).get(fleetPluginKey(agent)) ?? []
  const elsewhere = fleetInstalls.length > 1
  const onUpdateEverywhere = () => openFleetUpdate(fleetInstalls, f.titleOne(agent.name))

  const main = (
    <Button
      className={cn('h-5 px-1.5 text-[0.65rem]', elsewhere && 'rounded-r-none')}
      disabled={busy}
      onClick={onUpdate}
      size="xs"
      variant="outline"
    >
      {label}
    </Button>
  )

  if (!elsewhere) {
    return main
  }

  return (
    <span className="inline-flex" data-testid="plugin-update-split">
      {main}
      <DropdownMenu>
        <DropdownMenuTrigger asChild disabled={busy}>
          <Button
            aria-label={f.updateEverywhere(fleetInstalls.length)}
            className="-ml-px h-5 w-3.5 rounded-l-none px-0"
            size="xs"
            variant="outline"
          >
            <ChevronDown className="size-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem className={dropdownMenuRow} onSelect={onUpdateEverywhere}>
            {f.updateEverywhere(fleetInstalls.length)}
          </DropdownMenuItem>
          <DropdownMenuItem className={dropdownMenuRow} onSelect={onUpdate}>
            {f.updateHereOnly(scopeLabel)}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  )
}
