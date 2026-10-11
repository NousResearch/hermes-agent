/**
 * Kanban alerts-mode picker (#123596). One row, two mounts: Settings →
 * Notifications (`NOTIFICATIONS_AREAS.extra`) and the board's own settings
 * panel. The preference is device-global (see ./alerts-mode), so the board
 * mount carries an "On this device" caption to keep it from reading as
 * backend orchestration config.
 */

import { ListRow, Select, SelectContent, SelectItem, SelectTrigger, SelectValue, useValue } from '@hermes/plugin-sdk'

import { $alertsMode, ALERTS_MODES, parseAlertsMode } from './alerts-mode'
import { useKanban } from './ui'

export function AlertsModeRow({ showDeviceScope = false }: { showDeviceScope?: boolean }) {
  const k = useKanban()
  const mode = useValue($alertsMode)

  return (
    <ListRow
      action={
        <Select onValueChange={value => $alertsMode.set(parseAlertsMode(value))} value={mode}>
          <SelectTrigger aria-label={k.alerts.title} className="min-w-44 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ALERTS_MODES.map(option => (
              <SelectItem key={option} value={option}>
                {k.alerts[option]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
      description={showDeviceScope ? `${k.alerts.thisDevice} · ${k.alerts[`${mode}Desc`]}` : k.alerts[`${mode}Desc`]}
      title={k.alerts.title}
    />
  )
}
