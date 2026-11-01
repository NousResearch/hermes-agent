/** Settings ▸ Plugins ▸ Kanban — what the board tells you about worker events.
 *
 *  The host renders plugin settings pages inside `SettingsContent`, so the page
 *  returns its rows only.
 */

import { ToggleRow, useValue } from '@hermes/plugin-sdk'

import { useKanban } from './i18n'
import { $kanbanNotifyPrefs, setBlockedPopups } from './notify-prefs'

export function KanbanSettings() {
  const k = useKanban()
  const prefs = useValue($kanbanNotifyPrefs)

  return (
    <ToggleRow
      checked={prefs.blockedPopups}
      description={k.settings.blockedPopupsDesc}
      id="kanban-blocked-popups"
      label={k.settings.blockedPopups}
      onChange={setBlockedPopups}
    />
  )
}
