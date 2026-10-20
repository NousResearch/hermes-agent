import { useI18n, useValue } from '@hermes/plugin-sdk'

import { $lastRoster, useRoster } from './data'
import { useBots } from './i18n'
import { useRosterDialogState } from './roster-dialog-state'
import { renderRosterDialogs } from './roster-pane-dialogs'

export function RosterDialogHost() {
  const state = useRosterDialogState()

  const open =
    state.createOpen ||
    state.groupCreateOpen ||
    state.editing ||
    state.deleting ||
    state.deletingGroup ||
    state.grouping ||
    state.sectionDialog

  return open ? <OpenRosterDialogs /> : null
}

function OpenRosterDialogs() {
  const { t } = useI18n()
  const b = useBots()
  const state = useRosterDialogState()
  const { data, refetch } = useRoster()
  const cached = useValue($lastRoster)
  const roster = Array.isArray(data?.profiles) ? data.profiles : cached

  return renderRosterDialogs({
    ...state,
    b,
    t,
    roster,
    activeSourceRoster: roster.filter(bot => !bot.remoteSource),
    refetch
  })
}
