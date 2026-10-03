import { atom, useValue } from '@hermes/plugin-sdk'

import type { GroupMember, RosterRow } from './types'
import type { SectionDialogState } from './user-sections'

interface RosterDialogState {
  createOpen: boolean
  groupCreateOpen: boolean
  editing: RosterRow | null
  deleting: (RosterRow & { path?: string }) | null
  deletingGroup: { members: GroupMember[]; name: string } | null
  grouping: RosterRow | null
  sectionDialog: SectionDialogState
}

const empty: RosterDialogState = {
  createOpen: false,
  groupCreateOpen: false,
  editing: null,
  deleting: null,
  deletingGroup: null,
  grouping: null,
  sectionDialog: null
}

// Only window-lifetime dialog intents are shared. Draft fields stay local to
// their mounted dialog and are never written to disk or profile settings.
const $dialogs = atom<RosterDialogState>(empty)

const set = <K extends keyof RosterDialogState>(key: K, value: RosterDialogState[K]) =>
  $dialogs.set({ ...$dialogs.get(), [key]: value })

const actions = {
  setCreateOpen: (value: boolean) => set('createOpen', value),
  setGroupCreateOpen: (value: boolean) => set('groupCreateOpen', value),
  setEditing: (value: RosterDialogState['editing']) => set('editing', value),
  setDeleting: (value: RosterDialogState['deleting']) => set('deleting', value),
  setDeletingGroup: (value: RosterDialogState['deletingGroup']) => set('deletingGroup', value),
  setGrouping: (value: RosterDialogState['grouping']) => set('grouping', value),
  setSectionDialog: (value: RosterDialogState['sectionDialog']) => set('sectionDialog', value)
}

export const resetRosterDialogs = () => $dialogs.set(empty)

export function useRosterDialogState() {
  return { ...useValue($dialogs), ...actions }
}
