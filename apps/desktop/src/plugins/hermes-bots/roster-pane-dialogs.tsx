import { ConfirmDialog, host } from '@hermes/plugin-sdk'
import type { useI18n } from '@hermes/plugin-sdk'

import { CreateAgentDialog, CreateGroupChatDialog, GroupDialog } from './create-dialog'
import type { useRoster } from './data'
import { EditProfileDialog } from './edit-profile-dialog'
import { disbandGroupChat, openGroupChat } from './group-chat-view'
import type { useBots } from './i18n'
import { deleteBot } from './profile-ops'
import type { GroupMember, RosterRow } from './types'
import { createBotSection, renameBotSection } from './user-sections'
import { SectionNameDialog } from './user-sections-ui'

interface renderRosterDialogsProps {
  b: ReturnType<typeof useBots>
  t: ReturnType<typeof useI18n>['t']
  createOpen: boolean
  setCreateOpen: (value: boolean) => void
  groupCreateOpen: boolean
  setGroupCreateOpen: (value: boolean) => void
  editing: RosterRow | null
  setEditing: (value: RosterRow | null) => void
  deleting: (RosterRow & { path?: string }) | null
  setDeleting: (value: (RosterRow & { path?: string }) | null) => void
  deletingGroup: { members: GroupMember[]; name: string } | null
  setDeletingGroup: (value: { members: GroupMember[]; name: string } | null) => void
  grouping: RosterRow | null
  setGrouping: (value: RosterRow | null) => void
  sectionDialog: null | { bot?: RosterRow; mode: 'create' } | { id: string; mode: 'rename'; name: string }
  setSectionDialog: (
    value: null | { bot?: RosterRow; mode: 'create' } | { id: string; mode: 'rename'; name: string }
  ) => void
  roster: RosterRow[]
  activeSourceRoster: RosterRow[]
  refetch: ReturnType<typeof useRoster>['refetch']
}

export function renderRosterDialogs({
  b,
  t,
  createOpen,
  setCreateOpen,
  groupCreateOpen,
  setGroupCreateOpen,
  editing,
  setEditing,
  deleting,
  setDeleting,
  deletingGroup,
  setDeletingGroup,
  grouping,
  setGrouping,
  sectionDialog,
  setSectionDialog,
  roster,
  activeSourceRoster,
  refetch
}: renderRosterDialogsProps) {
  return (
    <>
      <CreateAgentDialog
        onClose={() => {
          setCreateOpen(false)
          void refetch()
        }}
        open={createOpen}
        roster={activeSourceRoster}
      />
      <CreateGroupChatDialog
        onClose={() => setGroupCreateOpen(false)}
        onCreated={groupName => openGroupChat(groupName)}
        open={groupCreateOpen} // Full multi-source roster: group chats can seat bots from other
        // registered connections — their turns route to their own machines.
        roster={roster}
      />
      <SectionNameDialog
        initialName={sectionDialog?.mode === 'rename' ? sectionDialog.name : ''}
        mode={sectionDialog?.mode === 'rename' ? 'rename' : 'create'}
        onOpenChange={open => {
          if (!open) {
            setSectionDialog(null)
          }
        }}
        onSubmit={name => {
          if (sectionDialog?.mode === 'rename') {
            renameBotSection(sectionDialog.id, name)
          } else {
            createBotSection(name, sectionDialog?.bot ? [sectionDialog.bot] : [])
          }
        }}
        open={Boolean(sectionDialog)}
      />
      <EditProfileDialog
        bot={editing}
        onClose={() => {
          setEditing(null)
          void refetch()
        }}
        open={Boolean(editing)}
      />
      {grouping ? <GroupDialog bot={grouping} onClose={() => setGrouping(null)} /> : null}
      <ConfirmDialog
        busyLabel="Deleting…"
        confirmLabel={t.common.delete}
        description={
          deleting ? (
            <span>
              {'This will permanently delete the bot '}
              <span className="font-medium text-foreground">{deleting.name}</span>
              {' and its associated Hermes profile at '}
              <span className="font-mono text-xs">{deleting.path}</span>. This cannot be undone.
            </span>
          ) : null
        }
        destructive
        doneLabel="Deleted"
        onClose={() => setDeleting(null)}
        onConfirm={async () => {
          if (!deleting) {
            return
          }

          const name = deleting.name
          await deleteBot(deleting)
          await refetch()
          host.notify({
            kind: 'success',
            message: `Deleted profile ${name}`
          })
        }}
        open={Boolean(deleting)}
        title={b.bot.deleteTitle}
      />
      <ConfirmDialog
        busyLabel="Deleting…"
        confirmLabel={b.group.deleteAction}
        description={
          deletingGroup
            ? `This removes “${deletingGroup.name}” from its bots and clears the shared room log. The bots and their individual chats are kept.`
            : null
        }
        destructive
        doneLabel="Deleted"
        onClose={() => setDeletingGroup(null)}
        onConfirm={async () => {
          if (!deletingGroup) {
            return
          }

          await disbandGroupChat(deletingGroup.name, deletingGroup.members)
          host.notify({
            kind: 'success',
            message: `Deleted group “${deletingGroup.name}”`
          })
        }}
        open={Boolean(deletingGroup)}
        title={b.group.deleteTitle}
      />
    </>
  )
}
