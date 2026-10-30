import { ConfirmDialog, host } from '@hermes/plugin-sdk'
import type { useI18n } from '@hermes/plugin-sdk'

import { disbandGroupChats } from './bulk-disband'
import { CreateAgentDialog, CreateGroupChatDialog, GroupDialog } from './create-dialog'
import type { useRoster } from './data'
import { EditProfileDialog } from './edit-profile-dialog'
import { disbandGroupChat, openGroupChat } from './group-chat-view'
import type { useBots } from './i18n'
import { deleteBot } from './profile-ops'
import type { GroupMember, RosterRow } from './types'
import { createBotSection, moveGroupChatsToSection, renameBotSection, type SectionDialogState } from './user-sections'
import { SectionNameDialog } from './user-sections-ui'

/** One room, or many rooms for bulk delete (still one confirm). */
export type DeletingGroupState =
  | { items: Array<{ members: GroupMember[]; name: string }>; kind?: 'bulk' | 'clear-all' }
  | { members: GroupMember[]; name: string }
  | null

function deletingGroupItems(
  deletingGroup: DeletingGroupState
): Array<{ members: GroupMember[]; name: string }> {
  if (!deletingGroup) {
    return []
  }
  if ('items' in deletingGroup && Array.isArray(deletingGroup.items)) {
    return deletingGroup.items
  }
  if ('name' in deletingGroup) {
    return [{ name: deletingGroup.name, members: deletingGroup.members }]
  }
  return []
}

function deletingGroupKind(deletingGroup: DeletingGroupState): 'bulk' | 'clear-all' | 'single' {
  if (!deletingGroup) {
    return 'single'
  }
  if ('items' in deletingGroup && Array.isArray(deletingGroup.items)) {
    if (deletingGroup.kind === 'clear-all') {
      return 'clear-all'
    }
    return deletingGroup.items.length > 1 ? 'bulk' : 'single'
  }
  return 'single'
}

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
  deletingGroup: DeletingGroupState
  setDeletingGroup: (value: DeletingGroupState) => void
  grouping: RosterRow | null
  setGrouping: (value: RosterRow | null) => void
  sectionDialog: SectionDialogState
  setSectionDialog: (value: SectionDialogState) => void
  roster: RosterRow[]
  activeSourceRoster: RosterRow[]
  refetch: ReturnType<typeof useRoster>['refetch']
  /** Clears multi-select after a successful bulk disband. */
  onGroupDisbandDone?: () => void
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
  refetch,
  onGroupDisbandDone
}: renderRosterDialogsProps) {
  const groupItems = deletingGroupItems(deletingGroup)
  const bulkCount = groupItems.length
  const kind = deletingGroupKind(deletingGroup)
  const bulk = kind === 'bulk' || kind === 'clear-all'
  const singleName = kind === 'single' && bulkCount === 1 ? groupItems[0]?.name : null

  return (
    <>
      <CreateAgentDialog
        onClose={() => {
          setCreateOpen(false)
          void refetch()
        }}
        onConfigureModel={setEditing}
        open={createOpen}
        roster={activeSourceRoster}
      />
      <CreateGroupChatDialog
        onClose={() => setGroupCreateOpen(false)}
        onCreated={(group, origin) => openGroupChat(group, origin)}
        open={groupCreateOpen} // The dialog applies the active gateway mode to
        // the full roster before allowing a group to be created.
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
            renameBotSection(sectionDialog.id, name, roster)
          } else {
            const section = createBotSection(name, sectionDialog?.bot ? [sectionDialog.bot] : [])

            if (section && sectionDialog?.group) {
              moveGroupChatsToSection([sectionDialog.group], section.id)
            }
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
        confirmLabel={
          kind === 'clear-all'
            ? b.group.clearAllAction
            : bulk
              ? b.group.deleteNAction(bulkCount)
              : b.group.deleteAction
        }
        description={
          kind === 'clear-all'
            ? b.group.clearAllDesc(bulkCount)
            : bulk
              ? b.group.deleteNDesc(bulkCount)
              : singleName
                ? `This removes “${singleName}” from its bots and clears the shared room log. The bots and their individual chats are kept.`
                : null
        }
        destructive
        doneLabel="Deleted"
        onClose={() => setDeletingGroup(null)}
        onConfirm={async () => {
          if (!groupItems.length) {
            return
          }

          if (bulk) {
            const { disbanded } = await disbandGroupChats(groupItems)
            onGroupDisbandDone?.()
            host.notify({
              kind: 'success',
              message: b.group.deletedN(disbanded.length)
            })
            return
          }

          const one = groupItems[0]
          await disbandGroupChat(one.name, one.members)
          onGroupDisbandDone?.()
          host.notify({
            kind: 'success',
            message: `Deleted group “${one.name}”`
          })
        }}
        open={bulkCount > 0}
        title={
          kind === 'clear-all'
            ? b.group.clearAllTitle
            : bulk
              ? b.group.deleteNTitle(bulkCount)
              : b.group.deleteTitle
        }
      />
    </>
  )
}
