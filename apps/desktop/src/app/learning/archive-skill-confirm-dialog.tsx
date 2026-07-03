import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { deleteLearningNode } from '@/hermes'
import { notify } from '@/store/notifications'

export const ARCHIVE_SKILL_DESCRIPTION = 'The skill is archived and can be restored with `hermes curator restore`.'

export function notifySkillArchived(): void {
  // TODO(i18n): literals until the UX settles.
  notify({ kind: 'success', message: 'Restorable via hermes curator restore.', title: 'Skill archived' })
}

export async function archiveLearningSkill(id: string): Promise<void> {
  const res = await deleteLearningNode(id)

  if (!res.ok) {
    throw new Error(res.message || 'Archive failed')
  }
}

/** Fire-and-forget a mutation whose UI already applied optimistically; a failure just rolls it back + reports. */
export function fireOptimistic(action: Promise<void>, rollback: () => void, onFailure: (err: unknown) => void): void {
  void action.catch(err => {
    rollback()
    onFailure(err)
  })
}

interface ArchiveSkillConfirmDialogProps {
  /** Apply optimistic UI updates; return rollback if the background archive fails. */
  onApply: () => () => void
  onClose: () => void
  onFailure?: (err: unknown, skillName: string) => void
  onSuccess?: () => void
  open: boolean
  skillId: string
  skillName: string
}

/** Shared archive confirm for learned skills (capabilities page + memory graph). */
export function ArchiveSkillConfirmDialog({
  onApply,
  onClose,
  onFailure,
  onSuccess,
  open,
  skillId,
  skillName
}: ArchiveSkillConfirmDialogProps) {
  return (
    <ConfirmDialog
      confirmLabel="Archive"
      description={ARCHIVE_SKILL_DESCRIPTION}
      destructive
      dismissOnConfirm
      onClose={onClose}
      onConfirm={() => {
        const rollback = onApply()

        fireOptimistic(
          archiveLearningSkill(skillId).then(() => {
            notifySkillArchived()
            onSuccess?.()
          }),
          rollback,
          err => onFailure?.(err, skillName)
        )
      }}
      open={open}
      title={`Archive ${skillName}?`}
    />
  )
}
