import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { deleteProfile } from '@/hermes'

// Thin wrapper over ConfirmDialog: owns the deleteProfile call, inherits
// Enter-to-confirm + busy/done/error from the shared dialog.
export function DeleteProfileDialog({
  profile,
  onClose,
  onDeleted,
  open
}: {
  profile: { name: string; path: string } | null
  onClose: () => void
  onDeleted?: () => Promise<void> | void
  open: boolean
}) {
  return (
    <ConfirmDialog
      busyLabel="Deleting…"
      confirmLabel="Delete"
      description={
        profile ? (
          <>
            This will delete <span className="font-medium text-foreground">{profile.name}</span> and remove its{' '}
            <span className="font-mono text-xs">{profile.path}</span> directory. This cannot be undone.
          </>
        ) : null
      }
      destructive
      doneLabel="Deleted"
      onClose={onClose}
      onConfirm={async () => {
        if (!profile) {
          return
        }

        await deleteProfile(profile.name)
        await onDeleted?.()
      }}
      open={open}
      title="Delete profile?"
    />
  )
}
