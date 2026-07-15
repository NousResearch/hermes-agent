import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { deleteProfile } from '@/hermes'
import { useI18n } from '@/i18n'
import { removeProfileLocal } from '@/store/profile'

// Thin wrapper over ConfirmDialog: owns the deleteProfile call, inherits
// Enter-to-confirm + busy/done/error from the shared dialog. The single choke
// point for every delete entry point (rail + Profiles view).
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
  const { t } = useI18n()
  const p = t.profiles

  return (
    <ConfirmDialog
      busyLabel={p.deleting}
      confirmLabel={t.common.delete}
      description={
        profile ? (
          <>
            {p.deleteDescPrefix}
            <span className="font-medium text-foreground">{profile.name}</span>
            {p.deleteDescMid}
            <span className="font-mono text-xs">{profile.path}</span>
            {p.deleteDescSuffix}
          </>
        ) : null
      }
      destructive
      doneLabel={p.deleted}
      onClose={onClose}
      onConfirm={async () => {
        if (!profile) {
          return
        }

        await deleteProfile(profile.name)
        // Drop it from the rail immediately and unwind any routing still
        // pointed at it (gateway socket, active profile, new-chat target) —
        // the onDeleted refresh is best-effort and can fail when the deleted
        // profile's own backend was the routed one, which would otherwise
        // strand the gateway reconnect loop on a backend that can never
        // come back.
        removeProfileLocal(profile.name)
        await onDeleted?.()
      }}
      open={open}
      title={p.deleteTitle}
    />
  )
}
