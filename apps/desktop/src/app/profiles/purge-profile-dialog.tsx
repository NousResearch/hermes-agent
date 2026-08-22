import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { purgeProfile } from '@/hermes'
import { useI18n } from '@/i18n'

export function PurgeProfileDialog({
  profile,
  onClose,
  onPurged,
  open
}: {
  profile: { name: string; path: string } | null
  onClose: () => void
  onPurged?: () => Promise<void> | void
  open: boolean
}) {
  const { t } = useI18n()
  const p = t.profiles

  return (
    <ConfirmDialog
      busyLabel={p.purging}
      confirmLabel={p.purgeMenu}
      description={profile ? p.purgeDesc(profile.name) : null}
      destructive
      doneLabel={p.purged}
      onClose={onClose}
      onConfirm={async () => {
        if (!profile) {
          return
        }

        await purgeProfile(profile.name)
        await onPurged?.()
      }}
      open={open}
      title={p.purgeTitle}
    />
  )
}
