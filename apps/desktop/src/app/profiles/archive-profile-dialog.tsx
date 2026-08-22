import { useEffect, useState } from 'react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { archiveProfile, getProfileArchiveManifest, type ProfileArchiveManifest } from '@/hermes'
import { useI18n } from '@/i18n'
import { retireLocalProfileGateways } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey, selectProfile, setActiveProfile } from '@/store/profile'

export function ArchiveProfileDialog({
  profile,
  onArchived,
  onClose,
  open
}: {
  profile: { name: string; path: string } | null
  onArchived?: () => Promise<void> | void
  onClose: () => void
  open: boolean
}) {
  const { t } = useI18n()
  const p = t.profiles
  const genericFailure = t.errors.genericFailure
  const [manifest, setManifest] = useState<null | ProfileArchiveManifest>(null)
  const [manifestError, setManifestError] = useState<null | string>(null)

  useEffect(() => {
    setManifest(null)
    setManifestError(null)

    if (!open || !profile) {
      return
    }

    let current = true

    void getProfileArchiveManifest(profile.name)
      .then(value => {
        if (current) {
          setManifest(value)
        }
      })
      .catch(error => {
        if (current) {
          setManifestError(error instanceof Error ? error.message : genericFailure)
        }
      })

    return () => {
      current = false
    }
  }, [genericFailure, open, profile])

  return (
    <ConfirmDialog
      busyLabel={p.archiving}
      confirmLabel={p.archiveMenu}
      description={
        profile ? (
          <div className="space-y-3">
            <p>{p.archiveDesc(profile.name)}</p>
            <div>
              <p className="font-medium text-foreground">{p.archivePreserved}</p>
              <p className="font-mono text-xs">{manifest?.preserved.join(', ') ?? manifestError ?? p.loading}</p>
            </div>
            <div>
              <p className="font-medium text-foreground">{p.archiveExcluded}</p>
              <p className="text-xs">{manifest?.excluded.join(', ') ?? p.loading}</p>
            </div>
          </div>
        ) : null
      }
      doneLabel={p.archived}
      onClose={onClose}
      onConfirm={async () => {
        if (!profile) {
          return
        }

        if (!manifest) {
          throw new Error(manifestError ?? p.archiveManifestLoading)
        }

        const wasActive = normalizeProfileKey(profile.name) === normalizeProfileKey($activeGatewayProfile.get())

        retireLocalProfileGateways(profile.name)
        await archiveProfile(profile.name)
        await onArchived?.()

        if (wasActive) {
          selectProfile('default')
          setActiveProfile('default')
        }
      }}
      open={open}
      title={p.archiveTitle}
    />
  )
}
