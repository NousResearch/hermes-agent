import type { ProfileScope } from '@/api/client'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { deleteProfile } from '@/hermes'
import { useI18n } from '@/i18n'
import { retireLocalProfileGateways } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey, selectProfile, setActiveProfile } from '@/store/profile'
import { dropTilesForProfile } from '@/store/session-states'

// Thin wrapper over ConfirmDialog: owns the deleteProfile call, inherits
// Enter-to-confirm + busy/done/error from the shared dialog. The single choke
// point for every delete entry point (rail + Profiles view).
export function DeleteProfileDialog({
  gatewayLabel,
  profile,
  onClose,
  onDeleted,
  open,
  scope
}: {
  /** Names the owning machine in the copy when the profile lives on a gateway
   *  other than the foreground one — two "omer"s must never read the same. */
  gatewayLabel?: string
  profile: { name: string; path: string } | null
  onClose: () => void
  onDeleted?: () => Promise<void> | void
  open: boolean
  /** Explicit (connection, profile) owner for a remote-gateway profile. The
   *  delete then executes on THAT gateway and never touches local backends. */
  scope?: ProfileScope
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
            {gatewayLabel ? p.fleet.deleteOn(gatewayLabel) : null}
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

        // Deleting the profile the live gateway is on strands it on a dead
        // backend. Capture that before the delete; reset *after* the host's
        // onDeleted refresh so our reset is the last write — a refreshActiveProfile
        // racing the (still-dying) backend can't clobber the pill back to it.
        const remote = scope !== undefined && scope !== null

        const wasActive =
          !remote && normalizeProfileKey(profile.name) === normalizeProfileKey($activeGatewayProfile.get())

        if (!remote) {
          retireLocalProfileGateways(profile.name)
        }

        // Legacy arity when unscoped: callers and tests pin the one-arg call.
        await (remote ? deleteProfile(profile.name, scope) : deleteProfile(profile.name))
        // The profile is gone. Drop its persisted tiles now — a leftover
        // session/Bot tile restores on relaunch and dials the deleted
        // profile's backend, whose ensure_hermes_home() re-creates the
        // directory the delete just removed (hermes-agent#94235).
        dropTilesForProfile(profile.name)
        await onDeleted?.()

        if (wasActive) {
          // Swap gateway/sidebar to default and set the pill now — the primary
          // backend is always default, so this is correct, not just optimistic.
          selectProfile('default')
          setActiveProfile('default')
        }
      }}
      open={open}
      title={p.deleteTitle}
    />
  )
}
