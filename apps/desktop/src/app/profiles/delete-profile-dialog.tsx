import type { ProfileScope } from '@/api/client'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { PathText } from '@/components/ui/path-text'
import { deleteProfile } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { activeGatewayConnectionId, retireLocalProfileGateways } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey, selectProfile, setActiveProfile } from '@/store/profile'
import { dropTilesForProfile } from '@/store/session-states'
import { usePortalShell } from '@/store/shell-variant'

/** An agent page's own route: its machine, which may be this window's ('local') or another. */
export interface ProfileOwnerRoute {
  connectionId: string
  profile: string
}

/** Who the delete belongs to. `scope` is a profile on a gateway other than this window's (the classic
 *  rail's fleet squares): nothing local is retired and the window's agent never changes. `owner` is an
 *  agent page's resolved route, wherever it lives: a local owner retires its local gateways, and
 *  deleting the agent this window is on moves the window to the default agent. */
type DeleteProfileOwnership = { owner: ProfileOwnerRoute; scope?: never } | { owner?: never; scope?: ProfileScope }

// Thin wrapper over ConfirmDialog: owns the deleteProfile call, inherits
// Enter-to-confirm + busy/done/error from the shared dialog. The single choke
// point for every delete entry point (rail + Profiles view).
export function DeleteProfileDialog({
  gatewayLabel,
  profile,
  onClose,
  onDeleted,
  open,
  owner,
  scope
}: {
  /** Names the owning machine in the copy when the profile lives on a gateway
   *  other than the foreground one — two "omer"s must never read the same. */
  gatewayLabel?: string
  profile: { name: string; path: string } | null
  onClose: () => void
  onDeleted?: () => Promise<void> | void
  open: boolean
} & DeleteProfileOwnership) {
  const { t } = useI18n()
  const p = t.profiles
  const redesign = usePortalShell()

  return (
    <ConfirmDialog
      busyLabel={p.deleting}
      confirmLabel={t.common.delete}
      description={
        profile ? (
          <>
            {p.deleteDescPrefix}
            <span className={cn('font-medium text-foreground', redesign && 'break-words')}>{profile.name}</span>
            {gatewayLabel ? p.fleet.deleteOn(gatewayLabel) : null}
            {p.deleteDescMid}
            {redesign ? (
              <PathText className="max-w-full align-bottom text-xs" path={profile.path} />
            ) : (
              <span className="font-mono text-xs">{profile.path}</span>
            )}
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
        const ambient = activeGatewayConnectionId() ?? 'local'
        const route = owner ?? (scope == null ? undefined : scopedRoute(scope, profile.name, ambient))
        const local = owner ? owner.connectionId === 'local' : route === undefined

        const wasActive =
          (owner ? owner.connectionId === ambient : local) &&
          normalizeProfileKey(profile.name) === normalizeProfileKey($activeGatewayProfile.get())

        if (local) {
          retireLocalProfileGateways(profile.name)
        }

        // Legacy arity when unscoped: callers and tests pin the one-arg call.
        await (route ? deleteProfile(profile.name, owner ?? scope) : deleteProfile(profile.name))

        // The profile is gone. Drop its persisted tiles now — a leftover
        // session/Bot tile restores on relaunch and dials the deleted
        // profile's backend, whose ensure_hermes_home() re-creates the
        // directory the delete just removed (hermes-agent#94235). Only a
        // route-less drop reaches id-less Bot tiles, `url:` buckets and
        // target-matched tiles, so a delete with no owner stays route-less.
        if (route) {
          dropTilesForProfile(profile.name, route)
        } else {
          dropTilesForProfile(profile.name)
        }

        await onDeleted?.()

        // An owner's window may have moved during the delete; only the window still on it resets.
        const stillActive =
          !owner ||
          ((activeGatewayConnectionId() ?? 'local') === ambient &&
            normalizeProfileKey($activeGatewayProfile.get()) === normalizeProfileKey(profile.name))

        if (wasActive && stillActive) {
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

/** The tile route a `scope` delete drops: its gateway (the window's when unnamed) and profile. */
function scopedRoute(scope: NonNullable<ProfileScope>, name: string, ambient: string): ProfileOwnerRoute {
  return typeof scope === 'object'
    ? { connectionId: scope.connectionId ?? ambient, profile: scope.profile ?? name }
    : { connectionId: ambient, profile: scope }
}
