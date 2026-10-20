// `host.deleteProfile`: delete a named profile through the teardown-routed REST path, or reset the
// default profile, which cannot be removed (its home is the install and holds every other profile).
import type { ProfileScope } from '@/api/client'
import { deleteProfile, resetProfile } from '@/hermes'
import { activeGatewayConnectionId, retireLocalProfileGateways } from '@/store/gateway'
import {
  $activeGatewayProfile,
  normalizeProfileKey,
  refreshProfiles,
  selectProfile,
  setActiveProfile
} from '@/store/profile'
import { dropTilesForProfile } from '@/store/session-states'

import type { PluginProfileRoute } from './profile-request'

interface ProfileDeleteTarget {
  /** Logical name the caller used (the route's `profile` for a routed call). */
  name: string
  /** Owner pin for the REST call; undefined = the ambient local backend. */
  ownerScope?: ProfileScope
  /** True when a name-only call lands on the active remote registry connection. */
  remoteAmbient: boolean
  route: (PluginProfileRoute & { connectionId: string; profile: string; targetProfile: string }) | null
  /** The profile the backend acts on (an alias route's `targetProfile`). */
  targetProfile: string
}

function resolveDeleteTarget(profile: string | PluginProfileRoute): ProfileDeleteTarget {
  const route =
    typeof profile === 'string'
      ? null
      : {
          ...profile,
          connectionId: String(profile.connectionId || '').trim(),
          profile: String(profile.profile || '').trim(),
          targetProfile: String(profile.targetProfile || '').trim()
        }

  const name = typeof profile === 'string' ? profile.trim() : route?.profile || ''

  if (route && (!route.connectionId || !route.profile || !route.targetProfile)) {
    throw new Error('deleteProfile: route requires connectionId, profile, and targetProfile')
  }

  // A name-only call is ambient, not local: Bot Mode's active SSH roster
  // rows deliberately use the ambient gateway door and therefore carry no
  // explicit owner route. Preserve the active registry connection so the
  // profile teardown and DELETE both land on the VPS instead of retiring the
  // unrelated local pool and leaving the warmed remote backend to recreate
  // the deleted profile.
  const ambientConnectionId = route ? null : String(activeGatewayConnectionId() || '').trim()
  const ambientRemoteConnectionId = ambientConnectionId && ambientConnectionId !== 'local' ? ambientConnectionId : null

  if (!name) {
    throw new Error('deleteProfile: profile name required')
  }

  const ownerScope = route
    ? { connectionId: route.connectionId, profile: route.profile }
    : ambientRemoteConnectionId
      ? { connectionId: ambientRemoteConnectionId, profile: name }
      : undefined

  return {
    name,
    ownerScope,
    remoteAmbient: Boolean(ambientRemoteConnectionId),
    route,
    targetProfile: route?.targetProfile || name
  }
}

function dropTargetTiles({ name, route }: ProfileDeleteTarget): void {
  dropTilesForProfile(
    route ? route.profile : name,
    route ? { connectionId: route.connectionId, profile: route.profile, targetProfile: route.targetProfile } : undefined
  )
}

/** The default profile's backend is the gateway serving every other profile: it stays up, so
 *  nothing is retired and the app is not re-homed. Its chats are gone, so its tiles go too. */
async function resetDefaultProfile(target: ProfileDeleteTarget): Promise<void> {
  await resetProfile(target.targetProfile, target.ownerScope)
  dropTargetTiles(target)
  await refreshProfiles().catch(() => undefined)
}

async function deleteNamedProfile(target: ProfileDeleteTarget, activeConnectionId: () => null | string) {
  const { name, route } = target

  // Capture before the delete; re-home after so our write is the last one
  // (mirrors DeleteProfileDialog — a refreshActiveProfile racing the dying
  // backend can't clobber the pill back to the deleted profile).
  const wasActive = route
    ? route.connectionId === (activeConnectionId() || '') &&
      normalizeProfileKey(route.profile) === normalizeProfileKey($activeGatewayProfile.get())
    : normalizeProfileKey(name) === normalizeProfileKey($activeGatewayProfile.get())

  // A hover-warmed Bot Mode row owns a retained renderer socket. Retire it
  // before Electron stops the profile backend so the socket closure cannot
  // schedule a reconnect that resurrects the deleted profile.
  if (route?.mode === 'local' || (!route && !target.remoteAmbient)) {
    retireLocalProfileGateways(target.targetProfile)
  }

  await deleteProfile(target.targetProfile, target.ownerScope)

  // The profile is gone. Drop its persisted tiles now — a leftover tile
  // restores on relaunch and re-creates the deleted profile (hermes-agent#94235).
  dropTargetTiles(target)

  // The profile rail paints from the shared $profiles cache; without a
  // refresh the deleted profile's badge survives and clicking it starts a
  // doomed spawn-retry loop against Electron's deletion guard (#88769).
  // Best-effort: the delete itself already succeeded.
  await refreshProfiles().catch(() => undefined)

  if (wasActive) {
    selectProfile('default')
    setActiveProfile('default')
  }
}

export async function deletePluginProfile(
  profile: string | PluginProfileRoute,
  activeConnectionId: () => null | string
): Promise<void> {
  const target = resolveDeleteTarget(profile)

  if (normalizeProfileKey(target.targetProfile) === 'default') {
    return resetDefaultProfile(target)
  }

  return deleteNamedProfile(target, activeConnectionId)
}
