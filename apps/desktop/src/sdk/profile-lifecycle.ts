/**
 * Profile lifecycle verbs a plugin may run on the user's behalf: delete,
 * export, import. Each goes through the same door the app's own profile
 * surfaces use, so a plugin action and a hand click can never disagree.
 */

import { computed } from 'nanostores'

import { deleteProfile } from '@/hermes'
import { activeGatewayConnectionId, retireLocalProfileGateways } from '@/store/gateway'
import {
  $activeGatewayProfile,
  normalizeProfileKey,
  refreshProfiles,
  selectProfile,
  setActiveProfile
} from '@/store/profile'
import { runExportProfileFlow, runImportProfileFlow } from '@/store/profile-share'
import { $connection } from '@/store/session'
import { dropTilesForProfile } from '@/store/session-states'

import type { PluginProfileRoute } from './index'

export const $activeConnectionId = computed($connection, connection => {
  if (!connection) {
    return null
  }

  if (connection.connectionId) {
    return connection.connectionId
  }

  // mode:'local' used to report null, which made Bot Mode fall back to the
  // registry primary (often an SSH box) and treat Spark as the active source
  // while this window was actually local.
  return connection.mode === 'local' ? 'local' : null
})

export const profileLifecycleHost = {
  /** Delete a profile THROUGH the desktop's teardown-routed REST path — the
   *  same door core surfaces use (DeleteProfileDialog). Electron intercepts
   *  the DELETE, tears down that profile's pool/primary backend first, and
   *  routes the follow-up request away from it, so a live (or hover-warmed)
   *  backend can't hold the profile dir open or respawn mid-delete and
   *  resurrect the directory (issue #52279). Plugins must prefer this over
   *  `cli.exec ['profile','delete',…]`, which bypasses that interception
   *  entirely. When the deleted profile was the live gateway's, the app is
   *  re-homed to the default profile — same semantics as the core dialog.
   *  Rejects with the backend's error when the delete fails. */
  deleteProfile: async (profile: string | PluginProfileRoute): Promise<void> => {
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

    const targetProfile = route?.targetProfile || name
    // A name-only call is ambient, not local: Bot Mode's active SSH roster
    // rows deliberately use the ambient gateway door and therefore carry no
    // explicit owner route. Preserve the active registry connection so the
    // profile teardown and DELETE both land on the VPS instead of retiring the
    // unrelated local pool and leaving the warmed remote backend to recreate
    // the deleted profile.
    const ambientConnectionId = route ? null : String(activeGatewayConnectionId() || '').trim()

    const ambientRemoteConnectionId =
      ambientConnectionId && ambientConnectionId !== 'local' ? ambientConnectionId : null

    if (!name) {
      throw new Error('deleteProfile: profile name required')
    }

    if (normalizeProfileKey(targetProfile) === 'default') {
      throw new Error('The default profile cannot be deleted.')
    }

    // Capture before the delete; re-home after so our write is the last one
    // (mirrors DeleteProfileDialog — a refreshActiveProfile racing the dying
    // backend can't clobber the pill back to the deleted profile).
    const wasActive = route
      ? route.connectionId === ($activeConnectionId.get() || '') &&
        normalizeProfileKey(route.profile) === normalizeProfileKey($activeGatewayProfile.get())
      : normalizeProfileKey(name) === normalizeProfileKey($activeGatewayProfile.get())

    // A hover-warmed Bot Mode row owns a retained renderer socket. Retire it
    // before Electron stops the profile backend so the socket closure cannot
    // schedule a reconnect that resurrects the deleted profile.
    if (route?.mode === 'local' || (!route && !ambientRemoteConnectionId)) {
      retireLocalProfileGateways(targetProfile)
    }

    await deleteProfile(
      targetProfile,
      route
        ? { connectionId: route.connectionId, profile: route.profile }
        : ambientRemoteConnectionId
          ? { connectionId: ambientRemoteConnectionId, profile: name }
          : undefined
    )

    // The profile is gone. Drop its persisted tiles now — a leftover tile
    // restores on relaunch and re-creates the deleted profile (hermes-agent#94235).
    dropTilesForProfile(
      route ? route.profile : name,
      route
        ? { connectionId: route.connectionId, profile: route.profile, targetProfile: route.targetProfile }
        : undefined
    )

    // The profile rail paints from the shared $profiles cache; without a
    // refresh the deleted profile's badge survives and clicking it starts a
    // doomed spawn-retry loop against Electron's deletion guard (#88769).
    // Best-effort: the delete itself already succeeded.
    await refreshProfiles().catch(() => undefined)

    if (wasActive) {
      selectProfile('default')
      setActiveProfile('default')
    }
  },

  /** Save a profile as a portable `.tar.gz` (config, skills, SOUL.md, cron,
   *  avatar, Bot Mode metadata — credentials excluded) through the same native
   *  save dialog + toasts as the core "Export profile…" menu. Resolves to the
   *  archive path, or null when the user cancelled or the export failed. */
  exportProfile: (profile: string): Promise<null | string> => runExportProfileFlow(profile),

  /** Pick a profile archive and import it as a new profile, WITHOUT switching
   *  the app into it. Resolves to the new profile name, or null when the user
   *  cancelled or the import failed (already toasted). */
  importProfile: (): Promise<null | string> => runImportProfileFlow({ select: false })
}
