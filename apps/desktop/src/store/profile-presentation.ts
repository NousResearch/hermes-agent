import { dropStatusDrawersForProfile, migrateStatusDrawersForProfile } from './composer-status-drawer'
import { dropPreviewArtifactsForProfile, migratePreviewArtifactsForProfile } from './preview-status'
import { dropUnreadMutesForScope, migrateUnreadMutesForProfile } from './session-unread-mute'

interface ProfilePresentationRoute {
  connectionId?: string
  profile?: string
  targetProfile?: string
}

/** The renderer-owned profile families follow the same delete/rename lifecycle as tabs. */
export function dropProfilePresentation(
  profile: string,
  route: ProfilePresentationRoute | undefined,
  ambientConnection: string
): void {
  dropPreviewArtifactsForProfile(profile, route)
  dropStatusDrawersForProfile(profile, route)
  dropUnreadMutesForScope(route ? route.connectionId : ambientConnection, route?.profile || profile)
}

export function migrateProfilePresentation(from: string, to: string): void {
  migratePreviewArtifactsForProfile(from, to)
  migrateStatusDrawersForProfile(from, to)
  migrateUnreadMutesForProfile(from, to)
}
