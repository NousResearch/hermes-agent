import { useStore } from '@nanostores/react'

import { $activeGatewayProfile, $newChatProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  sessionMatchesStoredId
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { ProfileTag } from './profile-tag'

export interface SessionTabProfileTagProps {
  storedSessionId: null | string
}

/**
 * Tab lead identity. Owner ladder mirrors ownerLookupSessionRows() (recents,
 * cron, messaging, unlisted-draft stubs) plus the project tree; each slice is
 * subscribed via useStore so the lead stays live. A tab with no row anywhere
 * is the main draft: it shows the profile the create path will use
 * ($newChatProfile || active gateway profile).
 */
export function SessionTabProfileTag({ storedSessionId }: SessionTabProfileTagProps) {
  const sessions = useStore($sessions)
  const cron = useStore($cronSessions)
  const messaging = useStore($messagingSessions)
  const stubs = useStore($unlistedSessionOwnerRows)
  const projectTree = useStore($projectTree)
  const newChatProfile = useStore($newChatProfile)
  const activeProfile = useStore($activeGatewayProfile)
  const profiles = useStore($profiles)

  // Same rule as the chat header (#66003): ownership is only worth a slot
  // once a second profile exists; single-profile tabs stay unchanged.
  if (profiles.length <= 1) {
    return null
  }

  let row: SessionInfo | undefined

  if (storedSessionId !== null) {
    const match = (s: SessionInfo) => sessionMatchesStoredId(s, storedSessionId)

    row =
      sessions.find(match) ??
      cron.find(match) ??
      messaging.find(match) ??
      stubs.find(match) ??
      projectTree
        .flatMap(p => [...p.repos.flatMap(r => r.groups.flatMap(g => g.sessions)), ...(p.previewSessions ?? [])])
        .find(match)
  }

  return <ProfileTag profile={row ? row.profile : newChatProfile || normalizeProfileKey(activeProfile)} showName />
}
