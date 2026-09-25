import { useStore } from '@nanostores/react'

import { $newChatProfile } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import { $sessions, sessionMatchesStoredId } from '@/store/session'

import { ProfileTag } from './profile-tag'

export interface SessionTabProfileTagProps {
  /** The tab's stored session id — null on the fresh draft that has no
   *  session yet (the same key the status dot takes). */
  storedSessionId: null | string
}

/**
 * A session tab's LEAD IDENTITY — the owning profile's colored glyph + name
 * at the very start of the tab, before its status dot, under one
 * "Profile: <name>" tip.
 *
 * Self-subscribing, the way the tab's status dot is (`watchSessionTiles`:
 * "Self-subscribing … so the strip needn't re-sync"): the strip registers its
 * lead once, so identity resolves live HERE instead of being frozen at
 * registration — a draft gaining its stored row on first turn updates in
 * place, no re-register.
 *
 * Resolution: the stored row's `profile`, found in recents first, then the
 * project tree (the two sources `tileStoredRow` walks — a tab opened from a
 * project group can predate the recents page); with no row anywhere the tab
 * is a draft, so it shows the profile its first turn would be created under
 * (`$newChatProfile`, null → default).
 */
export function SessionTabProfileTag({ storedSessionId }: SessionTabProfileTagProps) {
  const sessions = useStore($sessions)
  const projectTree = useStore($projectTree)
  const newChatProfile = useStore($newChatProfile)

  const row =
    storedSessionId === null
      ? undefined
      : (sessions.find(session => sessionMatchesStoredId(session, storedSessionId)) ??
        projectTree
          .flatMap(project => [
            ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions)),
            ...(project.previewSessions ?? [])
          ])
          .find(session => sessionMatchesStoredId(session, storedSessionId)))

  return <ProfileTag profile={row ? row.profile : newChatProfile} showName />
}
