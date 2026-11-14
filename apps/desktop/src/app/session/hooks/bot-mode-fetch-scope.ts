import { useStore } from '@nanostores/react'

import { $workspaceMode, $workspaceNewSessionTarget } from '@/components/pane-shell/workspace-scope'
import { ALL_PROFILES, sidebarProfileForScope } from '@/store/profile'

export interface BotModeFetchScope {
  displayProfile: string
  fetchProfile: string
}

/**
 * Bot Mode fetch scope (#126732): with a bot selected, the Sessions sidebar
 * lists THAT bot's profile, where its Telegram threads live, not the ambient
 * gateway scope's. The workspace target carries the route: `profile` is the
 * logical name the fetch uses, `targetProfile` the backend name rows carry.
 * A blocked or empty target (group chat, no selection) and the explicit
 * All-profiles view (already a fan-out) return null. Reads workspace state
 * only: never parses the opaque owner key, never imports the Bots plugin.
 */
export function botModeFetchScope(profileScope: string): BotModeFetchScope | null {
  if (profileScope === ALL_PROFILES || $workspaceMode.get() !== 'bots') {
    return null
  }

  const target = $workspaceNewSessionTarget.get()

  if (target?.kind !== 'route' || !target.route) {
    return null
  }

  const fetchProfile = String(target.route.profile || '').trim() || 'default'
  const displayProfile = String(target.route.targetProfile || target.route.profile || '').trim() || fetchProfile

  return { displayProfile, fetchProfile }
}

/** The profile a sidebar fetch for `profileScope` reads: the selected bot's in Bot Mode, else the scope's. */
export function sidebarFetchProfile(profileScope: string): string {
  return botModeFetchScope(profileScope)?.fetchProfile ?? sidebarProfileForScope(profileScope)
}

/** Profile names a sidebar refresh's failure map may report its scope under:
 *  the fetch name, plus the backend name when a remote alias differs from it. */
export function sidebarRetryProfiles(profileScope: string): string[] {
  const fetchProfile = sidebarFetchProfile(profileScope)
  const displayProfile = botModeFetchScope(profileScope)?.displayProfile

  return displayProfile && displayProfile !== fetchProfile ? [fetchProfile, displayProfile] : [fetchProfile]
}

/** Call `onChange` whenever a Bot Mode switch moves the sidebar fetch profile
 *  (a mode flip, or bot to bot inside Bot Mode); `$profileScope` stays put. */
export function watchSidebarFetchProfile(currentScope: () => string, onChange: () => void): () => void {
  let last = sidebarFetchProfile(currentScope())

  const check = () => {
    const next = sidebarFetchProfile(currentScope())

    if (next !== last) {
      last = next
      onChange()
    }
  }

  const unsubMode = $workspaceMode.listen(check)
  const unsubTarget = $workspaceNewSessionTarget.listen(check)

  return () => {
    unsubMode()
    unsubTarget()
  }
}

/** The profile scope the sidebar renders, and the profile its rows match:
 *  the selected bot's backend profile in Bot Mode (rows carry `targetProfile`),
 *  else `profileScope` unchanged. */
export function useSidebarDisplayScope(profileScope: string): [scope: string, profile: string] {
  // Subscribed so a bot switch re-renders; botModeFetchScope reads the same atoms.
  useStore($workspaceMode)
  useStore($workspaceNewSessionTarget)

  const scope = botModeFetchScope(profileScope)?.displayProfile ?? profileScope

  return [scope, sidebarProfileForScope(scope)]
}
