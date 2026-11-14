import { useStore } from '@nanostores/react'

import {
  $workspaceMode,
  $workspaceNewSessionTarget,
  type WorkspaceMode,
  type WorkspaceNewSessionTarget
} from '@/components/pane-shell/workspace-scope'
import type { HermesConnection } from '@/global'
import { ALL_PROFILES, sidebarProfileForScope } from '@/store/profile'
import { $connection } from '@/store/session'

export interface BotModeFetchScope {
  displayProfile: string
  fetchProfile: string
}

/** The registry source the window's gateway (and so its ambient list and
 *  search reads) is on: the same identity the plugin SDK publishes as
 *  `host.state.connectionId`, so a bot route resolved from either compares. */
function windowConnectionId(connection: HermesConnection | null): string {
  return connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')
}

/**
 * Bot Mode fetch scope (#126732): with a bot selected, the Sessions sidebar
 * lists THAT bot's profile, where its Telegram threads live, not the ambient
 * gateway scope's. The workspace target carries the route: `profile` is the
 * logical name the fetch uses, `targetProfile` the backend name rows carry.
 * A blocked or empty target (group chat, no selection) and the explicit
 * All-profiles view (already a fan-out) return null. So does a bot on a
 * connection other than the one the window's requests ride: the sidebar's
 * list/search reads go to the ambient connection, so re-scoping them to a
 * bot on another source would paint this source's rows under that bot.
 * Reads workspace state only: never parses the opaque owner key, never
 * imports the Bots plugin.
 */
export function botModeFetchScope(profileScope: string): BotModeFetchScope | null {
  return scopeFor(profileScope, $workspaceMode.get(), $workspaceNewSessionTarget.get(), $connection.get())
}

function scopeFor(
  profileScope: string,
  mode: WorkspaceMode,
  target: WorkspaceNewSessionTarget | null,
  connection: HermesConnection | null
): BotModeFetchScope | null {
  if (profileScope === ALL_PROFILES || mode !== 'bots') {
    return null
  }

  if (target?.kind !== 'route' || !target.route || target.route.connectionId !== windowConnectionId(connection)) {
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
 *  (a mode flip, bot to bot inside Bot Mode, or the window's request
 *  connection moving onto or off the bot's source); `$profileScope` stays put. */
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
  const unsubConnection = $connection.listen(check)

  return () => {
    unsubMode()
    unsubTarget()
    unsubConnection()
  }
}

/** The profile scope the sidebar renders, the profile its rows match, and
 *  the scope its server-side search reads: the selected bot's backend profile
 *  in Bot Mode (rows carry `targetProfile`) and its logical fetch profile,
 *  else `profileScope` unchanged for all three. */
export function useSidebarDisplayScope(
  profileScope: string
): [scope: string, profile: string, searchScope: string] {
  // The subscribed values feed the computation (not re-read inside it), so a
  // bot or connection switch recomputes under the React compiler's memo.
  const botScope = scopeFor(
    profileScope,
    useStore($workspaceMode),
    useStore($workspaceNewSessionTarget),
    useStore($connection)
  )

  const scope = botScope?.displayProfile ?? profileScope

  return [scope, sidebarProfileForScope(scope), botScope?.fetchProfile ?? profileScope]
}
