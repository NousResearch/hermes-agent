import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $defaultProfileRoute } from '@/store/default-profile'
import { notifyError } from '@/store/notifications'
import {
  $newChatProfile,
  $newChatRoute,
  type AgentProfileRoute,
  captureNewChatSource,
  clearNewChatIntent,
  ensureGatewayAgent,
  ensureGatewayProfile,
  pinLegacyNewChatProfile
} from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import {
  isPeerInstanceWindow,
  isProfilePinnedWindow,
  isSecondaryWindowSearch,
  windowConnectionOverride,
  windowProfileOverride
} from '@/store/windows'

/** Only generic New Session actions consult this preference. Explicit profile,
 * agent, project and existing-session actions keep their captured owner. */
export function defaultNewSessionTarget(): { profile: string; route: AgentProfileRoute | null } | null {
  const project = $projectScope.get()

  if (project !== ALL_PROJECTS && project !== NO_PROJECT_ID) {
    return null
  }

  // An ordinary New Window inherits its opener only for boot. Reusing that
  // seed here would undo a later device/profile selection. Only an explicit
  // "Open profile in new window" makes the peer's launch route a default.
  //
  // A `win=secondary` pop-out carries the same `?profile=`/`?connectionId=`
  // pair, but that is the route of the ONE session it was opened to display —
  // not a preference for every future new chat. Treating it as a default made
  // a pop-out opened on a local session keep minting local chats after the
  // user pointed the window at a remote gateway. Let secondary windows fall
  // through to the live active source instead.
  const search = typeof window === 'undefined' ? '' : window.location.search
  const usesWindowRoute = isProfilePinnedWindow(search) || (!isPeerInstanceWindow(search) && !isSecondaryWindowSearch(search))
  const profile = usesWindowRoute ? windowProfileOverride() : null
  const saved = profile ? { connectionId: windowConnectionOverride(), profile } : $defaultProfileRoute.get()

  if (!saved) {
    return null
  }

  // A captured target can explicitly choose the legacy profile door (null
  // route), distinct from no default and from the override-bypassing `local`.
  return {
    profile: saved.profile,
    route: saved.connectionId === null ? null : { connectionId: saved.connectionId, profile: saved.profile }
  }
}

export function prepareDefaultNewSession(): void {
  // A generic New session starts a NEW draft: forget any owner a previous
  // draft (or this window's boot hint) pinned, then apply the configured
  // default if there is one. Otherwise the draft resolves from the live
  // active gateway, so a stale route can't hijack it.
  clearNewChatIntent()

  const target = defaultNewSessionTarget()

  if (!target) {
    return
  }

  if (target.route) {
    $newChatProfile.set(target.profile)
    $newChatRoute.set(target.route)
    captureNewChatSource(target.route.connectionId)
  } else {
    pinLegacyNewChatProfile(target.profile)
  }

  const activation = target.route
    ? ensureGatewayAgent(target.route.connectionId, target.profile)
    : ensureGatewayProfile(target.profile, { forceLegacyRoute: true })

  void activation.catch(error => {
    notifyError(error, `Failed to open profile "${target.profile}"`)
  })
}
