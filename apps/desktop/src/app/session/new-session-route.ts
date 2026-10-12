import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $defaultProfileRoute } from '@/store/default-profile'
import { $interfaceMode } from '@/store/interface-mode'
import { $sidebarProfileFilter } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  type AgentProfileRoute,
  captureNewChatSource,
  ensureGatewayAgent,
  ensureGatewayProfile,
  pinLegacyNewChatProfile,
  pinNewChatProfile,
  profilePickOwnerRoute
} from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import {
  isPeerInstanceWindow,
  isProfilePinnedWindow,
  windowConnectionOverride,
  windowProfileOverride
} from '@/store/windows'

/** Only generic New Session actions consult this preference. Explicit profile,
 * agent, project and existing-session actions keep their captured owner. */
export function defaultNewSessionTarget(): { profile: string; route: AgentProfileRoute | null } | null {
  if ($interfaceMode.get() === 'simple') {
    const filter = $sidebarProfileFilter.get()
    const profile = filter.length === 1 ? filter[0] : $activeGatewayProfile.get()

    return { profile, route: profilePickOwnerRoute(profile) }
  }

  const project = $projectScope.get()

  if (project !== ALL_PROJECTS && project !== NO_PROJECT_ID) {
    return null
  }

  // An ordinary New Window inherits its opener only for boot. Reusing that
  // seed here would undo a later device/profile selection. Only an explicit
  // "Open profile in new window" makes the peer's launch route a default.
  const profile = !isPeerInstanceWindow() || isProfilePinnedWindow() ? windowProfileOverride() : null
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
  const target = defaultNewSessionTarget()

  if (!target) {
    return
  }

  if (target.route) {
    $newChatProfile.set(target.profile)
    $newChatRoute.set(target.route)
    captureNewChatSource(target.route.connectionId)
  } else {
    if ($interfaceMode.get() === 'simple') {
      pinNewChatProfile(target.profile)
    } else {
      pinLegacyNewChatProfile(target.profile)
    }
  }

  const activation = target.route
    ? ensureGatewayAgent(target.route.connectionId, target.profile)
    : ensureGatewayProfile(target.profile, $interfaceMode.get() === 'simple' ? undefined : { forceLegacyRoute: true })

  void activation.catch(error => {
    notifyError(error, `Failed to open profile "${target.profile}"`)
  })
}
