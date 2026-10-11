import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $defaultProfileRoute } from '@/store/default-profile'
import { notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  type AgentProfileRoute,
  captureNewChatSource,
  ensureGatewayAgent,
  ensureGatewayProfile,
  normalizeProfileKey,
  pinLegacyNewChatProfile,
  resolveNewChatOwnerRoute
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
  const project = $projectScope.get()

  if (project !== ALL_PROJECTS && project !== NO_PROJECT_ID) {
    return null
  }

  // An ordinary New Window inherits its opener only for boot. Reusing that
  // seed here would undo a later device/profile selection. Only an explicit
  // "Open profile in new window" makes the peer's launch route a default.
  const profile = !isPeerInstanceWindow() || isProfilePinnedWindow() ? windowProfileOverride() : null
  const saved = profile ? { connectionId: windowConnectionOverride(), profile } : $defaultProfileRoute.get()

  // A persisted default-route preference must not re-home a generic New
  // Session away from the profile that is live in this window (rail
  // selection / new-chat pin): a conflicting saved preference previously
  // won, silently creating the chat in the saved profile. The live
  // selection wins a conflict; with no preference, no live profile, or a
  // preference that merely confirms the live selection, behaviour is
  // unchanged.
  if (!profile && saved) {
    const liveProfile = normalizeProfileKey($newChatProfile.get() || $activeGatewayProfile.get())

    if (liveProfile && normalizeProfileKey(saved.profile) !== liveProfile) {
      return { profile: liveProfile, route: resolveNewChatOwnerRoute(liveProfile) }
    }
  }

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
    pinLegacyNewChatProfile(target.profile)
  }

  const activation = target.route
    ? ensureGatewayAgent(target.route.connectionId, target.profile)
    : ensureGatewayProfile(target.profile, { forceLegacyRoute: true })

  void activation.catch(error => {
    notifyError(error, `Failed to open profile "${target.profile}"`)
  })
}
