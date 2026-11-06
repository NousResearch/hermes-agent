import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $defaultProfileRoute } from '@/store/default-profile'
import { notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  type AgentProfileRoute,
  captureNewChatSource,
  ensureGatewayAgent,
  ensureGatewayProfile,
  pinLegacyNewChatProfile
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
  // An explicit profile window stays pinned. Otherwise a newer captured
  // gateway/profile selection outranks the app-wide default.
  const profile = !isPeerInstanceWindow() || isProfilePinnedWindow() ? windowProfileOverride() : null

  if (profile) {
    const saved = { connectionId: windowConnectionOverride(), profile }

    return {
      profile: saved.profile,
      route: saved.connectionId === null ? null : { connectionId: saved.connectionId, profile: saved.profile }
    }
  }

  // A new-chat owner captured by a profile/gateway selection is newer than
  // the app-wide default. The default still routes generic drafts when there
  // is no newer owner intent (including the initial boot restore).
  const saved = $defaultProfileRoute.get()
  const selectedProfile = $newChatProfile.get()
  const selectedConnectionId = $newChatConnectionId.get()
  const hasSelectedOwner = selectedProfile !== null || selectedConnectionId !== null

  if (hasSelectedOwner) {
    const selected = {
      profile: selectedProfile ?? $activeGatewayProfile.get(),
      route:
        selectedConnectionId === null
          ? null
          : { connectionId: selectedConnectionId, profile: selectedProfile ?? $activeGatewayProfile.get() }
    }

    if (!saved || saved.connectionId !== selected.route?.connectionId || saved.profile !== selected.profile) {
      return selected
    }
  }

  if (saved) {
    return {
      profile: saved.profile,
      route: saved.connectionId === null ? null : { connectionId: saved.connectionId, profile: saved.profile }
    }
  }

  return null
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
