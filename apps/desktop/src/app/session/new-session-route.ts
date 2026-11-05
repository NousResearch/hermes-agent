import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $defaultProfileRoute } from '@/store/default-profile'
import { activeGatewayConnectionId } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
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

type NewSessionTarget = { profile: string; route: AgentProfileRoute | null }

/** Hermes' root profile (`~/.hermes`): what "default" means with no saved
 * Desktop preference. */
const ROOT_PROFILE = 'default'

/** Only generic New Session actions consult this preference. Explicit profile,
 * agent, project and existing-session actions keep their captured owner. */
export function defaultNewSessionTarget(): NewSessionTarget | null {
  if (inProjectScope()) {
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

function inProjectScope(): boolean {
  const project = $projectScope.get()

  return project !== ALL_PROJECTS && project !== NO_PROJECT_ID
}

/** The root profile on the source the user is looking at (legacy door when
 * the primary has no registry identity). */
function rootProfileTarget(): NewSessionTarget {
  const connectionId = activeGatewayConnectionId()

  return { profile: ROOT_PROFILE, route: connectionId ? { connectionId, profile: ROOT_PROFILE } : null }
}

/** The generic "New session" doors (sidebar row, ⌘N): a plain session, never
 * the owner an earlier action pinned. A bot chat pins its exact route
 * (`pinNewChatRoute`) and selecting another bot does not unpin it, so a door
 * that kept the pin opened the PREVIOUS bot's session. Drops the whole intent
 * (profile, route, source), then targets the default-profile preference, else
 * the root `default` profile: the ambient profile is whichever bot or
 * workspace was selected last, which the user never chose as a default. A
 * project scope keeps its own owner. */
export function prepareGenericNewSession(): void {
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  captureNewChatSource()
  applyNewSessionTarget(defaultNewSessionTarget() ?? (inProjectScope() ? null : rootProfileTarget()))
}

export function prepareDefaultNewSession(): void {
  applyNewSessionTarget(defaultNewSessionTarget())
}

function applyNewSessionTarget(target: NewSessionTarget | null): void {
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
