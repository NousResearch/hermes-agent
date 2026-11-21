/**
 * Which session this window is presenting, and the live connection a submit for it would use.
 *
 * Read-only by construction: it never dials, resumes, creates or binds anything. It resolves the
 * same pair a submit does: `use-prompt-actions/submit.ts` checks the routed stored session against
 * the bound runtime before it sends, and `submission-destination.ts` captures the owner with
 * `knownOwnerForSession`. An exact route owner is sent on directly; any other owner is handed to
 * the window's session dispatcher, which decides the owner again (`sessionRpcOwner`, where the
 * runtime's own events can name another connection), so that decision is the one used here. It
 * answers "nothing in view" wherever the composer would first have to resume or create one.
 */

import { registryBackendScopeKey } from '@hermes/shared'

import { sessionRpcOwner } from '@/app/contrib/session-rpc-dispatcher'
import { routeSessionId } from '@/app/routes'
import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, $paneVisible, isPaneVisible } from '@/components/pane-shell/tree/store'

import { $gateway, gatewayRouteState } from './gateway'
import { $primaryGatewayRoute } from './gateway-primary-route'
import { isPrimaryRider } from './gateway-shared-route'
import type { PresenceChannel, PresenceView } from './presence'
import { $activeSessionId, $selectedStoredSessionId } from './session'
import { isSessionOwnerRoute, type SessionOwnerScope, sessionRpcNeedsProfileRoute } from './session-request-router'
import { $sessionStates, knownOwnerForSession } from './session-states'

/** The runtime bound to the chat on this window's route, or null when a submit would have to resume
 *  or create one first (a draft, another page, a stale or cross-wired binding). */
function boundRuntimeInView(): null | { runtimeId: string; storedId: string } {
  // Desktop mounts HashRouter: the route on screen is `location.hash`.
  const storedId = routeSessionId(window.location.hash.replace(/^#/, ''))
  const runtimeId = $activeSessionId.get()

  if (!storedId || !runtimeId || $selectedStoredSessionId.get() !== storedId) {
    return null
  }

  // Ownership proof from the runtime's own state, as submit's cache check requires.
  return $sessionStates.get()[runtimeId]?.storedSessionId === storedId ? { runtimeId, storedId } : null
}

/** The open socket a request for this owner would ride, or `undefined` when the backend could not
 *  tell this session from another one: a profile riding the shared primary socket is told apart only
 *  by a `profile` param, which a report does not carry. */
function ownerChannel(owner: SessionOwnerScope): PresenceChannel | null | undefined {
  // An unknown owner is the composer's ambient dispatcher.
  if (!sessionRpcNeedsProfileRoute(owner)) {
    return $gateway.get()
  }

  const connectionId = isSessionOwnerRoute(owner) ? owner.connectionId : null
  const profile = isSessionOwnerRoute(owner) ? owner.profile : String(owner)
  const route = gatewayRouteState(connectionId, profile)
  const primary = { connectionId: $primaryGatewayRoute.get().connectionId, gateway: route.gateway }

  if (isPrimaryRider(registryBackendScopeKey(connectionId, profile), primary)) {
    return undefined
  }

  return route.gateway
}

/** The pane that presents the chat on this window's route. */
const MAIN_CHAT_PANE = 'workspace'

/** Whether the chat on this window's route is the tab in front. Another tab of its zone (a second
 *  conversation, a preview, a tool) can cover it, and its zone can be minimized or collapsed, while
 *  the route and the main session stay the same. Without a layout that places the chat, nothing can. */
export function mainChatShowing(): boolean {
  const tree = $layoutTree.get()

  return !tree || !findGroupOfPane(tree, MAIN_CHAT_PANE) || isPaneVisible(MAIN_CHAT_PANE)
}

/** The stores `mainChatShowing` reads, for a listener. */
export const mainChatShowingStores = () => [$layoutTree, $paneVisible(MAIN_CHAT_PANE)] as const

export function presenceView(): PresenceView {
  const bound = boundRuntimeInView()

  if (!bound) {
    return { channel: $gateway.get(), kind: 'none' }
  }

  // Captured with the stored id first, exactly as the composer captures its destination; only an
  // exact route skips the dispatcher's own decision.
  const captured = knownOwnerForSession(bound.storedId)
  const owner = isSessionOwnerRoute(captured) ? captured : sessionRpcOwner(bound.storedId, bound.runtimeId)
  const channel = ownerChannel(owner)

  return channel === undefined ? { kind: 'unroutable' } : { channel, kind: 'session', sessionId: bound.runtimeId }
}
