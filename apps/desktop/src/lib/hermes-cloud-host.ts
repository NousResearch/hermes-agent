/**
 * The Webapp tab is served BY a Hermes Cloud agent (its own origin is the
 * agent). Cloud then owns every update: the app itself comes from that agent.
 * Kept free of the session store so low-level modules (notifications, the
 * settings manifest) can ask it without an import cycle.
 */

import { isHermesCloudUrl } from '@hermes/shared'

import { browserHostRunsNoAgent, isBrowserHostedDesktop } from '@/lib/platform'

export function servedByHermesCloud(): boolean {
  return typeof window !== 'undefined' && isBrowserHostedDesktop() && isHermesCloudUrl(window.location.origin)
}

/**
 * This tab's bridge signs in to a Hermes Cloud agent and saves it as a connection
 * (`cloud.agentSignIn`). A host that runs no agent does not: its list is the account's agents,
 * each signed in already.
 */
export function signsInToCloudAgents(): boolean {
  return typeof window === 'undefined' || !browserHostRunsNoAgent()
}

/**
 * Settings offers Transfer only where it can finish: it signs in to the target cloud agent. A tab
 * Hermes Cloud serves is a cloud agent itself. Backup is offered everywhere.
 */
export function transferAvailable(): boolean {
  return !servedByHermesCloud() && signsInToCloudAgents()
}
