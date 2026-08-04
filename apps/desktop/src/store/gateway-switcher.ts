import { atom } from 'nanostores'

import type { DesktopCloudAgent, DesktopConnectionConfig } from '@/global'
import { resetGatewayForProfile } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'

export interface CloudAgentTarget {
  agent: DesktopCloudAgent
  org: string | null
}

/** A directly-switchable connection kind (Cloud agents switch via their own action). */
export type GatewayModeTarget = 'local' | 'remote' | 'ssh'

export const $cloudAgentTargets = atom<CloudAgentTarget[]>([])
export const $cloudAgentTargetsLoading = atom(false)
export const $cloudAgentTargetsError = atom<unknown | null>(null)
export const $starredCloudAgentIds = atom<string[]>([])
export const $cloudAgentSwitching = atom<string | null>(null)
export const $gatewayModeSwitching = atom<GatewayModeTarget | null>(null)
// The GLOBAL connection config snapshot (scope null) — tells the switcher which
// local/remote/SSH targets are configured and therefore offerable.
export const $gatewayConnectionConfig = atom<DesktopConnectionConfig | null>(null)

// The 'default' profile has no per-profile connection override (Settings →
// Gateway offers no scope chip for it), so its switches must target the GLOBAL
// connection. A 'default'-scoped write would create an override nothing in the
// UI can see or clear — with override-over-global precedence, that wedged the
// desktop on the chosen gateway ("can't switch back to local").
function connectionScope(profile: string): string | undefined {
  return profile === 'default' ? undefined : profile
}

function agentOrg(org: { id: string; slug: string | null } | null | undefined): string | null {
  return org ? (org.slug ?? org.id) : null
}

function dedupeTargets(targets: CloudAgentTarget[]): CloudAgentTarget[] {
  const ids = new Set<string>()

  return targets.filter(target => {
    if (ids.has(target.agent.id)) {
      return false
    }

    ids.add(target.agent.id)

    return true
  })
}

export async function refreshCloudAgentStars(): Promise<string[]> {
  const ids = await window.hermesDesktop.cloud.starredAgents()
  $starredCloudAgentIds.set(ids.ids)

  return ids.ids
}

export async function refreshGatewayConnectionConfig(): Promise<DesktopConnectionConfig> {
  const config = await window.hermesDesktop.getConnectionConfig(null)
  $gatewayConnectionConfig.set(config)

  return config
}

/** Discover every visible Cloud agent. Multi-org accounts are expanded into one list. */
export async function refreshCloudAgentTargets(): Promise<CloudAgentTarget[]> {
  const cloud = window.hermesDesktop.cloud
  $cloudAgentTargetsLoading.set(true)

  try {
    const status = await cloud.status()

    if (!status.signedIn) {
      $cloudAgentTargets.set([])
      $cloudAgentTargetsError.set(null)

      return []
    }

    const initial = await cloud.discover()
    let targets: CloudAgentTarget[]

    if ('needsOrgSelection' in initial && initial.needsOrgSelection) {
      const byOrg = await Promise.all(
        initial.orgs.map(async org => {
          const result = await cloud.discover(agentOrg(org) ?? undefined)

          return 'agents' in result ? result.agents.map(agent => ({ agent, org: agentOrg(org) })) : []
        })
      )

      targets = byOrg.flat()
    } else {
      targets = initial.agents.map(agent => ({ agent, org: agentOrg(initial.org) }))
    }

    const next = dedupeTargets(targets)
    $cloudAgentTargets.set(next)
    $cloudAgentTargetsError.set(null)

    return next
  } catch (error) {
    $cloudAgentTargetsError.set(error)
    $cloudAgentTargets.set([])
    notifyError(error, 'Could not load Hermes Cloud agents')
    throw error
  } finally {
    $cloudAgentTargetsLoading.set(false)
  }
}

export async function refreshGatewaySwitcher(): Promise<void> {
  // Each source refreshes independently: a signed-out portal (Cloud discovery
  // rejecting) must not block the local/remote/SSH targets, and vice versa.
  await Promise.all([
    refreshGatewayConnectionConfig().catch(() => undefined),
    refreshCloudAgentStars().catch(() => undefined),
    refreshCloudAgentTargets().catch(() => undefined)
  ])
}

export async function setCloudAgentStarred(id: string, starred: boolean): Promise<string[]> {
  try {
    const result = await window.hermesDesktop.cloud.setAgentStarred(id, starred)
    $starredCloudAgentIds.set(result.ids)

    return result.ids
  } catch (error) {
    notifyError(error, 'Could not update starred Cloud agents')
    throw error
  }
}

let switchQueue: Promise<void> | null = null

// Serialize every gateway switch (cloud agents AND mode switches) through one
// queue so repeated clicks can't race the active connection pointer.
function queueSwitch(run: () => Promise<void>): Promise<void> {
  const previous = switchQueue
  const operation = (previous ? previous.catch(() => undefined) : Promise.resolve()).then(run)
  switchQueue = operation.catch(() => undefined)

  return operation
}

/**
 * Reuses the ordinary cloud connection apply/rehome path. A Cloud agent is a
 * target, not a Hermes profile, so it is applied to the profile currently live
 * in the desktop and then reactivated through the normal pool lifecycle.
 */
export function switchToCloudAgent(target: CloudAgentTarget): Promise<void> {
  const profile = normalizeProfileKey($activeGatewayProfile.get())

  return queueSwitch(async () => {
    $cloudAgentSwitching.set(target.agent.id)

    try {
      if (!target.agent.dashboardUrl) {
        throw new Error('This Hermes Cloud agent is still provisioning.')
      }

      const signIn = await window.hermesDesktop.cloud.agentSignIn(target.agent.dashboardUrl)

      if (!signIn.connected) {
        throw new Error('Could not establish a Hermes Cloud session for this agent.')
      }

      await window.hermesDesktop.applyConnectionConfig({
        cloudOrg: target.org ?? undefined,
        mode: 'cloud',
        profile: connectionScope(profile),
        remoteAuthMode: 'oauth',
        remoteUrl: target.agent.dashboardUrl
      })
      void refreshGatewayConnectionConfig().catch(() => undefined)
      resetGatewayForProfile(profile)
      await ensureGatewayProfile(profile)
    } catch (error) {
      notifyError(error, 'Could not switch Hermes Cloud agent')
      throw error
    } finally {
      if ($cloudAgentSwitching.get() === target.agent.id) {
        $cloudAgentSwitching.set(null)
      }
    }
  })
}

/**
 * Switch the desktop to one of its configured non-cloud targets. 'local'
 * applies to the active profile scope (a named profile clears its override and
 * inherits the global connection); 'remote' and 'ssh' re-adopt the saved
 * GLOBAL target (main-process snapshot), so they always apply globally. The
 * primary rehome itself is event-driven (use-gateway-boot listens for the
 * connection-applied broadcast from the apply IPC).
 */
export function switchToGatewayMode(mode: GatewayModeTarget): Promise<void> {
  const profile = normalizeProfileKey($activeGatewayProfile.get())

  return queueSwitch(async () => {
    $gatewayModeSwitching.set(mode)

    try {
      await window.hermesDesktop.applyConnectionConfig({
        mode,
        profile: mode === 'local' ? connectionScope(profile) : undefined
      })
      void refreshGatewayConnectionConfig().catch(() => undefined)
      resetGatewayForProfile(profile)
      await ensureGatewayProfile(profile)
    } catch (error) {
      notifyError(error, 'Could not switch gateway')
      throw error
    } finally {
      if ($gatewayModeSwitching.get() === mode) {
        $gatewayModeSwitching.set(null)
      }
    }
  })
}

export function cloudAgentsStarredFirst(targets: CloudAgentTarget[], starredIds: string[]): CloudAgentTarget[] {
  const starred = new Set(starredIds)

  return [...targets].sort((left, right) => {
    const leftStarred = starred.has(left.agent.id)
    const rightStarred = starred.has(right.agent.id)

    if (leftStarred !== rightStarred) {
      return leftStarred ? -1 : 1
    }

    return left.agent.name.localeCompare(right.agent.name)
  })
}

function sameGatewayUrl(left: string | null | undefined, right: string | null | undefined): boolean {
  const normalize = (value: string) => value.trim().replace(/\/+$/, '').toLowerCase()

  return Boolean(left && right && normalize(left) === normalize(right))
}

// Match by URL alone, NOT by the connection's recorded kind: a cloud instance
// connected through an older flow is stored as mode 'remote' (kind 'url'), and
// the truthful row to highlight is still the agent the desktop is talking to.
export function cloudAgentIsActive(target: CloudAgentTarget, baseUrl: string | undefined): boolean {
  return sameGatewayUrl(baseUrl, target.agent.dashboardUrl)
}

/**
 * The direct-remote target to offer in the switcher, or '' when there is none
 * worth showing. A saved target whose URL matches a discovered Cloud agent is
 * a phantom — a cloud connection recorded as mode 'remote' by flows predating
 * cloud provenance, not a user-owned gateway. Selecting it would reconnect to
 * that same agent, so the Cloud row stands for it instead.
 */
export function offerableRemoteUrl(config: DesktopConnectionConfig | null, targets: CloudAgentTarget[]): string {
  const url = config?.savedRemoteUrl || ''

  if (!url) {
    return ''
  }

  return targets.some(target => sameGatewayUrl(url, target.agent.dashboardUrl)) ? '' : url
}
