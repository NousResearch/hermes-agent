import { atom, computed } from 'nanostores'

import { getProfiles } from '@/api/profiles'
import type { DesktopAgentRoster } from '@/global'
import {
  activeGatewayConnectionId,
  activeGatewayProfileKey,
  requestGatewayForAgent,
  type SpawnPriority
} from '@/store/gateway'

import { type AgentPluginRow, isDesktopRelevantPlugin, normalizeAgentPluginRow } from './agent-plugins'

/**
 * Fleet view of agent-plugin updates: every profile on every registered
 * gateway, not just the profile the Plugins tab has selected.
 *
 * The scan reuses the exact per-profile primitive the tab already uses
 * (`plugins.manage list` / `update`), routed per owner. One socket per
 * gateway serves all of that gateway's profiles through the RPC's `profile`
 * param — the same scoping the tab's profile selector relies on — so a scan
 * never spawns one backend per profile. It runs on demand (tab mount, rescan,
 * dialog open), never on a timer.
 */

export interface FleetTarget {
  connectionId: null | string
  connectionLabel: string
  /** Desktop-side profile key (roster identity). */
  profile: string
  /** Profile name on the gateway's host (differs for an SSH `remoteProfile`). */
  targetProfile: string
  /** The profile this window is currently on. */
  current: boolean
  /** Profile whose socket carries requests to this gateway (one socket per gateway). */
  via: string
}

export interface FleetInstall {
  target: FleetTarget
  row: AgentPluginRow
}

export interface FleetSkippedSource {
  connectionId: null | string
  label: string
  /** `idle`: an SSH source never dialed this session — the scan does not open tunnels on its own.
   *  `unreadable`: the gateway answered, but this one profile's list did not. */
  reason: 'idle' | 'offline' | 'sign-in' | 'unreadable'
}

export interface PluginFleetScan {
  /** Installs whose catalog pin moved (`update_available`), every gateway and profile. */
  outdated: FleetInstall[]
  skipped: FleetSkippedSource[]
  scannedAt: number
}

export type FleetScanStatus = 'idle' | 'scanning' | 'ready'

export const $pluginFleet = atom<PluginFleetScan | null>(null)
export const $pluginFleetStatus = atom<FleetScanStatus>('idle')

/** The open fleet-update review, if any: which installs it covers and its title. */
export const $fleetUpdateRequest = atom<null | { installs: FleetInstall[]; title: string }>(null)

export const openFleetUpdate = (installs: FleetInstall[], title: string): void =>
  $fleetUpdateRequest.set({ installs, title })

export const closeFleetUpdate = (): void => $fleetUpdateRequest.set(null)

/** Stable plugin identity across profiles: the catalog entry when there is one. */
export const fleetPluginKey = (row: Pick<AgentPluginRow, 'catalog_name' | 'name'>): string =>
  row.catalog_name || row.name

export const fleetInstallKey = (install: FleetInstall): string =>
  `${install.target.connectionId ?? 'local'}::${install.target.profile}::${install.row.name}`

/** Outdated installs grouped by plugin key, in scan order. */
export const $pluginFleetByPlugin = computed($pluginFleet, scan => {
  const groups = new Map<string, FleetInstall[]>()

  for (const install of scan?.outdated ?? []) {
    const key = fleetPluginKey(install.row)
    groups.set(key, [...(groups.get(key) ?? []), install])
  }

  return groups
})

const STALE_MS = 60_000
const SCAN_CONCURRENCY = 4
// Re-pinning clones the plugin repository and may prepare dependencies; the
// default 30 s RPC deadline expires on healthy updates (same budget as install).
const UPDATE_TIMEOUT_MS = 120_000
// A list is a local disk scan on the backend; anything slower is a dead route.
const LIST_TIMEOUT_MS = 15_000

let inflight: null | Promise<PluginFleetScan | null> = null

interface FleetSource {
  connectionId: null | string
  label: string
  targets: FleetTarget[]
}

async function enumerateSources(): Promise<{ sources: FleetSource[]; skipped: FleetSkippedSource[] }> {
  const activeConnection = activeGatewayConnectionId()
  const activeProfile = activeGatewayProfileKey() || 'default'
  const rosterBridge = window.hermesDesktop?.getAgentRoster
  let roster: DesktopAgentRoster | null = null

  if (rosterBridge) {
    roster = await rosterBridge().catch(() => null)
  }

  if (roster && roster.agents.length > 0) {
    const sources = new Map<string, FleetSource>()
    const skipped: FleetSkippedSource[] = []

    // 'connect-on-demand' is a deliberately undialed source, not a failure: the
    // roster seeds a placeholder profile for it, and asking that placeholder
    // for its plugins would dial a tunnel/backend the user never opened.
    // An unreachable source still lists its last-known profiles (the roster
    // caches them); dialing those would just wait out a dead socket.
    const idle = new Set<string>()

    for (const source of roster.sources) {
      if (source.error === 'connect-on-demand') {
        idle.add(source.connectionId)
        skipped.push({ connectionId: source.connectionId, label: source.label, reason: 'idle' })
      } else if (!source.reachable) {
        idle.add(source.connectionId)
        skipped.push({
          connectionId: source.connectionId,
          label: source.label,
          reason: source.needsSignIn ? 'sign-in' : 'offline'
        })
      }
    }

    for (const agent of roster.agents) {
      const id = agent.connectionId

      if (idle.has(id)) {
        continue
      }

      const isActive = id === (activeConnection ?? 'local')

      // The gateway's own registry label wins over the per-agent copy.
      const label = roster.sources.find(source => source.connectionId === id)?.label || agent.connectionLabel
      const entry = sources.get(id) ?? { connectionId: id, label, targets: [] }

      entry.targets.push({
        connectionId: id,
        connectionLabel: entry.label,
        profile: agent.profile,
        targetProfile: agent.targetProfile || agent.profile,
        current: isActive && agent.profile === activeProfile,
        via: ''
      })
      sources.set(id, entry)
    }

    for (const entry of sources.values()) {
      const isActive = entry.connectionId === (activeConnection ?? 'local')

      // The active gateway rides the window's own socket; another gateway
      // dials one socket, preferring its default profile.
      const via = isActive
        ? activeProfile
        : (entry.targets.find(target => target.profile === 'default') ?? entry.targets[0]).profile

      entry.targets.forEach(target => (target.via = via))
    }

    return { sources: [...sources.values()], skipped }
  }

  // Single-gateway / legacy topology: this window's backend serves every profile.
  const { profiles } = await getProfiles()

  return {
    sources: [
      {
        connectionId: activeConnection,
        label: '',
        targets: profiles.map(profile => ({
          connectionId: activeConnection,
          connectionLabel: '',
          profile: profile.name,
          targetProfile: profile.name,
          current: profile.name === activeProfile,
          via: activeProfile
        }))
      }
    ],
    skipped: []
  }
}

function fleetRequest<T>(
  target: FleetTarget,
  params: Record<string, unknown>,
  timeoutMs?: number,
  spawnPriority: SpawnPriority = 'background'
): Promise<T> {
  return requestGatewayForAgent<T>(
    target.connectionId,
    target.via,
    'plugins.manage',
    { ...params, profile: target.targetProfile },
    timeoutMs,
    undefined,
    { spawnPriority }
  )
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let next = 0

  const worker = async () => {
    while (next < items.length) {
      const index = next++
      results[index] = await fn(items[index])
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))

  return results
}

/** Scan every reachable (gateway, profile) for outdated catalog plugins. */
export function scanPluginFleet(options: { force?: boolean } = {}): Promise<PluginFleetScan | null> {
  const current = $pluginFleet.get()

  if (!options.force && current && Date.now() - current.scannedAt < STALE_MS) {
    return Promise.resolve(current)
  }

  if (inflight) {
    return inflight
  }

  $pluginFleetStatus.set('scanning')

  inflight = (async () => {
    try {
      const { sources, skipped } = await enumerateSources()
      const answered = new Map<FleetSource, number>()
      const unreadable: FleetTarget[] = []
      const jobs = sources.flatMap(source => source.targets.map(target => ({ source, target })))

      const lists = await mapLimit(jobs, SCAN_CONCURRENCY, async ({ source, target }) => {
        try {
          const result = await fleetRequest<{ plugins?: AgentPluginRow[] }>(target, { action: 'list' }, LIST_TIMEOUT_MS)
          answered.set(source, (answered.get(source) ?? 0) + 1)

          return (result?.plugins ?? [])
            .map(normalizeAgentPluginRow)
            .filter(row => row.update_available && isDesktopRelevantPlugin(row))
            .map(row => ({ target, row }))
        } catch {
          unreadable.push(target)

          return []
        }
      })

      // A gateway that answered none of its profiles reads as offline.
      for (const source of sources) {
        if (!answered.get(source)) {
          if (!skipped.some(entry => entry.connectionId === source.connectionId)) {
            skipped.push({ connectionId: source.connectionId, label: source.label, reason: 'offline' })
          }

          continue
        }

        // The gateway answered, so a profile that did not is its own failure:
        // dropping it would quietly undercount what is out of date.
        for (const target of unreadable.filter(entry => source.targets.includes(entry))) {
          skipped.push({
            connectionId: target.connectionId,
            label: source.label ? `${source.label} · ${target.profile}` : target.profile,
            reason: 'unreadable'
          })
        }
      }

      // An older backend ignores the `profile` param and answers every profile
      // with its launch profile's plugins. Two profiles of one gateway sharing
      // one install folder is that signature; updating them would re-pin the
      // same folder N times and tick profiles that were never touched.
      const outdated = lists.flat()
      const folderOwners = new Map<string, Set<string>>()

      for (const { target, row } of outdated) {
        if (row.install_dir) {
          const folder = `${target.connectionId ?? 'local'}::${row.install_dir}`
          folderOwners.set(folder, (folderOwners.get(folder) ?? new Set()).add(target.profile))
        }
      }

      const unscoped = (install: FleetInstall) =>
        Boolean(install.row.install_dir) &&
        (folderOwners.get(`${install.target.connectionId ?? 'local'}::${install.row.install_dir}`)?.size ?? 0) > 1

      for (const install of outdated.filter(unscoped)) {
        skipped.push({
          connectionId: install.target.connectionId,
          label: install.target.connectionLabel
            ? `${install.target.connectionLabel} · ${install.target.profile}`
            : install.target.profile,
          reason: 'unreadable'
        })
      }

      const scan: PluginFleetScan = {
        outdated: outdated.filter(install => !unscoped(install)),
        skipped,
        scannedAt: Date.now()
      }

      $pluginFleet.set(scan)

      return scan
    } catch (error) {
      console.warn('[plugin-fleet] scan failed', error)

      return $pluginFleet.get()
    } finally {
      $pluginFleetStatus.set('ready')
      inflight = null
    }
  })()

  return inflight
}

export type FleetUpdateOutcome =
  | { kind: 'applied' | 'unchanged' }
  | { kind: 'failed'; error: string }
  | { kind: 'consent'; sha: string; deltaLines: string[] }

/** Re-pin one install on its own gateway/profile (backend `plugins.manage update`). */
export async function updateFleetInstall(
  install: FleetInstall,
  acceptCapabilities = false
): Promise<FleetUpdateOutcome> {
  const { target, row } = install

  try {
    const result = await fleetRequest<{
      ok?: boolean
      unchanged?: boolean
      consent_required?: boolean
      sha?: string
      delta_lines?: string[]
      error?: string
    }>(
      target,
      { action: 'update', name: row.name, ...(acceptCapabilities ? { accept_capabilities: true } : {}) },
      UPDATE_TIMEOUT_MS,
      'foreground'
    )

    if (result?.consent_required) {
      return { kind: 'consent', sha: (result.sha ?? '').slice(0, 8), deltaLines: result.delta_lines ?? [] }
    }

    if (!result?.ok) {
      return { kind: 'failed', error: result?.error || 'Update failed' }
    }

    return { kind: result.unchanged ? 'unchanged' : 'applied' }
  } catch (error) {
    return { kind: 'failed', error: error instanceof Error ? error.message : String(error) }
  }
}

/** Drop installs that no longer need an update from the cached scan. */
export function settleFleetInstalls(keys: Set<string>): void {
  const scan = $pluginFleet.get()

  if (!scan) {
    return
  }

  $pluginFleet.set({ ...scan, outdated: scan.outdated.filter(install => !keys.has(fleetInstallKey(install))) })
}

/** @internal */
export function _resetPluginFleetForTests(): void {
  $pluginFleet.set(null)
  $pluginFleetStatus.set('idle')
  $fleetUpdateRequest.set(null)
  inflight = null
}
