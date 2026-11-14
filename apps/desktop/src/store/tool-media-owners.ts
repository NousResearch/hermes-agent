import { atom } from 'nanostores'

import { agentPackageName, desktopPackageName } from '@/app/capabilities/plugins/plugin-packages'
import { $pluginRecords } from '@/contrib/plugins-store'
import type { ToolOwnership } from '@/lib/tool-media-renderers'
import { $agentPlugins, type AgentPluginRow } from '@/store/agent-plugins'
import {
  activeGateway,
  activeGatewayConnectionId,
  activeGatewayProfileKey,
  gatewayActivationEpoch
} from '@/store/gateway'
import { isSessionOwnerRoute, requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

/**
 * Which tools each installed package's own code registered, per chat owner
 * (connection + profile) — the authority behind tool media renderers. Read
 * from the owning backend's `plugins.manage list` (`tools`), never from the
 * renderer: a desktop half could otherwise name any tool. Loaded lazily the
 * first time a claimed attachment needs it, retried with a bounded backoff
 * after a failure. A plugin-list change re-reads every list while the previous
 * answer keeps drawing (no widget remounts), and a claimed tool nobody owns
 * yet (its MCP server still connecting) re-reads on a bounded backoff. A
 * backend that predates the field reports nothing, so every claim keeps the
 * built-in player.
 */

export type ToolMediaOwnersEntry =
  | { status: 'loading' | 'error'; request: number }
  | {
      status: 'ready'
      request: number
      owners: ReadonlyMap<string, ToolOwnership>
      /** `due`: re-read on next use (keeps `owners` meanwhile); `pending`: re-reading. */
      refresh?: 'due' | 'pending'
    }

export const $toolMediaOwners = atom<Readonly<Record<string, ToolMediaOwnersEntry>>>({})

type OwnershipRow = AgentPluginRow & { tools?: string[] }

let requests = 0
const failures = new Map<string, number>()
// Re-reads spent on a claimed-but-unowned tool, per key, and the pending timer.
const revalidations = new Map<string, { attempts: number; timer?: ReturnType<typeof setTimeout> }>()

/** How many times a claimed tool with no owner re-reads ownership (1 s, then
 *  5 s doubling to 60 s: about two and a half minutes in all) before it stays built-in. */
export const MAX_OWNERSHIP_REVALIDATIONS = 6

export function ownersFromPluginRows(rows: readonly OwnershipRow[]): Map<string, ToolOwnership> {
  const owners = new Map<string, ToolOwnership>()

  for (const row of rows) {
    const tools = Array.isArray(row.tools) ? row.tools.filter(name => typeof name === 'string' && name) : []

    if (tools.length > 0) {
      owners.set(agentPackageName(row), { tools: new Set(tools) })
    }
  }

  return owners
}

function sameOwners(a: ReadonlyMap<string, ToolOwnership>, b: ReadonlyMap<string, ToolOwnership>): boolean {
  if (a.size !== b.size) {
    return false
  }

  for (const [pkg, owned] of a) {
    const other = b.get(pkg)

    if (!other || other.tools.size !== owned.tools.size || [...owned.tools].some(tool => !other.tools.has(tool))) {
      return false
    }
  }

  return true
}

/** The backend profile a session owner names (the profile `plugins.manage` must list). */
function ownerProfile(owner: SessionOwnerScope): null | string {
  if (!owner) {
    return null
  }

  return typeof owner === 'string' ? owner : (owner.targetProfile ?? owner.profile)
}

const ACTIVE_PREFIX = 'active:'

/** Cache key for the chat `sessionId` belongs to: its exact owner when known, else the active route. */
export function toolMediaOwnerKey(sessionId: null | string): string {
  const owner = knownOwnerForSession(sessionId)

  if (owner) {
    return typeof owner === 'string' ? `profile:${owner}` : `route:${owner.connectionId}|${ownerProfile(owner)}`
  }

  return `${ACTIVE_PREFIX}${activeGatewayConnectionId() ?? ''}|${activeGatewayProfileKey()}|${gatewayActivationEpoch()}`
}

function forgetKey(key: string) {
  failures.delete(key)
  clearTimeout(revalidations.get(key)?.timer)
  revalidations.delete(key)
}

function setEntry(key: string, entry: null | ToolMediaOwnersEntry) {
  const next = { ...$toolMediaOwners.get() }
  // An `active:` key names one activation of the route; older ones can never be read again.
  const epoch = `|${gatewayActivationEpoch()}`

  for (const stale of Object.keys(next)) {
    if (stale.startsWith(ACTIVE_PREFIX) && !stale.endsWith(epoch)) {
      delete next[stale]
      forgetKey(stale)
    }
  }

  if (entry) {
    next[key] = entry
  } else {
    delete next[key]
  }

  $toolMediaOwners.set(next)
}

function scheduleRetry(key: string, stillCurrent: () => boolean) {
  const attempt = (failures.get(key) ?? 0) + 1

  failures.set(key, attempt)
  // Bounded backoff (2 s doubling to 60 s): clearing the entry lets the next
  // render of a claimed attachment fetch again once the connection is back.
  setTimeout(
    () => {
      if (stillCurrent()) {
        setEntry(key, null)
      }
    },
    Math.min(60_000, 2_000 * 2 ** (attempt - 1))
  )
}

/** Fetch the ownership list for the chat `sessionId` belongs to, unless it is
 *  loaded (and not due a re-read), loading or backing off. */
export function ensureToolMediaOwners(sessionId: null | string): void {
  const key = toolMediaOwnerKey(sessionId)
  const existing = $toolMediaOwners.get()[key]
  const owner = knownOwnerForSession(sessionId)
  const gateway = activeGateway()

  if (existing && !(existing.status === 'ready' && existing.refresh === 'due')) {
    return
  }

  // A routed owner names its own connection; everything else needs the active one.
  if (!gateway && !isSessionOwnerRoute(owner)) {
    return
  }

  const profile = ownerProfile(owner)
  const request = ++requests
  const previous = existing?.status === 'ready' ? existing.owners : null

  // A re-read keeps drawing with the previous answer until the new one lands.
  setEntry(
    key,
    previous ? { owners: previous, refresh: 'pending', request, status: 'ready' } : { request, status: 'loading' }
  )
  // Only the newest request for a key may write it: an invalidation (plugin
  // toggle) mid-flight re-stamps the entry, and the older answer is dropped.
  const stillCurrent = () => $toolMediaOwners.get()[key]?.request === request

  requestForSessionProfile<{ plugins?: OwnershipRow[] }>(
    owner,
    (method, params) => (gateway ? gateway.request(method, params) : Promise.reject(new Error('No active gateway'))),
    'plugins.manage',
    profile ? { action: 'list', profile } : { action: 'list' }
  )
    .then(result => {
      if (stillCurrent()) {
        const owners = ownersFromPluginRows(result?.plugins ?? [])
        const changed = !previous || !sameOwners(previous, owners)

        failures.delete(key)

        if (changed) {
          revalidations.delete(key)
        }

        setEntry(key, { owners: previous && !changed ? previous : owners, request, status: 'ready' })
      }
    })
    .catch(() => {
      if (stillCurrent()) {
        setEntry(key, { request, status: 'error' })
        scheduleRetry(key, stillCurrent)
      }
    })
}

/** A claimed tool has no owner in the chat's loaded answer: its package's MCP
 *  server may not be connected yet (or was lazy when the list was read). Mark
 *  the answer due a re-read after a bounded backoff; the next use fetches it
 *  while the current answer keeps drawing. */
export function revalidateToolMediaOwners(sessionId: null | string): void {
  const key = toolMediaOwnerKey(sessionId)
  const entry = $toolMediaOwners.get()[key]
  const state = revalidations.get(key) ?? { attempts: 0 }

  if (entry?.status !== 'ready' || entry.refresh || state.timer || state.attempts >= MAX_OWNERSHIP_REVALIDATIONS) {
    return
  }

  const delay = state.attempts === 0 ? 1_000 : Math.min(60_000, 5_000 * 2 ** (state.attempts - 1))
  const request = entry.request

  state.attempts += 1
  state.timer = setTimeout(() => {
    state.timer = undefined
    const current = $toolMediaOwners.get()[key]

    if (current?.status === 'ready' && current.request === request && !current.refresh) {
      setEntry(key, { ...current, refresh: 'due' })
    }
  }, delay)
  revalidations.set(key, state)
}

/** Every loaded list is due a re-read: it keeps drawing until the answer lands
 *  (widgets do not remount), and anything in flight is orphaned by a new stamp.
 *  Called when what a package registered can have changed: the plugin list
 *  reloads (toggle/install/update) or MCP reconnects (`reload.mcp`). */
export function refreshToolMediaOwners(): void {
  failures.clear()

  for (const key of [...revalidations.keys()]) {
    forgetKey(key)
  }

  const next: Record<string, ToolMediaOwnersEntry> = {}

  for (const [key, entry] of Object.entries($toolMediaOwners.get())) {
    if (entry.status === 'ready') {
      next[key] = { owners: entry.owners, refresh: 'due', request: ++requests, status: 'ready' }
    }
  }

  $toolMediaOwners.set(next)
}

$agentPlugins.listen(refreshToolMediaOwners)

/** The agent package a contribution's plugin is the desktop half of (`plugin:<id>` →
 *  its record's `packageName`); null for standalone desktop plugins and core. */
export function packageOfContributionSource(source: string | undefined): null | string {
  if (!source?.startsWith('plugin:')) {
    return null
  }

  const record = $pluginRecords.get()[source.slice('plugin:'.length)]

  return record?.status === 'loaded' ? desktopPackageName(record) : null
}
