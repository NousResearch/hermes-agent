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
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

/**
 * Which tools each installed package's own code registered, per chat owner
 * (connection + profile) — the authority behind tool media renderers. Read
 * from the owning backend's `plugins.manage list` (`tools`), never from the
 * renderer: a desktop half could otherwise name any tool. Loaded lazily the
 * first time a claimed attachment needs it, refetched when the agent plugin
 * list changes, retried with a bounded backoff after a failure. A backend that
 * predates the field reports nothing, so every claim keeps the built-in player.
 */

export type ToolMediaOwnersEntry =
  | { status: 'loading' | 'error'; request: number }
  | { status: 'ready'; request: number; owners: ReadonlyMap<string, ToolOwnership> }

export const $toolMediaOwners = atom<Readonly<Record<string, ToolMediaOwnersEntry>>>({})

type OwnershipRow = AgentPluginRow & { tools?: string[] }

let requests = 0
const failures = new Map<string, number>()

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

/** The backend profile a session owner names (the profile `plugins.manage` must list). */
function ownerProfile(owner: SessionOwnerScope): null | string {
  if (!owner) {
    return null
  }

  return typeof owner === 'string' ? owner : (owner.targetProfile ?? owner.profile)
}

/** Cache key for the chat `sessionId` belongs to: its exact owner when known, else the active route. */
export function toolMediaOwnerKey(sessionId: null | string): string {
  const owner = knownOwnerForSession(sessionId)

  if (owner) {
    return typeof owner === 'string' ? `profile:${owner}` : `route:${owner.connectionId}|${ownerProfile(owner)}`
  }

  return `active:${activeGatewayConnectionId() ?? ''}|${activeGatewayProfileKey()}|${gatewayActivationEpoch()}`
}

function setEntry(key: string, entry: null | ToolMediaOwnersEntry) {
  const next = { ...$toolMediaOwners.get() }

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

/** Fetch the ownership list for the chat `sessionId` belongs to, unless it is loaded, loading or backing off. */
export function ensureToolMediaOwners(sessionId: null | string): void {
  const key = toolMediaOwnerKey(sessionId)
  const gateway = activeGateway()

  if ($toolMediaOwners.get()[key] || !gateway) {
    return
  }

  const owner = knownOwnerForSession(sessionId)
  const profile = ownerProfile(owner)
  const request = ++requests

  setEntry(key, { request, status: 'loading' })
  // Only the newest request for a key may write it: an invalidation (plugin
  // toggle) mid-flight clears the entry, and the older answer is dropped.
  const stillCurrent = () => $toolMediaOwners.get()[key]?.request === request

  requestForSessionProfile<{ plugins?: OwnershipRow[] }>(
    owner,
    (method, params) => gateway.request(method, params),
    'plugins.manage',
    profile ? { action: 'list', profile } : { action: 'list' }
  )
    .then(result => {
      if (stillCurrent()) {
        failures.delete(key)
        setEntry(key, { owners: ownersFromPluginRows(result?.plugins ?? []), request, status: 'ready' })
      }
    })
    .catch(() => {
      if (stillCurrent()) {
        setEntry(key, { request, status: 'error' })
        scheduleRetry(key, stillCurrent)
      }
    })
}

// A toggle/install/update in Capabilities reloads `$agentPlugins`: drop every
// cached list (orphaning in-flight answers) so the next use re-reads ownership.
$agentPlugins.listen(() => {
  failures.clear()
  $toolMediaOwners.set({})
})

/** The agent package a contribution's plugin is the desktop half of (`plugin:<id>` →
 *  its record's `packageName`); null for standalone desktop plugins and core. */
export function packageOfContributionSource(source: string | undefined): null | string {
  if (!source?.startsWith('plugin:')) {
    return null
  }

  const record = $pluginRecords.get()[source.slice('plugin:'.length)]

  return record?.status === 'loaded' ? desktopPackageName(record) : null
}
