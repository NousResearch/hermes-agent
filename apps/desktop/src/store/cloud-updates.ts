/**
 * Hermes Cloud owns updates for its agents: it replaces the agent's image, so
 * no in-app updater may be offered there. Two shapes reach a Cloud agent:
 *
 * - the Webapp served BY a Cloud agent (the tab's own origin is the agent):
 *   every update target is Cloud's, because the app itself comes from it;
 * - native Desktop CONNECTED to a Cloud agent: only the backend target is;
 *   the Desktop app on this machine still updates itself.
 *
 * A Cloud agent is a `*.agents.nousresearch.com` host (staging:
 * `*.agents.staging-nousresearch.com`), the public URL the portal gives it.
 */

import { computed } from 'nanostores'

import type { HermesConnection } from '@/global'
import { isBrowserHostedDesktop } from '@/lib/platform'
import type { UpdateTarget } from '@/lib/update-copy'
import { $connection } from '@/store/session'

const CLOUD_HOST_SUFFIXES = ['.agents.nousresearch.com', '.agents.staging-nousresearch.com']

export function isHermesCloudUrl(url: null | string | undefined): boolean {
  if (!url) {
    return false
  }

  try {
    const host = new URL(url).hostname.toLowerCase()

    return CLOUD_HOST_SUFFIXES.some(suffix => host.endsWith(suffix))
  } catch {
    return false
  }
}

function servedByHermesCloud(): boolean {
  return typeof window !== 'undefined' && isBrowserHostedDesktop() && isHermesCloudUrl(window.location.origin)
}

function connectedToHermesCloud(connection: HermesConnection | null): boolean {
  return connection?.mode === 'remote' && (connection.remoteKind === 'cloud' || isHermesCloudUrl(connection.baseUrl))
}

/** Whether Hermes Cloud owns updates for `target` under `connection`. */
export function cloudOwnsUpdates(target: UpdateTarget, connection: HermesConnection | null = $connection.get()): boolean {
  return servedByHermesCloud() || (target === 'backend' && connectedToHermesCloud(connection))
}

/** Per target: true while every update control for it must stay hidden. */
export const $cloudOwnsUpdates = computed($connection, connection => ({
  backend: cloudOwnsUpdates('backend', connection),
  client: cloudOwnsUpdates('client', connection)
}))
