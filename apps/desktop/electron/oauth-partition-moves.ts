/**
 * oauth-partition-moves.ts (ForgeGuard fork)
 *
 * Which cookie-flow gateways change cookie jars when the connections registry
 * changes, so their session cookies can be carried across.
 *
 * `resolveOauthPartition` (oauth-partition.ts, #92183) decides a gateway's jar
 * from the registry's CURRENT shape: a non-primary `remote` + `oauth` entry
 * rides `persist:hermes-remote-oauth:conn:<id>`, while the primary, the v1
 * remote and any URL the registry does not know yet ride the legacy shared
 * jar. The decision is right at every instant, but it moves under a gateway
 * whose status changes, and the cookies do not move with it:
 *
 *   - Settings → Connections signs in BEFORE the entry is saved. The URL is
 *     not in the registry yet, so the login window writes the session into
 *     the legacy jar; once saved as a non-primary entry, every read goes to
 *     its own (empty) jar, and switching to it reports "not signed in".
 *   - "Make primary", or applying a connection from the connection dialog,
 *     moves the promoted entry onto the legacy jar and the demoted one off
 *     it, stranding both sessions.
 *
 * This module is the pure half of the repair: diff two registry snapshots
 * (before and after one mutation) and list, per cookie-flow gateway URL, the
 * jar it was on and the jar it is on now. main.ts moves the cookies
 * (oauth-cookie-move.ts). The resolver itself is untouched, so nothing moves
 * at boot and upgrades still never sign anyone out.
 *
 * No `electron` import, so it unit-tests in the electron vitest project.
 */

import { LEGACY_OAUTH_PARTITION, resolveOauthPartition, type ResolveOauthPartitionOptions } from './oauth-partition'

export interface OauthPartitionMove {
  /** The gateway's registered base URL (cookies are read and written for it). */
  url: string
  from: string
  to: string
}

function cookieFlowRemoteUrls(opts: ResolveOauthPartitionOptions | null | undefined): string[] {
  const connections = opts?.registry?.connections

  if (!Array.isArray(connections)) {
    return []
  }

  const urls: string[] = []

  for (const entry of connections) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    const { kind, authMode, url } = entry as Record<string, unknown>

    if (kind !== 'remote' || authMode !== 'oauth' || typeof url !== 'string') {
      continue
    }

    const trimmed = url.trim()

    if (/^https?:\/\//i.test(trimmed)) {
      urls.push(trimmed)
    }
  }

  return urls
}

/**
 * Every cookie-flow gateway (a `remote` entry with `authMode: 'oauth'`) still
 * registered in `after` whose jar differs between `before` and `after`.
 *
 * Only gateways registered AFTER the mutation are considered. A URL that was
 * removed, or edited away, resolves to the shared legacy jar afterwards —
 * moving its session there would hand it to whichever gateway on the same
 * host rides that jar, the leak #92183 exists to prevent. Its old jar is
 * simply left behind, as upstream already does.
 *
 * Moves OUT of the legacy jar sort first: on a primary flip the demoted
 * gateway must leave the shared jar before the promoted one arrives in it, or
 * the promoted session could be skipped as "destination already signed in"
 * when both gateways sit on one host. Never throws.
 */
export function planOauthPartitionMoves(
  before: ResolveOauthPartitionOptions | null | undefined,
  after: ResolveOauthPartitionOptions | null | undefined
): OauthPartitionMove[] {
  try {
    const urls = new Set(cookieFlowRemoteUrls(after))
    const moves: OauthPartitionMove[] = []

    for (const url of urls) {
      const from = resolveOauthPartition(url, before || {})
      const to = resolveOauthPartition(url, after || {})

      if (from !== to) {
        moves.push({ url, from, to })
      }
    }

    return moves.sort((a, b) => {
      const aLegacy = a.from === LEGACY_OAUTH_PARTITION ? 0 : 1
      const bLegacy = b.from === LEGACY_OAUTH_PARTITION ? 0 : 1

      return aLegacy - bLegacy || a.url.localeCompare(b.url)
    })
  } catch {
    return []
  }
}
