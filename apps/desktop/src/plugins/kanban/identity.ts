/**
 * Assignee-avatar identity: maps a kanban assignee name to the Bot Mode look
 * the Bots roster renders, and resolves the profile's avatar image asset.
 *
 * The roster row comes from the same `GET /profiles` query the board, drawer
 * and orchestration panels already run (one react-query cache entry — no
 * second fetch); the image rides the gateway's `profiles.get_asset` RPC via
 * the plugin host, with the same bounded-cache discipline as the chat
 * thread's agent avatar resolver
 * (apps/desktop/src/components/assistant-ui/thread/user-message.tsx).
 * Free-text assignees and profiles without bot identity resolve to null and
 * stay on the slug-initial chip.
 */

import { host, useQuery } from '@hermes/plugin-sdk'
import { useEffect, useMemo, useState } from 'react'

import { fetchProfiles, profilesKey, useKanbanScope } from './api'
import type { KanbanProfileBot } from './types'

/** The bot identity a kanban assignee name maps to, or null when the name is
 *  not a profile (free-text assignee) or the profile carries no bot identity. */
export function useKanbanBot(name: null | string | undefined): null | KanbanProfileBot {
  const scope = useKanbanScope()

  const { data } = useQuery({
    queryFn: () => fetchProfiles(),
    queryKey: profilesKey(scope),
    // The board refetches on socket events; a longer staleTime here only
    // widens the window the shared cache already covers.
    staleTime: 60_000
  })

  return useMemo(() => data?.profiles.find(profile => profile.name === name)?.bot ?? null, [data, name])
}

// Bounded LRU of resolved avatar images (data URLs — pinning every bot's PNG
// for the session is fine at roster scale, but the cap keeps a long-running
// board honest). Misses expire: an avatar can appear at any moment (bot just
// created, art still uploading), and a permanent negative cache froze the
// fallback until restart in the chat thread's first cut of this same cache.
const AVATAR_IMAGE_CACHE_MAX = 128
const AVATAR_IMAGE_MISS_TTL_MS = 30_000
const avatarImageCache = new Map<string, { at: number; url: null | string }>()
const avatarImageInflight = new Map<string, Promise<null | string>>()

function cachedAvatarImage(name: null | string): null | string {
  if (!name) {
    return null
  }

  return avatarImageCache.get(name.trim().toLowerCase())?.url ?? null
}

/** The profile's avatar image as a data URL, or null. Positive results cache
 *  for the session; misses re-probe after a TTL. */
export async function kanbanAvatarImage(name: null | string): Promise<null | string> {
  const key = name?.trim().toLowerCase() ?? ''

  if (!key) {
    return null
  }

  const hit = avatarImageCache.get(key)

  if (hit) {
    if (hit.url !== null) {
      avatarImageCache.delete(key)
      avatarImageCache.set(key, hit)

      return hit.url
    }

    if (Date.now() - hit.at < AVATAR_IMAGE_MISS_TTL_MS) {
      return null
    }
  }

  const inflight = avatarImageInflight.get(key)

  if (inflight) {
    return inflight
  }

  const run = (async (): Promise<null | string> => {
    try {
      const asset = await host.request<null | { data?: string; found?: boolean }>('profiles.get_asset', {
        asset: 'avatar',
        name: key
      })

      return asset?.found && asset.data ? asset.data : null
    } catch {
      // Older gateway without the profiles.* RPCs, or a transient socket
      // error — the shape/color fallback is always correct.
      return null
    } finally {
      avatarImageInflight.delete(key)
    }
  })()

  avatarImageInflight.set(key, run)
  const url = await run

  if (avatarImageCache.size >= AVATAR_IMAGE_CACHE_MAX) {
    const oldest = avatarImageCache.keys().next().value

    if (oldest !== undefined) {
      avatarImageCache.delete(oldest)
    }
  }

  avatarImageCache.set(key, { at: Date.now(), url })

  return url
}

/** Drop every cached avatar image (positive and negative). Test seam — the
 *  caches are module state, and assertions about the probe need a clean slate. */
export function resetKanbanAvatarImageCache(): void {
  avatarImageCache.clear()
  avatarImageInflight.clear()
}

/** Resolved avatar image for one bot: the legacy inline data URL paints
 *  immediately; the server-side asset is probed only when the backend's
 *  `has_avatar` flag says one exists (no per-render RPC on a maybe). */
export function useKanbanAvatarImage(bot: null | KanbanProfileBot, name: string): null | string {
  const wantImage = Boolean(bot?.has_avatar)
  const [url, setUrl] = useState<null | string>(() => bot?.image ?? cachedAvatarImage(name))

  useEffect(() => {
    if (!wantImage) {
      return
    }

    let live = true

    void kanbanAvatarImage(name).then(resolved => {
      if (live && resolved) {
        setUrl(resolved)
      }
    })

    return () => {
      live = false
    }
  }, [name, wantImage])

  return url
}
