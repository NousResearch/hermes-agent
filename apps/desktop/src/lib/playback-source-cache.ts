// Resolved audio/video stream URLs, kept across remounts. Transcript rows
// remount routinely (a finished turn re-renders its markdown, the render
// budget recycles older rows, a reconnect re-resumes the chat). In the Webapp
// every mount minted a new stream ticket, so the browser fetched the clip
// again under a new URL and the clip's remembered position (keyed by URL)
// missed: over a slow link the clip went blank and restarted.
//
// Only URL strings are kept, never bytes, and for well under a ticket's life
// (the server's stream tickets last an hour). A URL that fails anyway is
// re-resolved by its player, which replaces the entry.

import { LruCache } from '@/lib/lru-cache'

const PLAYBACK_TTL_MS = 20 * 60_000

const playback = new LruCache<string, { at: number; src: string }>(128)

export function cachedPlaybackSrc(key: string, now = Date.now()): string | undefined {
  const entry = playback.get(key)

  if (entry && now - entry.at < PLAYBACK_TTL_MS) {
    return entry.src
  }

  playback.delete(key)

  return undefined
}

export function rememberPlaybackSrc(key: string, src: string, now = Date.now()): void {
  // A blob URL is revoked by the element that minted it; never hand one to another mount.
  if (!src || src.startsWith('blob:')) {
    playback.delete(key)

    return
  }

  playback.set(key, { at: now, src })
}

export function forgetPlaybackSrc(key: string): void {
  playback.delete(key)
}

/** Process-memory only; tests reset it between cases that reuse one path. */
export function clearPlaybackSourceCache(): void {
  for (const key of [...playback.keys()]) {
    playback.delete(key)
  }
}
