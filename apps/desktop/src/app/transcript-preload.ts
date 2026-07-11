/**
 * Transcript preloader (startup-latency follow-up; Ace 2026-07-11).
 *
 * After boot settles (live list loaded, gateway open), background-fetch the
 * transcripts of PINNED + VISIBLE sidebar sessions into the render cache — so
 * switching to any session you can see paints instantly on the next launch
 * (and warms this launch's switch too, since the rows land in the same cache
 * the boot path reads).
 *
 * Design constraints:
 *  - GENTLE: strictly sequential fetches with a pause between each; the whole
 *    point of the perf work was to stop hammering the gateway at boot. Preload
 *    starts only AFTER the live reconcile and yields between sessions.
 *  - BOUNDED: at most MAX_PRELOAD sessions per launch (pinned first, then the
 *    top of the visible list). The render cache's own LRU cap (200 files)
 *    bounds disk.
 *  - FAIL-OPEN: any error skips that session; preload never surfaces to the UI.
 *  - FRESHNESS-AWARE: skips sessions whose transcript is already cached from
 *    this gateway unless the session has newer activity than the cached copy.
 */

import { getSessionMessages } from '@/hermes'
import { toChatMessages } from '@/lib/chat-messages'
import type { SessionInfo } from '@/types/hermes'

import { pushTranscriptToRenderCache } from './render-cache-hydration'

export const MAX_PRELOAD = 12
export const PRELOAD_GAP_MS = 750

export interface PreloadDeps {
  gatewayUrl: string | null
  /** Pinned first, then visible order. */
  sessions: SessionInfo[]
  /** Stored-session ids that already have a cached transcript at least as fresh as the row. */
  freshCached?: Set<string>
  fetchMessages?: typeof getSessionMessages
  push?: typeof pushTranscriptToRenderCache
  sleep?: (ms: number) => Promise<void>
  /** Abort signal: return true to stop (e.g. app going down / gateway lost). */
  shouldStop?: () => boolean
}

/** Order the preload set: pinned first (all of them), then visible top rows. */
export function selectPreloadSessions(sessions: SessionInfo[], max: number = MAX_PRELOAD): SessionInfo[] {
  const pinned = sessions.filter(s => s.pinned && !s.archived)
  const rest = sessions.filter(s => !s.pinned && !s.archived)
  return [...pinned, ...rest].slice(0, Math.max(0, max))
}

/**
 * Run the preload pass. Returns how many transcripts were fetched+cached.
 * Sequential + paced; never throws.
 */
export async function preloadTranscripts(deps: PreloadDeps): Promise<number> {
  const {
    gatewayUrl,
    sessions,
    freshCached = new Set<string>(),
    fetchMessages = getSessionMessages,
    push = pushTranscriptToRenderCache,
    sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
    shouldStop = () => false
  } = deps

  if (!gatewayUrl) {
    return 0
  }

  let cached = 0
  for (const session of selectPreloadSessions(sessions)) {
    if (shouldStop()) {
      break
    }
    if (freshCached.has(session.id)) {
      continue
    }
    try {
      const result = await fetchMessages(session.id, session.profile)
      const raw = Array.isArray(result?.messages) ? result.messages : []
      // Store in ChatMessage shape (what setMessages renders) so the paint
      // path never converts on click. toChatMessages is the same conversion
      // the live prefetch applies.
      const rows = raw.length > 0 ? toChatMessages(raw) : []
      if (rows.length > 0) {
        push(gatewayUrl, session.id, rows)
        cached += 1
      }
    } catch {
      // fail-open: skip this session, keep going
    }
    await sleep(PRELOAD_GAP_MS)
  }
  return cached
}
