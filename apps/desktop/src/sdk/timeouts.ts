/**
 * The hydration budgets the host hands to session opens.
 *
 * Split out of `index.ts` (a facade at its length cap): plain numbers with the reasoning
 * that set them, and plugins read them off the sdk namespace, which re-exports this file.
 */

/** Ordinary session opens fail fast when their gateway or socket is dead. */
export const DEFAULT_SESSION_HYDRATION_TIMEOUT_MS = 20_000
/** Cold Bot profiles get a larger per-attempt budget to start their backend
 *  and paint durable history. Bot Mode opts into one retry, so its effective
 *  ceiling is two bounded attempts rather than an unbounded wait. */
export const BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS = 60_000
/** Ceiling on the paint-first "Syncing…" badge. The profile gate it waits on is
 *  not guaranteed to ever fire (see beginHydrationBackgroundSync), so the badge
 *  needs a bound of its own or it outlives the wake it describes. */
export const HYDRATION_SYNC_BADGE_TIMEOUT_MS = 30_000
