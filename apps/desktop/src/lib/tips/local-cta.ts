/**
 * Reshow clock for the local-models campaign tips in the rotation (today: the
 * engine update). A tip ignored (timed out) may return, but on a much longer
 * clock than the rotation's: it is the same message twice, not a tour moving on.
 *
 * The local-setup offer used to live here too; it moved out of the rotation to
 * store/local-setup-offer.ts, which runs on events instead of this clock.
 */

export const LOCAL_TIP_RESHOW_MS = 7 * 24 * 60 * 60_000

/** Due = never shown, or shown long enough ago that repeating it reads as a
 *  reminder rather than a nag. Retirement is the caller's ledger, not ours. */
export function localTipDue(now: number, shownAt: number | undefined): boolean {
  return shownAt === undefined || now - shownAt >= LOCAL_TIP_RESHOW_MS
}
