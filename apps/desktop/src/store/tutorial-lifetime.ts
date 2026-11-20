import { readJson, writeJson } from '@/lib/storage'
import { $tipShownAt, setTipsEnabled } from '@/store/tips'
import { setToursEnabled } from '@/store/tours'

// Desktop-global, like the two Appearance switches: changing profiles is not
// learning the app again. Read through storage so another window's completed
// retirement cannot be replayed over a later manual re-enable.
const KEY = 'hermes.desktop.tutorials.lifetime.v1'
const INTRO_PERIOD_MS = 30 * 24 * 60 * 60_000

interface TutorialLifetime {
  autoDisabled: boolean
  startedAt: number
}

/** Retire tutorials once, after a month since first desktop use, not a month
 *  of accumulated foreground time. Settings can turn either feature back on. */
export function checkTutorialLifetime(): void {
  const now = Date.now()
  const stored = readJson<TutorialLifetime>(KEY)

  if (stored?.autoDisabled === true) {
    return
  }

  let startedAt = stored?.startedAt

  if (typeof startedAt !== 'number' || !Number.isFinite(startedAt) || startedAt <= 0) {
    // Older installs already have a tip ledger. Use its earliest valid date
    // instead of giving experienced users another month after this update.
    startedAt = Object.values($tipShownAt.get()).reduce(
      (earliest, shownAt) => (Number.isFinite(shownAt) && shownAt > 0 && shownAt < earliest ? shownAt : earliest),
      now
    )
    writeJson(KEY, { autoDisabled: false, startedAt })
  }

  if (now - startedAt < INTRO_PERIOD_MS) {
    return
  }

  writeJson(KEY, { autoDisabled: true, startedAt })
  setTipsEnabled(false)
  setToursEnabled(false)
}
