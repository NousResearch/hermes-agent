/**
 * The rotation's clock: a few minutes into a launch, then hours apart, offer a
 * tip if the app happens to be quiet.
 *
 * On unless the user turned it off (Settings → Appearance). An agent tip
 * doesn't come through here at all.
 *
 * The pacing is a loading-screen tip's, not a notification's. Two clocks have
 * to agree: a per-launch settling delay, so opening the app is never met with a
 * bubble, and a six-hour cooldown persisted across launches, so quitting and
 * reopening isn't a way to farm them. In practice that lands around one tip per
 * day of use and takes weeks to walk the catalog, which is the point — ten tips
 * in an afternoon is how a nicety turns into a thing people switch off.
 *
 * Then "quiet" does the rest. A tip is still the app interrupting, so it waits
 * for a moment that is genuinely idle — nothing streaming, no dialog, menu or
 * tour on screen, the window focused, and a few seconds since the last
 * keystroke. Fail any of those and the tick passes; being due only means the
 * next quiet moment is eligible, never that one gets taken.
 */

import { useEffect } from 'react'

import type { Translations } from '@/i18n/types'
import { resolveTipAnchor } from '@/lib/tips/anchor'
import { TIP_CATALOG } from '@/lib/tips/catalog'
import { nextTip } from '@/lib/tips/rotation'
import { $awaitingResponse, $busy } from '@/store/session'
import { $activeTip, $lastTipId, $nextTipAt, $retiredTips, $tipsEnabled, showTip } from '@/store/tips'

const TICK_MS = 30_000
/** Nothing in the first stretch of a launch, however long the cooldown says
 *  it's been: you opened the app to do a thing, and the tip can wait until
 *  you've done it. Jittered so it isn't the same beat every time. */
const SETTLE_MIN_MS = 5 * 60_000
const SETTLE_SPREAD_MS = 5 * 60_000
/** Typing is the clearest "I'm busy" signal the renderer gets for free. */
const TYPING_GRACE_MS = 5_000

/** Anything on screen a tip would be talking over. `.driver-popover` is the
 *  tour: two accent-lit bubbles at once is one too many. */
const BLOCKING_SURFACES =
  '[role="dialog"],[role="alertdialog"],[role="menu"],[role="listbox"],[data-overlay-surface],.driver-popover'

function appIsQuiet(lastTypedAt: number): boolean {
  if (document.visibilityState !== 'visible' || !document.hasFocus()) {
    return false
  }

  if ($busy.get() || $awaitingResponse.get()) {
    return false
  }

  if (Date.now() - lastTypedAt < TYPING_GRACE_MS) {
    return false
  }

  return !document.querySelector(BLOCKING_SURFACES)
}

/** Drive the ambient rotation for as long as the host is mounted. */
export function useTipRotation(copy: Translations['tips']) {
  useEffect(() => {
    let lastTypedAt = 0
    let settledAt = Date.now() + SETTLE_MIN_MS + Math.random() * SETTLE_SPREAD_MS

    const noteTyping = () => {
      lastTypedAt = Date.now()
    }

    const isDue = () => {
      const nextAt = $nextTipAt.get()

      return Date.now() >= settledAt && (nextAt === null || Date.now() >= nextAt)
    }

    const offer = () => {
      if (!$tipsEnabled.get() || $activeTip.get()) {
        return
      }

      if (!isDue() || !appIsQuiet(lastTypedAt)) {
        return
      }

      // Only tips with something on screen to point at are candidates, so the
      // rotation never burns a turn on a pane the user isn't showing.
      const onScreen = TIP_CATALOG.filter(tip => resolveTipAnchor(document, tip.targets))

      const chosen = nextTip(
        TIP_CATALOG.map(tip => tip.id),
        onScreen.map(tip => tip.id),
        { lastShownId: $lastTipId.get(), retired: $retiredTips.get() }
      )

      const tip = onScreen.find(candidate => candidate.id === chosen)

      if (!tip) {
        return
      }

      showTip({
        keybind: tip.keybind,
        side: tip.side,
        targets: tip.targets,
        text: copy.items[tip.id].text,
        tipId: tip.id,
        title: copy.items[tip.id].title
      })
    }

    // Turning tips on is its own kind of settled: the delay guards a
    // launch you came into with a purpose, and has nothing to say about someone
    // who just asked for tips and is owed the sight of one working.
    const unbindSwitch = $tipsEnabled.listen(enabled => {
      if (enabled) {
        settledAt = Date.now()
        offer()
      }
    })

    const timer = window.setInterval(offer, TICK_MS)

    window.addEventListener('keydown', noteTyping, true)

    return () => {
      unbindSwitch()
      window.clearInterval(timer)
      window.removeEventListener('keydown', noteTyping, true)
    }
  }, [copy])
}
