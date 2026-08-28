/**
 * The rotation's clock: when the app is quiet, offer a tip.
 *
 * Off unless the user turned it on (Settings → Appearance) — this is the half
 * of the feature that talks unprompted, so it is the half that has to be asked
 * for. An agent tip doesn't come through here at all.
 *
 * "Quiet" is doing the rest of the work. A tip is the app interrupting, so it
 * waits for a moment that is genuinely idle — nothing streaming, no dialog,
 * menu or tour on screen, the window focused, and a few seconds since the last
 * keystroke. Fail any of those and the tick simply passes; a tip is never
 * queued up to ambush the user the instant they stop typing.
 */

import { useEffect } from 'react'

import type { Translations } from '@/i18n/types'
import { resolveTipAnchor } from '@/lib/tips/anchor'
import { TIP_CATALOG } from '@/lib/tips/catalog'
import { nextTip } from '@/lib/tips/rotation'
import { $awaitingResponse, $busy } from '@/store/session'
import { $activeTip, $lastTipId, $retiredTips, $tipRotationEnabled, showTip } from '@/store/tips'

const TICK_MS = 5_000
/** Long enough that the first tip lands after you've settled in, not on boot. */
const FIRST_TIP_MS = 45_000
/** And rare enough afterwards to stay a nicety rather than a nag. */
const BETWEEN_TIPS_MS = 6 * 60_000
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
    let dueAt = Date.now() + FIRST_TIP_MS

    const noteTyping = () => {
      lastTypedAt = Date.now()
    }

    const offer = () => {
      if (!$tipRotationEnabled.get()) {
        return
      }

      if ($activeTip.get()) {
        // One is up; the next comes due a good while after it goes away.
        dueAt = Date.now() + BETWEEN_TIPS_MS

        return
      }

      if (Date.now() < dueAt || !appIsQuiet(lastTypedAt)) {
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

    const timer = window.setInterval(offer, TICK_MS)

    window.addEventListener('keydown', noteTyping, true)

    return () => {
      window.clearInterval(timer)
      window.removeEventListener('keydown', noteTyping, true)
    }
  }, [copy])
}
