/**
 * In-app tips — the app pointing at itself, plus the agent doing the same.
 *
 * Two sources, one bubble, one switch over both of them:
 *
 * - `$tipsEnabled` is the whole feature, ON with a switch to stop it. A feature
 *   nobody meets is a feature nobody has, and the pacing is what earns the
 *   default: minutes into a launch at the earliest, then six hours, which is a
 *   nicety rather than the nag that would owe you an opt-in.
 * - It covers Hermes too. "Off" from someone who has just closed a bubble means
 *   no bubbles, not "no bubbles unless the agent sends one" — so the switch is
 *   mirrored to the gateway, where it takes the `tip` tool out of the model's
 *   schema, and the bridge drops a stray tip on top of that.
 * - `$retiredTips` is the hard-close ledger for the rotation. A tip the user ✕'d
 *   never comes back on its own; Settings → Reset is the only way, and that is
 *   the whole contract behind the ✕ being a heavier gesture than letting the
 *   bubble time out.
 * - `$activeTip` is what is on screen. Ephemeral by design: a tip is a nicety,
 *   and one that survives a reload has overstayed.
 *
 * `$lastTipId` is the rotation's cursor and `$nextTipAt` is its clock. Both are
 * persisted, because the alternative is that every relaunch reopens the tour at
 * tip one and re-arms a schedule measured in hours.
 */

import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'
import { TIP_CATALOG, type TipSide } from '@/lib/tips/catalog'
import { mirrorDisplayToggle } from '@/store/display-toggles'

/** Hours, not minutes. The catalog is ten tips and it should take weeks. */
const COOLDOWN_MS = 6 * 60 * 60_000

/** A tip as the bubble needs it: resolved copy, resolved anchor. */
export interface ActiveTip {
  /** A call to action: one button under the text. What separates a campaign
   *  tip from the rotation's — the rotation teaches, this one offers to DO
   *  the thing, and the button is the only path (an ambient bubble must
   *  never make its whole face clickable). Clicking closes the tip. */
  action?: { label: string; onSelect: () => void }
  /** Keybind action id whose live combo the bubble prints. */
  keybind?: string
  side: TipSide
  /** Candidate anchors, best first — re-resolved while the bubble is up, so a
   *  tip follows an element that re-renders and leaves when it goes away. */
  targets: readonly string[]
  text: string
  /** Catalog id. Absent for an agent-authored tip, which has nothing to retire. */
  tipId?: string
  title?: string
}

// Key still says `rotation` from when the switch only covered that half.
// Renaming it would read as unset for anyone who had already turned tips off,
// and silently turning them back on is the one outcome worth avoiding here.
const ENABLED_KEY = 'hermes.desktop.tips.rotation.v1'

export const $tipsEnabled = persistentAtom(ENABLED_KEY, true, Codecs.bool)
export const $retiredTips = persistentAtom<string[]>('hermes.desktop.tips.retired.v1', [], Codecs.stringArray)
export const $lastTipId = persistentAtom<null | string>('hermes.desktop.tips.last.v1', null, Codecs.nullableText)
export const $nextTipAt = persistentAtom<null | number>(
  'hermes.desktop.tips.next.v1',
  null,
  Codecs.json(value => (typeof value === 'number' && Number.isFinite(value) ? value : null))
)
export const $activeTip = atom<ActiveTip | null>(null)

// Off has to reach the agent, not just the renderer: the `tip` tool leaves the
// model's schema entirely rather than staying on offer and being dropped.
mirrorDisplayToggle('display.in_app_tips', ENABLED_KEY, $tipsEnabled)

/** When each campaign tip (id outside the rotation catalog) last showed.
 *  Campaign tips re-offer on their own long clock instead of walking on;
 *  `$retiredTips` still owns the hard ✕. */
export const $tipShownAt = persistentAtom<Record<string, number>>(
  'hermes.desktop.tips.shownAt.v1',
  {},
  Codecs.json(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(value).filter((entry): entry is [string, number] => typeof entry[1] === 'number')
    )
  })
)

export function setTipsEnabled(enabled: boolean): void {
  if (!enabled) {
    // Including whichever one is up: the switch is answering a bubble on
    // screen as often as it is answering the idea of them.
    $activeTip.set(null)
  }

  $tipsEnabled.set(enabled)
}

/** Un-retire everything, and let the rotation start over from a full deck
 *  rather than from wherever a six-hour cooldown had left it. */
export function resetTips(): void {
  $retiredTips.set([])
  $nextTipAt.set(null)
}

/** Put a tip on screen, replacing whatever was there. */
export function showTip(tip: ActiveTip): void {
  if (tip.tipId) {
    // The cursor belongs to the rotation's walk. A campaign tip (an id the
    // catalog doesn't hold) records when it showed but must not move the
    // cursor — nextTip treats an unknown id as "start over at the top".
    if (TIP_CATALOG.some(def => def.id === tip.tipId)) {
      $lastTipId.set(tip.tipId)
    }

    $tipShownAt.set({ ...$tipShownAt.get(), [tip.tipId]: Date.now() })
  }

  // Any tip starts the cooldown, an agent's included: whoever just pointed at
  // something, the user has had their one interruption for a good while.
  $nextTipAt.set(Date.now() + COOLDOWN_MS)
  $activeTip.set(tip)
}

/** Soft close: this one has had its moment, the rotation carries on. */
export function dismissTip(): void {
  $activeTip.set(null)
}

/** Hard close (the ✕): retire the catalog tip behind the bubble for good. An
 *  agent tip has no catalog entry, so it just closes. */
export function retireActiveTip(): void {
  const tipId = $activeTip.get()?.tipId

  if (tipId && !$retiredTips.get().includes(tipId)) {
    $retiredTips.set([...$retiredTips.get(), tipId])
  }

  $activeTip.set(null)
}
