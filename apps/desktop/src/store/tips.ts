/**
 * In-app tips — the app pointing at itself, plus the agent doing the same.
 *
 * Two sources, one bubble, and only one of them is behind a switch:
 *
 * - `$tipRotationEnabled` is the ambient rotation, ON with a switch to stop it.
 *   A feature nobody meets is a feature nobody has, and the pacing is what
 *   earns the default: minutes into a launch at the earliest, then six hours,
 *   which is a nicety rather than the nag that would owe you an opt-in.
 * - An agent tip is not ambient. Hermes raises one mid-sentence, in a
 *   conversation the user is having, exactly as it raises a `tour` — so it
 *   answers to the same nothing a tour answers to.
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
import type { TipSide } from '@/lib/tips/catalog'

/** Hours, not minutes. The catalog is ten tips and it should take weeks. */
const COOLDOWN_MS = 6 * 60 * 60_000

/** A tip as the bubble needs it: resolved copy, resolved anchor. */
export interface ActiveTip {
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

export const $tipRotationEnabled = persistentAtom('hermes.desktop.tips.rotation.v1', true, Codecs.bool)
export const $retiredTips = persistentAtom<string[]>('hermes.desktop.tips.retired.v1', [], Codecs.stringArray)
export const $lastTipId = persistentAtom<null | string>('hermes.desktop.tips.last.v1', null, Codecs.nullableText)
export const $nextTipAt = persistentAtom<null | number>(
  'hermes.desktop.tips.next.v1',
  null,
  Codecs.json(value => (typeof value === 'number' && Number.isFinite(value) ? value : null))
)
export const $activeTip = atom<ActiveTip | null>(null)

export function setTipRotationEnabled(enabled: boolean): void {
  if (!enabled) {
    // Including whichever one is up: the switch is answering a bubble on
    // screen as often as it is answering the idea of them.
    $activeTip.set(null)
  }

  $tipRotationEnabled.set(enabled)
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
    $lastTipId.set(tip.tipId)
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
