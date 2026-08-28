/**
 * In-app tips — the app pointing at itself, plus the agent doing the same.
 *
 * Two sources, one bubble, and only one of them is behind a switch:
 *
 * - `$tipRotationEnabled` is the ambient rotation, OFF until asked for. The app
 *   volunteering commentary at idle is a taste, not a default, and a feature
 *   that talks unprompted has to be opted into rather than discovered and then
 *   switched off.
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
 * `$lastTipId` is the rotation's cursor — where the next walk through the
 * catalog resumes from. Persisted, because the alternative is that every
 * relaunch reopens the tour at tip one.
 */

import { atom } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'
import type { TipSide } from '@/lib/tips/catalog'

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

export const $tipRotationEnabled = persistentAtom('hermes.desktop.tips.rotation.v1', false, Codecs.bool)
export const $retiredTips = persistentAtom<string[]>('hermes.desktop.tips.retired.v1', [], Codecs.stringArray)
export const $lastTipId = persistentAtom<null | string>('hermes.desktop.tips.last.v1', null, Codecs.nullableText)
export const $activeTip = atom<ActiveTip | null>(null)

export function setTipRotationEnabled(enabled: boolean): void {
  if (!enabled) {
    // Including whichever one is up: the switch is answering a bubble on
    // screen as often as it is answering the idea of them.
    $activeTip.set(null)
  }

  $tipRotationEnabled.set(enabled)
}

/** Un-retire everything. The rotation starts over from a full deck. */
export function resetTips(): void {
  $retiredTips.set([])
}

/** Put a tip on screen, replacing whatever was there. */
export function showTip(tip: ActiveTip): void {
  if (tip.tipId) {
    $lastTipId.set(tip.tipId)
  }

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
