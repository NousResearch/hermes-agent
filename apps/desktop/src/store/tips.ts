/**
 * In-app tips — on by default, closable for good.
 *
 * Three pieces of state, three different lifetimes:
 *
 * - `$tipsEnabled` is the opt-OUT. Persisted, and mirrored into
 *   `display.in_app_tips` on the CONNECTED gateway so the agent's `tip` tool
 *   answers to the same switch the user flipped — one lever, not two.
 * - `$retiredTips` is the hard-close ledger. A tip the user ✕'d never comes
 *   back on its own; Settings → Reset is the only way, and that is the whole
 *   contract behind the ✕ being a heavier gesture than closing the bubble.
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
import { activeGateway } from '@/store/gateway'

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

// Opt-out: absent storage means on.
export const $tipsEnabled = persistentAtom('hermes.desktop.tips.v1', true, Codecs.bool)
export const $retiredTips = persistentAtom<string[]>('hermes.desktop.tips.retired.v1', [], Codecs.stringArray)
export const $lastTipId = persistentAtom<null | string>('hermes.desktop.tips.last.v1', null, Codecs.nullableText)
export const $activeTip = atom<ActiveTip | null>(null)

export function setTipsEnabled(enabled: boolean): void {
  if (!enabled) {
    $activeTip.set(null)
  }

  $tipsEnabled.set(enabled)
}

/** Un-retire everything. The rotation starts over from a full deck. */
export function resetTips(): void {
  $retiredTips.set([])
}

/** Put a tip on screen. A no-op while tips are off — every caller, agent
 *  included, goes through here so the opt-out has exactly one enforcement
 *  point. */
export function showTip(tip: ActiveTip): boolean {
  if (!$tipsEnabled.get()) {
    return false
  }

  if (tip.tipId) {
    $lastTipId.set(tip.tipId)
  }

  $activeTip.set(tip)

  return true
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

// The agent's `tip` tool gates on this key, and reading it off the CONNECTED
// gateway is what makes the toggle work the same whether that gateway is local,
// SSH, URL, or cloud. listen(), not subscribe(): boot must not write back the
// value it just read.
$tipsEnabled.listen(enabled => {
  void activeGateway()
    ?.request('config.set', { key: 'display.in_app_tips', value: String(enabled) })
    .catch(() => {
      // Not connected yet — the next toggle still lands, and absent config
      // reads as on, which matches the renderer's own default.
    })
})
