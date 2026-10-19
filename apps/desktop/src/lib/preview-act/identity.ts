/**
 * IDENTITY — deciding whether the element in front of us is one we already
 * have a handle on, and naming it if it is not.
 *
 * This is what makes a delta possible at all. If handles churned every time a
 * framework re-rendered, every look at the page would be a wall of removals and
 * additions and there would be nothing to send but the whole inventory again.
 *
 * The re-bind ladder is ported from anchortree (Apache-2.0), minus its geometry
 * rung. It is deliberately conservative: below the bar we mint a new handle
 * instead of guessing, because a re-render costing the agent a re-read is a
 * cheap mistake and a handle silently pointing at the wrong button is not.
 *
 * SELF-CONTAINMENT: see `naming.ts`. Same contract, same reason for the
 * factory shape — with one extra consequence. `identityKit` takes the naming
 * helpers as an argument rather than importing them, because an import would
 * be a cross-module identifier the bundler is free to rename out from under
 * the stringified body.
 */

import type { NamingKit } from './naming'
import type { PreviewActBinding, PreviewElement, PreviewElementChange } from './types'

export interface IdentityKit {
  /** How strongly a remembered element matches one just observed, 0 to 1. */
  affinity(was: PreviewActBinding, now: PreviewActBinding): number
  /** Do two labels share at least half their words? */
  alike(a: string, b: string): boolean
  /** Mint a handle, disambiguating against the ones already minted. */
  coin(coined: Record<string, number>, role: string, name: string): string
  /** What moved on an element that kept its handle, or null if it held still. */
  shifted(was: PreviewActBinding, entry: PreviewElement): null | PreviewElementChange
}

/** Build the identity helpers on top of a naming kit. */
export function identityKit(naming: NamingKit): IdentityKit {
  /** What moved on an element that kept its handle, or nothing if it held
   *  still. Field by field, so a status line ticking over costs the agent one
   *  short line instead of a re-run of everything already known about it. */
  const shifted = (was: PreviewActBinding, entry: PreviewElement): null | PreviewElementChange => {
    const off = !!entry.disabled
    const value = entry.value || ''

    if (was.label === entry.label && was.value === value && was.off === off) {
      return null
    }

    const moved: PreviewElementChange = { ref: was.ref }

    if (was.label !== entry.label) {
      moved.label = entry.label
    }

    if (was.value !== value) {
      moved.value = value
    }

    if (was.off !== off) {
      moved.disabled = off
    }

    return moved
  }

  /** Do two labels share at least half their words? Tolerates the count badge
   *  and the copy edit — "Inbox" against "Inbox (3)". */
  const alike = (a: string, b: string): boolean => {
    if (!a || !b) {
      return false
    }

    const one = a.toLowerCase().split(/\s+/).filter(Boolean)
    const two = b.toLowerCase().split(/\s+/).filter(Boolean)
    const both = one.filter(word => two.indexOf(word) !== -1).length
    const all = one.length + two.filter(word => one.indexOf(word) === -1).length

    return all > 0 && both / all >= 0.5
  }

  /** How strongly a remembered element matches one just observed, 0 to 1.
   *
   *  Ported from anchortree's re-bind ladder (Apache-2.0), minus its geometry
   *  rung: a centroid is only ever worth 0.1 there, it never reaches the 0.6
   *  bar on its own, and carrying coordinates through the book to buy a
   *  tie-break is not worth the measurement. */
  const affinity = (was: PreviewActBinding, now: PreviewActBinding): number => {
    // A button is not a link, however alike the rest of it reads.
    if (was.role !== now.role) {
      return 0
    }

    // Two elements that BOTH carry a stable attribute and disagree are the page
    // telling us outright that they are different things.
    if (was.stable && now.stable) {
      return was.stable === now.stable ? 1 : 0
    }

    let score = 0

    if (was.name && was.name === now.name) {
      score += 0.6
    } else if (alike(was.name, now.name)) {
      score += 0.4
    }

    if (was.path && was.path === now.path) {
      score += 0.3
    }

    return score
  }

  /** Mint a handle. Legible on purpose: the agent reads `btn-sign-in` in a
   *  three-line delta on turn nine and knows what it is, where `@e42` would
   *  send it back to an inventory twenty thousand tokens ago. Suffixes are
   *  never rewound, so a retired handle's name is not later handed to a
   *  different element on the same page. */
  const coin = (coined: Record<string, number>, role: string, name: string): string => {
    const named = naming.slug(name)
    const stem = naming.stemOf(role) + (named ? '-' + named : '')
    const nth = coined[stem] || 0

    coined[stem] = nth + 1

    return nth ? stem + '-' + nth : stem
  }

  return { affinity, alike, coin, shifted }
}
