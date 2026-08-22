import { atom } from 'nanostores'

import type { Contribution } from './types'

/** Bumped on every registry mutation — the reactive hook for non-React
 *  consumers (nanostores `computed`s, memo deps) that read `registry.getArea`
 *  imperatively. React trees should keep using `useContributions`. */
export const $registryVersion = atom(0)

type Listener = () => void

const EMPTY: readonly Contribution[] = Object.freeze([])

/**
 * The one registry every area reads from. Keyed by namespaced area id so the
 * same primitive resolves at any depth of the recursive Area scene graph
 * (`statusBar.right`, `rightColumn.panel`, `capabilities.detail.actions`, ...).
 *
 * Snapshots are cached per area and invalidated by either registry mutation or
 * a registered contribution's live visibility authority. The registry owns
 * those authority subscriptions, so replacing/removing a contribution detaches
 * the old source before it can keep invalidating a projection it no longer owns.
 *
 * Invalidation is AREA-SCOPED: mutating one area only clears that area's
 * snapshot and only notifies subscribers of that area, so registering a
 * statusbar item can never re-render a titlebar slot (or any other unrelated
 * area). `registerMany`/`removeMany` collapse a batch into one notification
 * per touched area. A global listener channel (`subscribe`) still fires on
 * every invalidation for engines that react to any change (pane adoption).
 */
export class ContributionRegistry {
  private byArea = new Map<string, Contribution[]>()
  private snapshot = new Map<string, readonly Contribution[]>()
  private areaListeners = new Map<string, Set<Listener>>()
  private globalListeners = new Set<Listener>()
  private conditionListeners = new Map<string, () => void>()

  /** Register one contribution. Returns a disposer that removes it. */
  register = (c: Contribution): (() => void) => this.registerMany([c])

  /** Register several at once. Returns a disposer that removes all of them.
   *  A batch touches each affected area exactly once — no per-item churn. */
  registerMany = (cs: Contribution[]): (() => void) => {
    cs.forEach(c => this.put(c))
    this.invalidate(cs.map(c => c.area))

    return () => this.removeMany(cs.map(c => ({ area: c.area, id: c.id })))
  }

  /** Resolved, sorted, filtered entries for an area. Stable ref until its
   *  registry contents or one of its visibility authorities changes. */
  getArea = (area: string): readonly Contribution[] => {
    const cached = this.snapshot.get(area)

    if (cached) {
      return cached
    }

    const raw = this.byArea.get(area)

    const resolved: readonly Contribution[] =
      !raw || raw.length === 0
        ? EMPTY
        : raw
            .filter(c => c.enabled !== false && (c.when ? c.when.get() : true))
            .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))

    this.snapshot.set(area, resolved)

    return resolved
  }

  /** Subscribe to ANY registry invalidation (engines that react to every
   *  change, e.g. pane adoption). React trees should prefer `subscribeArea`. */
  subscribe = (fn: Listener): (() => void) => {
    this.globalListeners.add(fn)

    return () => {
      this.globalListeners.delete(fn)
    }
  }

  /** Subscribe to invalidations of ONE area only — the leaf-level channel
   *  behind `useContributions`, so a slot re-renders solely for its own area. */
  subscribeArea = (area: string, fn: Listener): (() => void) => {
    const set = this.areaListeners.get(area) ?? new Set<Listener>()
    set.add(fn)
    this.areaListeners.set(area, set)

    return () => {
      set.delete(fn)

      if (set.size === 0) {
        this.areaListeners.delete(area)
      }
    }
  }

  private removeMany(entries: { area: string; id: string }[]) {
    const changed: string[] = []

    for (const { area, id } of entries) {
      if (this.take(area, id)) {
        changed.push(area)
      }
    }

    if (changed.length) {
      this.invalidate(changed)
    }
  }

  private conditionKey(area: string, id: string): string {
    return `${area}\u0000${id}`
  }

  /** Insert/replace an entry without notifying (batchable). */
  private put(c: Contribution) {
    const key = this.conditionKey(c.area, c.id)

    // Replacement transfers authority to the new contribution. The previous
    // source must be detached before the new one can become observable or a
    // stale source could continue invalidating an area it no longer owns.
    this.conditionListeners.get(key)?.()
    this.conditionListeners.delete(key)

    const list = this.byArea.get(c.area) ?? []
    this.byArea.set(c.area, [...list.filter(e => e.id !== c.id), c])

    if (c.when) {
      this.conditionListeners.set(key, c.when.listen(() => this.invalidate([c.area])))
    }
  }

  /** Remove an entry without notifying (batchable). Returns whether it existed. */
  private take(area: string, id: string): boolean {
    const list = this.byArea.get(area)

    if (!list) {
      return false
    }

    const next = list.filter(e => e.id !== id)

    if (next.length === list.length) {
      return false
    }

    const key = this.conditionKey(area, id)
    this.conditionListeners.get(key)?.()
    this.conditionListeners.delete(key)

    if (next.length) {
      this.byArea.set(area, next)
    } else {
      this.byArea.delete(area)
    }

    return true
  }

  private invalidate(areas: readonly string[]) {
    const unique = new Set(areas)

    for (const area of unique) {
      this.snapshot.delete(area)

      const listeners = this.areaListeners.get(area)

      if (listeners) {
        listeners.forEach(l => l())
      }
    }

    // Global listeners fire once per batch/authority change, regardless of how
    // many areas moved in a registry mutation.
    this.globalListeners.forEach(l => l())
    $registryVersion.set($registryVersion.get() + 1)
  }
}

export const registry = new ContributionRegistry()
