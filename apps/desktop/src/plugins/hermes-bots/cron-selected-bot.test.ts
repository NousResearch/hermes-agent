/**
 * Which bot the Routines pane is scoped to.
 *
 * #89625 (PR #89637 residual): nanostores' `.listen()` never replays the
 * current value the way `.subscribe()` does, so the focused-owner listener in
 * register() only kept `$selectedBot` current from the moment it was attached.
 * A disable → profile switch → re-enable cycle (Settings ▸ Plugins) left
 * `$selectedBot` pointed at whichever bot was active before the plugin was
 * disabled. `bindProfileSync` reseeds from the store's CURRENT value before
 * attaching the listener, so every register() starts in sync.
 *
 * #94516: the same ladder must not dead-end the pane. The SDK's
 * focusedSessionOwner store fails closed to null whenever the focused session
 * has no unique bot owner (a normal chat, ambiguous owner hints) — the common
 * case while the user browses the Bots pane. The pane falls back to the bot
 * the user clicked in the roster instead of pinning every bot on the
 * "has to appear in the roster first" placeholder.
 */

import { beforeEach, describe, expect, it } from 'vitest'

import { $selectedBot } from './bot-state'
import { bindProfileSync, resolveRoutineOwner } from './cron'
import type { RosterRow } from './types'

/** Real nanostores semantics — `get()` plus a NON-replaying `listen()` — so a
 *  regression back to a plain listen() fails honestly instead of being masked
 *  by a mock that assumes the fix. */
function fakeOwnerStore<T>(initial: T) {
  const listeners = new Set<(value: T) => void>()
  let value = initial

  return {
    get: () => value,
    listen: (listener: (value: T) => void) => {
      listeners.add(listener)

      return () => listeners.delete(listener)
    },
    set: (next: T) => {
      value = next
      listeners.forEach(listener => listener(value))
    }
  }
}

beforeEach(() => {
  $selectedBot.set('default')
})

describe('bindProfileSync reseeds, then follows', () => {
  it('resyncs on re-bind after a profile switch made while unbound', () => {
    const profile = fakeOwnerStore('blog-writer')
    const unbind = bindProfileSync(profile)

    expect($selectedBot.get()).toBe('blog-writer')

    unbind() // plugin disabled: listener torn down
    profile.set('researcher') // profile changes while nothing is listening

    bindProfileSync(profile) // plugin re-enabled: register() runs again

    expect($selectedBot.get()).toBe('researcher')
  })

  it('keeps forwarding live profile changes after (re)binding', () => {
    const profile = fakeOwnerStore('default')

    bindProfileSync(profile)
    profile.set('researcher')

    expect($selectedBot.get()).toBe('researcher')
  })

  it('follows a connection change for the same profile name', () => {
    const owner = fakeOwnerStore({ connectionId: 'source-a', profile: 'default' })

    bindProfileSync(owner)

    expect($selectedBot.get()).toBe('source-a::default')

    owner.set({ connectionId: 'source-b', profile: 'default' })

    expect($selectedBot.get()).toBe('source-b::default')
  })

  it('leaves the selection alone when the current value is empty', () => {
    bindProfileSync(fakeOwnerStore(''))

    expect($selectedBot.get()).toBe('default')
  })
})

describe('resolving the pane\u2019s owner', () => {
  const scoped = (name: string): RosterRow => ({
    connectionId: 'source-a',
    name,
    remoteSource: true,
    sourceScoped: true
  })

  it('fails closed when an authoritative focused owner has no roster row', () => {
    // Routing cron reads/mutations through a stale selection or an unscoped
    // profile name would address the wrong machine's cron store.
    expect(
      resolveRoutineOwner(
        [scoped('default')],
        { authoritative: true, connectionId: 'source-b', name: 'worker' },
        'source-a::default'
      )
    ).toBeNull()
  })

  it('#94516: falls back to the roster-clicked bot when nothing is focused', () => {
    const roster = [scoped('default'), scoped('blog-writer')]

    expect(resolveRoutineOwner(roster, null, 'source-a::blog-writer')).toBe(roster[1])
  })

  it('#94516: still fails closed when the selection has no roster row either', () => {
    // Nothing to scope cron reads/mutations to, so the pane keeps its
    // fail-closed state rather than guessing.
    expect(resolveRoutineOwner([scoped('default')], null, 'source-a::missing')).toBeNull()
  })

  it('lets a legacy SDK without focused-owner support use the selection', () => {
    const roster = [scoped('default')]

    expect(resolveRoutineOwner(roster, null, 'source-a::default')).toBe(roster[0])
  })
})
