import { afterEach, describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { createClientSessionState } from '@/lib/chat-runtime'
import { shouldAutoDrain } from '@/store/composer-queue'
import { $activeSessionId, $busy, $selectedStoredSessionId, $sessions } from '@/store/session'
import { $workingSessionIds, clearAllSessionStates, publishSessionState } from '@/store/session-states'

/**
 * Queue a prompt in session A while A is mid-turn, switch to B, switch back to
 * A: the queued prompt must still wait for A's in-flight turn.
 *
 * `PRIMARY_SESSION_VIEW.$busy` is what the composer's auto-drain gate consumes,
 * and `$workingSessionIds` is the authoritative per-session truth the sidebar
 * arc paints from. Any window where those two disagree about A is a window
 * where the queue drains into a still-running session.
 *
 * The gap these cover: selection moves to A immediately while `$activeSessionId`
 * still points at B (the runtime rebind is awaited behind `session.activate`),
 * so the pane would otherwise read B's idle flag for A.
 */
describe('composer busy gate across a session switch-back', () => {
  // nanostores computed atoms are lazy — they only recompute while subscribed.
  // Hold a subscription for the duration of each test so reads are live.
  const subscribeAtoms = () => {
    const unsubscribes = [PRIMARY_SESSION_VIEW.$busy.subscribe(() => {}), $workingSessionIds.subscribe(() => {})]

    return () => {
      for (const unsubscribe of unsubscribes) {
        unsubscribe()
      }
    }
  }

  afterEach(() => {
    clearAllSessionStates()
    $sessions.set([])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $busy.set(false)
  })

  it('reports session A as working while its turn is in flight', () => {
    const stop = subscribeAtoms()

    $sessions.set([{ id: 'stored-a' }, { id: 'stored-b' }] as never)
    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: true })

    expect($workingSessionIds.get()).toContain('stored-a')
    stop()
  })

  it('keeps the composer busy gate asserted across a switch away and back', () => {
    const stop = subscribeAtoms()

    $sessions.set([{ id: 'stored-a' }, { id: 'stored-b' }] as never)

    // A is mid-turn and focused.
    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: true })
    $selectedStoredSessionId.set('stored-a')
    $activeSessionId.set('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)

    // Switch to B (idle).
    publishSessionState('runtime-b', { ...createClientSessionState('stored-b'), busy: false })
    $selectedStoredSessionId.set('stored-b')
    $activeSessionId.set('runtime-b')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)

    // Switch BACK to A. Selection lands immediately; the runtime rebind is
    // awaited behind session.activate, so $activeSessionId still points at B.
    // This is the real ordering in use-session-actions.
    $selectedStoredSessionId.set('stored-a')

    // A is STILL busy per the authoritative projection...
    expect($workingSessionIds.get()).toContain('stored-a')

    // ...so the composer's gate must not report idle in this window.
    const busyDuringSwitchBack = PRIMARY_SESSION_VIEW.$busy.get()

    expect(
      shouldAutoDrain({ isBusy: busyDuringSwitchBack, parked: false, queueLength: 1 })
    ).toBe(false)

    // And once the runtime rebinds, the slice takes over and still says busy.
    $activeSessionId.set('runtime-a')
    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)

    stop()
  })

  it('goes idle once session A actually finishes its turn', () => {
    const stop = subscribeAtoms()

    $sessions.set([{ id: 'stored-a' }] as never)
    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: true })
    $selectedStoredSessionId.set('stored-a')
    $activeSessionId.set('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)

    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: false })

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)
    expect(shouldAutoDrain({ isBusy: false, parked: false, queueLength: 1 })).toBe(true)

    stop()
  })

  it('does not let a busy session A leak onto an idle session B', () => {
    const stop = subscribeAtoms()

    $sessions.set([{ id: 'stored-a' }, { id: 'stored-b' }] as never)

    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: true })
    publishSessionState('runtime-b', { ...createClientSessionState('stored-b'), busy: false })

    // Focus B while A runs. B must read idle even though A is busy.
    $selectedStoredSessionId.set('stored-b')
    $activeSessionId.set('runtime-b')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)

    // And in B's own pre-rebind window (active id still A, selection B).
    $activeSessionId.set('runtime-a')
    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)

    stop()
  })

  it('keeps the draft busy latch for a brand-new chat with no stored id', () => {
    const stop = subscribeAtoms()

    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
    $busy.set(true)

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)

    stop()
  })
})
