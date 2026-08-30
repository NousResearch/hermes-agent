import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $busy, $messages, $selectedStoredSessionId, $sessions } from '@/store/session'
import { $sessionStates, dropSessionState, publishSessionState } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { PRIMARY_SESSION_VIEW } from './session-view'

const message = (id: string, text: string) => ({
  id,
  parts: [{ type: 'text' as const, text }],
  role: 'assistant' as const
})

const stateWith = (runtimeId: string, text: string, busy: boolean) => ({
  ...createClientSessionState(`stored-${runtimeId}`),
  messages: [message(`${runtimeId}-msg`, text)],
  busy
})

/**
 * The workspace pane is just the first tab: it renders from the active
 * session's own `$sessionStates` slice, exactly like a ⌘T tile.
 *
 * The regression this guards: the pane used to render straight off the global
 * `$messages`/`$busy` atoms — a mirror of whichever session was active. With
 * two turns in flight, navigating away from a still-streaming session left it
 * painting into the surface now showing a different conversation.
 */
describe('primary session view reads its own session slice', () => {
  beforeEach(() => {
    $sessionStates.set({})
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    $messages.set([])
    $busy.set(false)
  })

  afterEach(cleanup)

  it('shows the active session transcript, not a background session still streaming', () => {
    publishSessionState('runtime-background', stateWith('runtime-background', 'background turn', true))
    publishSessionState('runtime-foreground', stateWith('runtime-foreground', 'foreground turn', false))

    $activeSessionId.set('runtime-foreground')

    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([message('runtime-foreground-msg', 'foreground turn')])
    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)
  })

  it('ignores a background session that keeps streaming after the user switches away', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    $activeSessionId.set('runtime-b')
    publishSessionState('runtime-b', stateWith('runtime-b', 'session B turn', false))

    // Session A streams on: another delta lands for the session the user left.
    publishSessionState('runtime-a', {
      ...stateWith('runtime-a', 'session A turn', true),
      messages: [message('runtime-a-msg', 'session A turn'), message('runtime-a-late', 'late delta')]
    })

    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([message('runtime-b-msg', 'session B turn')])
    expect(PRIMARY_SESSION_VIEW.$lastVisibleIsUser.get()).toBe(false)
    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)
  })

  it('falls back to the draft atoms while the chat has no runtime session yet', () => {
    $messages.set([message('draft-msg', 'unsent draft')])
    $busy.set(true)

    expect(PRIMARY_SESSION_VIEW.$runtimeId.get()).toBeNull()
    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([message('draft-msg', 'unsent draft')])
    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)
    expect(PRIMARY_SESSION_VIEW.$messagesEmpty.get()).toBe(false)
  })

  it('does not mark B busy when A is still running and B has no slice yet', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    $busy.set(true)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set('stored-runtime-b')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)
  })

  // Runtime-id vs stored-id skew. `$activeSessionId` rebinds asynchronously
  // after resumeSession() — nulled on the cold path, still pointing at the
  // OUTGOING session on the warm one — while the selection flips synchronously
  // on navigate. Answering "idle" for the whole of that window is what fired
  // the composer's level-triggered queue drain into a session mid-turn.
  it('reports the selected session busy during a COLD switch window (no runtime slice yet)', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    $activeSessionId.set(null)
    $selectedStoredSessionId.set('stored-runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)
  })

  it('reports the selected session busy during a WARM switch window (pane still bound to the old runtime)', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    publishSessionState('runtime-b', stateWith('runtime-b', 'session B turn', false))
    // The warm cache never nulls the runtime id, so the pane still reads B's
    // idle slice while the route and the composer queue key already say A.
    // That slice describes a different conversation and must not answer for A.
    $activeSessionId.set('runtime-b')
    $selectedStoredSessionId.set('stored-runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)

    // Once the resume lands, the slice matches the selection and is
    // authoritative again on its own.
    $activeSessionId.set('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)
  })

  it('trusts a matching slice over the working set when the running turn ends', () => {
    // The working set is the fallback for an unknown, never an override: a
    // slice that belongs to the selection wins even while the set lags.
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', false))
    $activeSessionId.set('runtime-a')
    $selectedStoredSessionId.set('stored-runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)
  })

  it('matches the working set across a compression tip rotation', () => {
    // The working set publishes under the lineage root; the route may still
    // hold the tip (or the reverse). A strict-equality answer reads idle.
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    $sessions.set([makeSessionInfo({ _lineage_root_id: 'lineage-root-a', id: 'stored-runtime-a' })])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set('lineage-root-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)
  })

  it('still reports idle for a selected session that is genuinely not running', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', false))
    $activeSessionId.set(null)
    $selectedStoredSessionId.set('stored-runtime-a')

    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(false)
  })

  it('returns to the draft atoms when the active session state is dropped', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    $activeSessionId.set('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([message('runtime-a-msg', 'session A turn')])

    dropSessionState('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([])
    expect(PRIMARY_SESSION_VIEW.$messagesEmpty.get()).toBe(true)
  })
})
