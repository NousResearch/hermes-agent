import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $busy, $messages, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates, dropSessionState, publishSessionState } from '@/store/session-states'

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

  it('returns to the draft atoms when the active session state is dropped', () => {
    publishSessionState('runtime-a', stateWith('runtime-a', 'session A turn', true))
    $activeSessionId.set('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([message('runtime-a-msg', 'session A turn')])

    dropSessionState('runtime-a')

    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([])
    expect(PRIMARY_SESSION_VIEW.$messagesEmpty.get()).toBe(true)
  })
})
