import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $busy, $currentCwd, $messages, $selectedStoredSessionId } from '@/store/session'
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
    $currentCwd.set('')
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

  it('publishes transcript identity and equal-signature messages as one session-owned snapshot', () => {
    const sourceMessage = message('shared-msg', 'source turn')
    const destinationMessage = message('shared-msg', 'destination turn')

    publishSessionState('runtime-a', {
      ...stateWith('runtime-a', 'source turn', false),
      cwd: '/project-a',
      messages: [sourceMessage]
    })
    publishSessionState('runtime-b', {
      ...stateWith('runtime-b', 'destination turn', false),
      cwd: '/project-b',
      messages: [destinationMessage]
    })
    $activeSessionId.set('runtime-a')

    const observed: ReturnType<NonNullable<typeof PRIMARY_SESSION_VIEW.$transcript>['get']>[] = []
    const unsubscribe = PRIMARY_SESSION_VIEW.$transcript?.subscribe(snapshot => observed.push(snapshot))

    $activeSessionId.set('runtime-b')

    expect(PRIMARY_SESSION_VIEW.$transcript?.get()).toEqual({
      identity: { cwd: '/project-b', runtimeId: 'runtime-b' },
      messages: [destinationMessage]
    })
    expect(
      observed.some(snapshot => snapshot.identity.runtimeId === 'runtime-b' && snapshot.messages[0] === sourceMessage)
    ).toBe(false)
    unsubscribe?.()
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

  it('preserves the transcript reference across unrelated session and mirror updates', () => {
    const foreground = stateWith('runtime-foreground', 'foreground turn', false)

    publishSessionState('runtime-foreground', foreground)
    publishSessionState('runtime-background', stateWith('runtime-background', 'background turn', true))
    $activeSessionId.set('runtime-foreground')

    const before = PRIMARY_SESSION_VIEW.$transcript?.get()
    let notifications = 0

    const unsubscribe = PRIMARY_SESSION_VIEW.$transcript?.subscribe(() => {
      notifications += 1
    })

    notifications = 0
    publishSessionState('runtime-background', stateWith('runtime-background', 'background delta', true))
    $messages.set(foreground.messages)

    expect(PRIMARY_SESSION_VIEW.$transcript?.get()).toBe(before)
    expect(notifications).toBe(0)
    unsubscribe?.()
  })

  it('falls back to the draft atoms while the chat has no runtime session yet', () => {
    $messages.set([message('draft-msg', 'unsent draft')])
    $busy.set(true)

    expect(PRIMARY_SESSION_VIEW.$runtimeId.get()).toBeNull()
    expect(PRIMARY_SESSION_VIEW.$messages.get()).toEqual([message('draft-msg', 'unsent draft')])
    expect(PRIMARY_SESSION_VIEW.$busy.get()).toBe(true)
    expect(PRIMARY_SESSION_VIEW.$messagesEmpty.get()).toBe(false)
  })

  it('does not pair a runtime id with mirrored draft messages before its session slice arrives', () => {
    const outgoing = message('shared-msg', 'outgoing turn')

    $messages.set([outgoing])
    $activeSessionId.set('runtime-destination')

    expect(PRIMARY_SESSION_VIEW.$transcript?.get()).toEqual({
      identity: { cwd: '', runtimeId: 'runtime-destination' },
      messages: []
    })
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
