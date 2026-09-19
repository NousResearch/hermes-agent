import { describe, expect, it } from 'vitest'

import {
  initialQuickComposerState,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickComposerState,
  type QuickEntrySubmitPayload,
  quickEntryResultEvent,
  type QuickEntrySubmitResult
} from './quick-entry'

// Drive the reducer like the window does, collecting every send it asked for.
function run(events: QuickComposerEvent[], from: QuickComposerState = initialQuickComposerState) {
  let state = from
  const sent: QuickEntrySubmitPayload[] = []

  for (const event of events) {
    const transition = quickComposerReducer(state, event)
    state = transition.state

    if (transition.send !== null) {
      sent.push(transition.send)
    }
  }

  return { sent, state }
}

// Most flows only make sense once the primary renderer has reported a live
// gateway — this is the push the quick window receives on open.
const connect: QuickComposerEvent = {
  connected: true,
  sessions: [
    { id: 's1', title: 'Fix the build' },
    { id: 's2', title: 'Research trip' }
  ],
  type: 'state'
}

describe('quickComposerReducer', () => {
  it('starts visible, empty, DISCONNECTED, and targeting the current chat', () => {
    expect(initialQuickComposerState).toEqual({
      connected: false,
      draft: '',
      error: null,
      lastSubmitText: '',
      orphanedFailure: null,
      pendingSubmitId: null,
      sessions: [],
      submitting: false,
      target: QUICK_TARGET_CURRENT,
      unknownSubmitId: null,
      visible: true
    })
  })

  it('submit sends the trimmed draft but waits for acknowledgement to clear and hide', () => {
    const { sent, state } = run([connect, { draft: '  ship it  ', type: 'edit' }, { submitId: 1, type: 'submit' }])

    expect(sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'ship it' }])
    expect(state.draft).toBe('  ship it  ')
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(true)

    const acknowledged = quickComposerReducer(state, { submitId: 1, type: 'submit-ok' }).state
    expect(acknowledged.draft).toBe('')
    expect(acknowledged.visible).toBe(false)
  })

  it('submit-error keeps the draft visible, records the error, and refocuses on retry', () => {
    const { state } = run([connect, { draft: '  keep me  ', type: 'edit' }, { submitId: 7, type: 'submit' }])
    const failed = quickComposerReducer(state, {
      message: 'Delivery failed',
      submitId: 7,
      type: 'submit-error'
    }).state

    expect(failed).toMatchObject({
      draft: '  keep me  ',
      error: 'Delivery failed',
      pendingSubmitId: null,
      submitting: false,
      visible: true
    })
  })

  it('keeps the current owner for stale acknowledgements and records an unmatched late failure', () => {
    const { state } = run([connect, { draft: 'keep', type: 'edit' }, { submitId: 7, type: 'submit' }])

    expect(quickComposerReducer(state, { submitId: 8, type: 'submit-ok' }).state).toEqual(state)
    const failed = quickComposerReducer(state, { message: 'stale', submitId: 8, type: 'submit-error' }).state

    expect(failed).toMatchObject({
      draft: 'keep',
      error: 'stale',
      lastSubmitText: '',
      orphanedFailure: { message: 'stale', text: 'keep' },
      pendingSubmitId: 7,
      submitting: true
    })
  })

  it('keeps resume and current-target submits on their existing routes', () => {
    const resumed = run([
      connect,
      { target: 's2', type: 'target' },
      { draft: 'resume here', type: 'edit' },
      { submitId: 1, type: 'submit' }
    ])
    const current = run([connect, { draft: 'current here', type: 'edit' }, { submitId: 2, type: 'submit' }])

    expect(resumed.sent).toEqual([{ target: 's2', text: 'resume here' }])
    expect(current.sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'current here' }])
  })

  it('an empty or whitespace-only submit sends nothing and stays open', () => {
    const blank = run([connect, { type: 'submit' }])
    expect(blank.sent).toEqual([])
    expect(blank.state.visible).toBe(true)

    const spaces = run([connect, { draft: '   ', type: 'edit' }, { type: 'submit' }])
    expect(spaces.sent).toEqual([])
    // A stray Enter must not make the window vanish out from under the user.
    expect(spaces.state.visible).toBe(true)
    expect(spaces.state.draft).toBe('   ')
  })

  it('submit is DISABLED while disconnected — the draft survives for the reconnect', () => {
    const { sent, state } = run([{ draft: 'hello?', type: 'edit' }, { type: 'submit' }])

    expect(sent).toEqual([])
    expect(state.visible).toBe(true)
    expect(state.draft).toBe('hello?')

    // The gateway comes back: the same draft now sends.
    const after = run([connect, { type: 'submit' }], state)
    expect(after.sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'hello?' }])
  })

  it('a disconnect push mid-composition keeps the draft but blocks the send', () => {
    const { sent, state } = run([
      connect,
      { draft: 'almost done', type: 'edit' },
      { connected: false, sessions: [], type: 'state' },
      { type: 'submit' }
    ])

    expect(sent).toEqual([])
    expect(state.connected).toBe(false)
    expect(state.draft).toBe('almost done')
  })

  it('a second submit while already submitting cannot double-send', () => {
    const { sent, state } = run([connect, { draft: 'hello', type: 'edit' }, { type: 'submit' }, { type: 'submit' }])

    expect(sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'hello' }])
    expect(state.submitting).toBe(true)
  })

  it('a picked session target rides the submit payload', () => {
    const { sent } = run([
      connect,
      { target: 's2', type: 'target' },
      { draft: 'send this there', type: 'edit' },
      { type: 'submit' }
    ])

    expect(sent).toEqual([{ target: 's2', text: 'send this there' }])
  })

  it('the new-session target rides the submit payload', () => {
    const { sent } = run([
      connect,
      { target: QUICK_TARGET_NEW, type: 'target' },
      { draft: 'fresh start', type: 'edit' },
      { type: 'submit' }
    ])

    expect(sent).toEqual([{ target: QUICK_TARGET_NEW, text: 'fresh start' }])
  })

  it('a picked session that vanishes from the pushed list falls back to current', () => {
    const { state } = run([
      connect,
      { target: 's2', type: 'target' },
      { connected: true, sessions: [{ id: 's1', title: 'Fix the build' }], type: 'state' }
    ])

    expect(state.target).toBe(QUICK_TARGET_CURRENT)
  })

  it('a state push that still contains the picked session keeps it', () => {
    const { state } = run([connect, { target: 's1', type: 'target' }, connect])

    expect(state.target).toBe('s1')
  })

  it('Escape dismisses without sending, discards the draft, and resets the target', () => {
    const { sent, state } = run([
      connect,
      { target: 's1', type: 'target' },
      { draft: 'never mind', type: 'edit' },
      { type: 'dismiss' }
    ])

    expect(sent).toEqual([])
    expect(state.draft).toBe('')
    expect(state.target).toBe(QUICK_TARGET_CURRENT)
    expect(state.visible).toBe(false)
  })

  it('blur dismisses without sending', () => {
    const { sent, state } = run([connect, { draft: 'clicked away', type: 'edit' }, { type: 'blur' }])

    expect(sent).toEqual([])
    expect(state.visible).toBe(false)
    expect(state.draft).toBe('')
  })

  it('the blur that follows a submit keeps that submit correlated and delivered', () => {
    const { sent, state } = run([connect, { draft: 'go', type: 'edit' }, { type: 'submit' }, { type: 'blur' }])

    expect(sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'go' }])
    expect(state.draft).toBe('go')
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(false)
  })

  it('being re-summoned reconnects to an in-flight submit and KEEPS the pushed gateway truth', () => {
    const afterSubmit = run([connect, { draft: 'first', type: 'edit' }, { type: 'submit' }]).state
    const { sent, state } = run([{ type: 'shown' }], afterSubmit)

    expect(sent).toEqual([])
    expect(state.draft).toBe('first')
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(true)
    // The gateway did not disconnect just because the window was re-opened.
    expect(state.connected).toBe(true)
    expect(state.sessions).toHaveLength(2)
  })

  it('re-summoning after a dismiss never carries the old draft back', () => {
    const dismissed = run([connect, { draft: 'stale text', type: 'edit' }, { type: 'dismiss' }]).state
    const reopened = quickComposerReducer(dismissed, { type: 'shown' }).state

    expect(reopened.draft).toBe('')
    expect(reopened.visible).toBe(true)
  })

  it('editing keeps the window open and never sends', () => {
    const { sent, state } = run([
      connect,
      { draft: 'a', type: 'edit' },
      { draft: 'ab', type: 'edit' },
      { draft: 'abc', type: 'edit' }
    ])

    expect(sent).toEqual([])
    expect(state.draft).toBe('abc')
    expect(state.visible).toBe(true)
  })

  it('a full summon → type → submit → summon cycle sends exactly once per round', () => {
    const first = run([connect, { draft: 'one', type: 'edit' }, { type: 'submit' }])
    const second = run([{ type: 'shown' }, { draft: 'two', type: 'edit' }, { type: 'submit' }], first.state)

    expect(first.sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'one' }])
    expect(second.sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'two' }])
  })

  it('keeps the correlation and draft when dismissed mid-submit', () => {
    const { state } = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { type: 'dismiss' }
    ])

    expect(state.draft).toBe('pending prompt')
    expect(state.pendingSubmitId).toBe(7)
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(false)
  })

  it('reconnects to the same generation when re-shown mid-submit', () => {
    const dismissed = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { type: 'dismiss' }
    ]).state
    const state = quickComposerReducer(dismissed, { type: 'shown' }).state

    expect(state.draft).toBe('pending prompt')
    expect(state.error).toBeNull()
    expect(state.pendingSubmitId).toBe(7)
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(true)
  })

  it('a late success for the owning generation clears the surface', () => {
    const dismissed = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { type: 'dismiss' }
    ]).state
    const shown = quickComposerReducer(dismissed, { type: 'shown' }).state
    const state = quickComposerReducer(shown, { submitId: 7, type: 'submit-ok' }).state

    expect(state.draft).toBe('')
    expect(state.pendingSubmitId).toBeNull()
    expect(state.visible).toBe(false)
  })

  it('a late failure for the owning generation keeps the text and shows the error', () => {
    const { state } = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { type: 'dismiss' },
      { message: 'gateway down', submitId: 7, type: 'submit-error' }
    ])

    expect(state.draft).toBe('pending prompt')
    expect(state.error).toBe('gateway down')
    expect(state.pendingSubmitId).toBeNull()
    expect(state.visible).toBe(true)
  })

  it('a timeout is unknown, keeps the correlation, and is not retryable', () => {
    const { state } = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { message: 'may still be delivered', submitId: 7, type: 'submit-unknown' }
    ])
    const retried = run(
      [
        { draft: 'try again', type: 'edit' },
        { submitId: 8, type: 'submit' }
      ],
      state
    )

    expect(state.draft).toBe('pending prompt')
    expect(state.error).toBe('may still be delivered')
    expect(state.pendingSubmitId).toBeNull()
    expect(state.submitting).toBe(false)
    expect(state.unknownSubmitId).toBe(7)
    expect(state.visible).toBe(true)
    expect(retried.sent).toEqual([])
    expect(retried.state.draft).toBe('try again')
    expect(retried.state.unknownSubmitId).toBe(7)
  })

  it('a late success clears an unknown outcome', () => {
    const { state } = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { message: 'may still be delivered', submitId: 7, type: 'submit-unknown' },
      { message: 'accepted', ok: true, type: 'late-result' }
    ])

    expect(state.draft).toBe('')
    expect(state.visible).toBe(false)
    expect(state.unknownSubmitId).toBeNull()
    expect(state.error).toBeNull()
  })

  it('a late failure proves non-acceptance and keeps the text', () => {
    const { state } = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { message: 'may still be delivered', submitId: 7, type: 'submit-unknown' },
      { message: 'gateway down', ok: false, type: 'late-result' }
    ])

    expect(state.unknownSubmitId).toBeNull()
    expect(state.error).toBe('gateway down')
    expect(state.draft).toBe('pending prompt')
    expect(state.visible).toBe(true)
  })

  it('a late result without an unknown submit changes nothing', () => {
    const { state } = run([
      connect,
      { draft: 'fresh draft', type: 'edit' },
      { message: 'accepted', ok: true, type: 'late-result' }
    ])

    expect(state.draft).toBe('fresh draft')
    expect(state.visible).toBe(true)
  })

  it('a late success reconciles the unknown outcome', () => {
    const unknown = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { message: 'may still be delivered', submitId: 7, type: 'submit-unknown' }
    ]).state
    const state = quickComposerReducer(unknown, { submitId: 7, type: 'submit-ok' }).state

    expect(state.draft).toBe('')
    expect(state.unknownSubmitId).toBeNull()
    expect(state.visible).toBe(false)
  })

  it('a late failure clears the unknown correlation so a retry is allowed', () => {
    const unknown = run([
      connect,
      { draft: 'pending prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { message: 'may still be delivered', submitId: 7, type: 'submit-unknown' }
    ]).state
    const state = quickComposerReducer(unknown, { message: 'rejected', submitId: 7, type: 'submit-error' }).state

    expect(state.draft).toBe('pending prompt')
    expect(state.error).toBe('rejected')
    expect(state.unknownSubmitId).toBeNull()
  })

  it('quickEntryResultEvent maps timeout to submit-unknown, ok to submit-ok, else submit-error', () => {
    const timeout: QuickEntrySubmitResult = { code: 'timeout', ok: false }
    const ok: QuickEntrySubmitResult = { ok: true }
    const failed: QuickEntrySubmitResult = { ok: false }
    const timeoutEvent = quickEntryResultEvent(timeout, 7)

    expect(timeoutEvent).toEqual({
      message: 'Hermes has not confirmed the prompt yet — it may still be delivered.',
      submitId: 7,
      type: 'submit-unknown'
    })
    expect(timeoutEvent.type).not.toBe('submit-error')
    expect(quickEntryResultEvent(ok, 7)).toEqual({ submitId: 7, type: 'submit-ok' })
    expect(quickEntryResultEvent(failed, 7)).toEqual({
      message: 'Quick Entry could not deliver the prompt.',
      submitId: 7,
      type: 'submit-error'
    })
  })

  it('a failure for a superseded generation is handed back on the next summon', () => {
    const { sent, state } = run([
      connect,
      { draft: 'first prompt', type: 'edit' },
      { submitId: 7, type: 'submit' },
      { type: 'dismiss' },
      { type: 'shown' },
      { draft: 'second prompt', type: 'edit' },
      { submitId: 8, type: 'submit' },
      { message: 'first failed', submitId: 7, type: 'submit-error' },
      { submitId: 8, type: 'submit-ok' },
      { type: 'shown' }
    ])

    expect(sent).toEqual([
      { target: QUICK_TARGET_CURRENT, text: 'first prompt' },
      { target: QUICK_TARGET_CURRENT, text: 'second prompt' }
    ])
    expect(state.draft).toBe('first prompt')
    expect(state.error).toBe('first failed')
  })
})
