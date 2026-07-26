import { describe, expect, it } from 'vitest'

import {
  initialQuickComposerState,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickComposerState
} from './quick-entry'

// Drive the reducer like the window does, collecting every send it asked for.
function run(events: QuickComposerEvent[], from: QuickComposerState = initialQuickComposerState) {
  let state = from
  const sent: string[] = []

  for (const event of events) {
    const transition = quickComposerReducer(state, event)
    state = transition.state

    if (transition.send !== null) {
      sent.push(transition.send)
    }
  }

  return { sent, state }
}

describe('quickComposerReducer', () => {
  it('starts visible with an empty draft', () => {
    expect(initialQuickComposerState).toEqual({ draft: '', submitting: false, visible: true })
  })

  it('submit sends the trimmed draft, clears it, and hides', () => {
    const { sent, state } = run([{ draft: '  ship it  ', type: 'edit' }, { type: 'submit' }])

    expect(sent).toEqual(['ship it'])
    expect(state).toEqual({ draft: '', submitting: true, visible: false })
  })

  it('an empty or whitespace-only submit sends nothing and stays open', () => {
    const blank = run([{ type: 'submit' }])
    expect(blank.sent).toEqual([])
    expect(blank.state.visible).toBe(true)

    const spaces = run([{ draft: '   ', type: 'edit' }, { type: 'submit' }])
    expect(spaces.sent).toEqual([])
    // A stray Enter must not make the window vanish out from under the user.
    expect(spaces.state.visible).toBe(true)
    expect(spaces.state.draft).toBe('   ')
  })

  it('a second submit while already submitting cannot double-send', () => {
    const { sent, state } = run([{ draft: 'hello', type: 'edit' }, { type: 'submit' }, { type: 'submit' }])

    expect(sent).toEqual(['hello'])
    expect(state.submitting).toBe(true)
  })

  it('Escape dismisses without sending and discards the draft', () => {
    const { sent, state } = run([{ draft: 'never mind', type: 'edit' }, { type: 'dismiss' }])

    expect(sent).toEqual([])
    expect(state).toEqual({ draft: '', submitting: false, visible: false })
  })

  it('blur dismisses without sending', () => {
    const { sent, state } = run([{ draft: 'clicked away', type: 'edit' }, { type: 'blur' }])

    expect(sent).toEqual([])
    expect(state.visible).toBe(false)
    expect(state.draft).toBe('')
  })

  it('the blur that follows a submit does not re-send or resurrect the draft', () => {
    const { sent, state } = run([{ draft: 'go', type: 'edit' }, { type: 'submit' }, { type: 'blur' }])

    expect(sent).toEqual(['go'])
    expect(state).toEqual({ draft: '', submitting: false, visible: false })
  })

  it('being re-summoned resets to a fresh capture surface', () => {
    const afterSubmit = run([{ draft: 'first', type: 'edit' }, { type: 'submit' }]).state
    const { sent, state } = run([{ type: 'shown' }], afterSubmit)

    expect(sent).toEqual([])
    expect(state).toEqual(initialQuickComposerState)
  })

  it('re-summoning after a dismiss never carries the old draft back', () => {
    const dismissed = run([{ draft: 'stale text', type: 'edit' }, { type: 'dismiss' }]).state
    const reopened = quickComposerReducer(dismissed, { type: 'shown' }).state

    expect(reopened.draft).toBe('')
    expect(reopened.visible).toBe(true)
  })

  it('editing keeps the window open and never sends', () => {
    const { sent, state } = run([
      { draft: 'a', type: 'edit' },
      { draft: 'ab', type: 'edit' },
      { draft: 'abc', type: 'edit' }
    ])

    expect(sent).toEqual([])
    expect(state).toEqual({ draft: 'abc', submitting: false, visible: true })
  })

  it('a full summon → type → submit → summon cycle sends exactly once per round', () => {
    const first = run([{ draft: 'one', type: 'edit' }, { type: 'submit' }])
    const second = run([{ type: 'shown' }, { draft: 'two', type: 'edit' }, { type: 'submit' }], first.state)

    expect(first.sent).toEqual(['one'])
    expect(second.sent).toEqual(['two'])
  })
})
