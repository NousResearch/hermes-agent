import { describe, expect, it } from 'vitest'

import {
  initialSideChatState,
  type SideChatAsk,
  type SideChatEvent,
  sideChatReducer,
  type SideChatState
} from './side-chat'

// Drive the reducer like the window does, collecting every ask it asked for.
function run(events: SideChatEvent[], from: SideChatState = initialSideChatState) {
  let state = from
  const asked: SideChatAsk[] = []

  for (const event of events) {
    const transition = sideChatReducer(state, event)
    state = transition.state

    if (transition.ask !== null) {
      asked.push(transition.ask)
    }
  }

  return { asked, state }
}

const context = (sessionId: string, question = ''): SideChatEvent => ({
  context: { question, sessionId, title: 'Fix the build' },
  type: 'context'
})

describe('sideChatReducer', () => {
  it('asks the seed question from `/btw <question>` as soon as the context lands', () => {
    const { asked, state } = run([context('s1', 'which file was that error in?')])

    expect(asked).toEqual([{ askId: 'seed-s1-0', sessionId: 's1', text: 'which file was that error in?' }])
    expect(state.turns).toEqual([
      { answer: '', askId: 'seed-s1-0', error: '', pending: true, question: 'which file was that error in?' }
    ])
  })

  it('opens empty for a bare `/btw`', () => {
    const { asked, state } = run([context('s1')])

    expect(asked).toEqual([])
    expect(state.turns).toEqual([])
    expect(state.context?.sessionId).toBe('s1')
  })

  it('sends the trimmed draft and clears it', () => {
    const { asked, state } = run([context('s1'), { draft: '  and the stack trace?  ', type: 'edit' }, { askId: 'a1', type: 'submit' }])

    expect(asked).toEqual([{ askId: 'a1', sessionId: 's1', text: 'and the stack trace?' }])
    expect(state.draft).toBe('')
    expect(state.turns.at(-1)?.pending).toBe(true)
  })

  it('refuses a blank submit and keeps the draft', () => {
    // A stray Enter must not make a half-typed follow-up disappear.
    const { asked, state } = run([context('s1'), { draft: '   ', type: 'edit' }, { askId: 'a1', type: 'submit' }])

    expect(asked).toEqual([])
    expect(state.draft).toBe('   ')
    expect(state.turns).toEqual([])
  })

  it('refuses to send before the parent conversation is known', () => {
    // prompt.btw snapshots ONE conversation; an ask with no session could never
    // be answered, so it must not leave a bubble spinning in the window.
    const { asked, state } = run([{ draft: 'why?', type: 'edit' }, { askId: 'a1', type: 'submit' }])

    expect(asked).toEqual([])
    expect(state.turns).toEqual([])
  })

  it('allows a second question while the first is still thinking', () => {
    // Each /btw is its own backend side agent over its own snapshot, so the
    // window must not impose a serialization the backend does not.
    const { asked, state } = run([
      context('s1'),
      { draft: 'first', type: 'edit' },
      { askId: 'a1', type: 'submit' },
      { draft: 'second', type: 'edit' },
      { askId: 'a2', type: 'submit' }
    ])

    expect(asked.map(ask => ask.text)).toEqual(['first', 'second'])
    expect(state.turns.every(turn => turn.pending)).toBe(true)
  })

  it('matches a reply by askId, not by position', () => {
    const { state } = run([
      context('s1'),
      { draft: 'first', type: 'edit' },
      { askId: 'a1', type: 'submit' },
      { draft: 'second', type: 'edit' },
      { askId: 'a2', type: 'submit' },
      // The second aside's agent finished first.
      { reply: { askId: 'a2', error: '', text: 'the second answer' }, type: 'reply' }
    ])

    expect(state.turns).toEqual([
      { answer: '', askId: 'a1', error: '', pending: true, question: 'first' },
      { answer: 'the second answer', askId: 'a2', error: '', pending: false, question: 'second' }
    ])
  })

  it('settles a failed ask instead of leaving it pending forever', () => {
    const { state } = run([
      context('s1', 'why?'),
      { reply: { askId: 'seed-s1-0', error: 'backend is retiring', text: '' }, type: 'reply' }
    ])

    expect(state.turns.at(-1)).toEqual({
      answer: '',
      askId: 'seed-s1-0',
      error: 'backend is retiring',
      pending: false,
      question: 'why?'
    })
  })

  it('drops a reply that matches no live ask', () => {
    const { state } = run([context('s1'), { reply: { askId: 'ghost', error: '', text: 'orphan' }, type: 'reply' }])

    expect(state.turns).toEqual([])
  })

  it('resets the thread when the side chat moves to another conversation', () => {
    const { asked, state } = run([
      context('s1', 'about s1'),
      { reply: { askId: 'seed-s1-0', error: '', text: 'answered' }, type: 'reply' },
      { draft: 'half typed', type: 'edit' },
      context('s2', 'about s2')
    ])

    expect(asked.map(ask => ask.sessionId)).toEqual(['s1', 's2'])
    expect(state.turns).toEqual([
      { answer: '', askId: 'seed-s2-0', error: '', pending: true, question: 'about s2' }
    ])
    expect(state.draft).toBe('')
  })

  it('keeps the thread when `/btw` runs again in the same conversation', () => {
    const { state } = run([
      context('s1', 'first aside'),
      { reply: { askId: 'seed-s1-0', error: '', text: 'answered' }, type: 'reply' },
      context('s1', 'second aside')
    ])

    expect(state.turns.map(turn => turn.question)).toEqual(['first aside', 'second aside'])
    expect(state.turns.at(-1)?.askId).toBe('seed-s1-1')
  })
})
