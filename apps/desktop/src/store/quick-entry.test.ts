import { describe, expect, it } from 'vitest'

import {
  initialQuickComposerState,
  QUICK_ENTRY_FALLBACK_AGENTS,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickComposerState,
  quickEntryAgentDisplayName,
  type QuickEntryAgentOption,
  type QuickEntrySubmitPayload
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
const agents: QuickEntryAgentOption[] = [
  { displayName: 'Hermes', profile: 'default', reachable: true },
  { displayName: 'Fizz', profile: 'fizz', reachable: true }
]

const connect: QuickComposerEvent = {
  agents,
  connected: true,
  groups: [{ displayName: 'Research Team', groupId: 'room-research', memberCount: 3, reachable: true }],
  sessions: [
    { id: 's1', title: 'Fix the build' },
    { id: 's2', title: 'Research trip' }
  ],
  type: 'state'
}

describe('quickComposerReducer', () => {
  it('ships only the default profile as a fallback and names agents by their own display name', () => {
    expect(QUICK_ENTRY_FALLBACK_AGENTS.map(agent => [agent.displayName, agent.profile])).toEqual([['Hermes', 'default']])
    expect(quickEntryAgentDisplayName('repokeeper', 'Linus')).toBe('Linus')
    expect(quickEntryAgentDisplayName('custom-agent', 'McCloud')).toBe('McCloud')
    expect(quickEntryAgentDisplayName('custom-agent', '  ')).toBe('custom-agent')
  })

  it('starts visible with fallback agents, DISCONNECTED, and targeting the current chat', () => {
    expect(initialQuickComposerState).toEqual({
      activeAgentIndex: 0,
      agents: QUICK_ENTRY_FALLBACK_AGENTS,
      connected: false,
      draft: '',
      groups: [],
      launchError: null,
      pendingRequestId: null,
      sessions: [],
      submitting: false,
      target: QUICK_TARGET_CURRENT,
      visible: true
    })
  })

  it('submit sends the trimmed draft with the target, clears it, and hides', () => {
    const { sent, state } = run([connect, { draft: '  ship it  ', type: 'edit' }, { type: 'submit' }])

    expect(sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'ship it' }])
    expect(state.draft).toBe('')
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(false)
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
      { agents: [], connected: false, groups: [], sessions: [], type: 'state' },
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

  it('opens a fresh HUD for a reachable selected agent without requiring prompt text', () => {
    const { sent, state } = run([connect, { profile: 'fizz', requestId: 'request-fizz-1', type: 'open-agent' }])

    expect(sent).toEqual([{ action: 'open-agent', profile: 'fizz', requestId: 'request-fizz-1' }])
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(true)
  })

  it('opens the selected group room without requiring prompt text', () => {
    const { sent, state } = run([connect, { groupId: 'room-research', requestId: 'request-group-1', type: 'open-group' }])

    expect(sent).toEqual([{ action: 'open-group', groupId: 'room-research', requestId: 'request-group-1' }])
    expect(state.submitting).toBe(true)
    expect(state.visible).toBe(true)
  })

  it('previews a reachable agent without launching it', () => {
    const { sent, state } = run([connect, { index: 1, type: 'select-agent' }])

    expect(sent).toEqual([])
    expect(state.activeAgentIndex).toBe(1)
    expect(state.submitting).toBe(false)
  })

  it('does not preview an unreachable or out-of-range agent', () => {
    const unreachable = {
      ...connect,
      agents: [...agents, { displayName: 'Offline', profile: 'offline', reachable: false }]
    } as QuickComposerEvent

    const connected = run([unreachable]).state

    expect(quickComposerReducer(connected, { index: 2, type: 'select-agent' }).state.activeAgentIndex).toBe(0)
    expect(quickComposerReducer(connected, { index: 99, type: 'select-agent' }).state.activeAgentIndex).toBe(0)
  })

  it('keeps the launcher open for an unknown or unreachable agent', () => {
    expect(run([connect, { profile: 'missing', requestId: 'request-missing-1', type: 'open-agent' }]).sent).toEqual([])

    const unreachable: QuickComposerEvent = {
      agents: [{ displayName: 'Remote', profile: 'remote', reachable: false }],
      connected: true,
      groups: [],
      sessions: [],
      type: 'state'
    }

    expect(run([unreachable, { profile: 'remote', requestId: 'request-remote-1', type: 'open-agent' }]).sent).toEqual(
      []
    )
  })

  it('lets a fallback agent initiate its own connection while the prompt gateway is disconnected', () => {
    const result = run([{ profile: 'default', requestId: 'request-default-1', type: 'open-agent' }])

    expect(result.sent).toEqual([{ action: 'open-agent', profile: 'default', requestId: 'request-default-1' }])
    expect(result.state.visible).toBe(true)
    expect(result.state.submitting).toBe(true)
  })

  it('hides only after the matching launch succeeds and keeps a failed launch retryable', () => {
    const pending = run([connect, { profile: 'fizz', requestId: 'request-fizz-2', type: 'open-agent' }]).state

    const stale = run(
      [{ result: { ok: true, profile: 'fizz', requestId: 'older-request' }, type: 'launch-result' }],
      pending
    ).state

    expect(stale).toEqual(pending)

    const failed = run(
      [
        {
          result: { error: 'Gateway unavailable', ok: false, profile: 'fizz', requestId: 'request-fizz-2' },
          type: 'launch-result'
        }
      ],
      pending
    ).state

    expect(failed.visible).toBe(true)
    expect(failed.launchError).toBe('Gateway unavailable')
    expect(failed.submitting).toBe(false)

    const retry = run([connect, { profile: 'fizz', requestId: 'request-fizz-3', type: 'open-agent' }], failed).state

    const succeeded = run(
      [{ result: { ok: true, profile: 'fizz', requestId: 'request-fizz-3' }, type: 'launch-result' }],
      retry
    ).state

    expect(succeeded.visible).toBe(false)
    expect(succeeded.launchError).toBeNull()
  })

  it('a picked session that vanishes from the pushed list falls back to current', () => {
    const { state } = run([
      connect,
      { target: 's2', type: 'target' },
      {
        agents,
        connected: true,
        groups: [],
        sessions: [{ id: 's1', title: 'Fix the build' }],
        type: 'state'
      }
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

  it('the blur that follows a submit does not re-send or resurrect the draft', () => {
    const { sent, state } = run([connect, { draft: 'go', type: 'edit' }, { type: 'submit' }, { type: 'blur' }])

    expect(sent).toEqual([{ target: QUICK_TARGET_CURRENT, text: 'go' }])
    expect(state.draft).toBe('')
    expect(state.submitting).toBe(false)
    expect(state.visible).toBe(false)
  })

  it('being re-summoned resets the capture surface but KEEPS the pushed gateway truth', () => {
    const afterSubmit = run([connect, { draft: 'first', type: 'edit' }, { type: 'submit' }]).state
    const { sent, state } = run([{ type: 'shown' }], afterSubmit)

    expect(sent).toEqual([])
    expect(state.draft).toBe('')
    expect(state.target).toBe(QUICK_TARGET_CURRENT)
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
})
