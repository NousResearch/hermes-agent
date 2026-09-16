import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $goalsBySession } from '@/store/goals'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const OTHER_SID = 'session-2'
let stream: MessageStreamHarness

describe('useMessageStream goal projection', () => {
  beforeEach(() => {
    $goalsBySession.set({
      [OTHER_SID]: {
        status: 'active',
        title: 'Keep the other session unchanged',
        updatedAt: 1
      }
    })
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    $goalsBySession.set({})
  })

  it('renders the structured persisted condition in the event session', () => {
    act(() =>
      stream.handleEvent({
        payload: {
          goal: {
            condition: 'Ship the exact backend-to-Desktop projection',
            exists: true,
            status: 'active'
          },
          kind: 'goal',
          text: '↻ Continuing toward goal (1/20): more work remains'
        },
        session_id: SID,
        type: 'status.update'
      })
    )

    expect($goalsBySession.get()[SID]).toMatchObject({
      detail: 'Continuing toward goal (1/20): more work remains',
      status: 'active',
      title: 'Ship the exact backend-to-Desktop projection'
    })
    expect($goalsBySession.get()[OTHER_SID]).toMatchObject({
      title: 'Keep the other session unchanged'
    })
  })

  it('keeps the prose fallback for older gateway events', () => {
    act(() =>
      stream.handleEvent({
        payload: {
          kind: 'goal',
          text: '⊙ Goal set (20-turn budget): legacy gateway goal'
        },
        session_id: SID,
        type: 'status.update'
      })
    )

    expect($goalsBySession.get()[SID]).toMatchObject({
      status: 'active',
      title: 'legacy gateway goal'
    })
  })
})
