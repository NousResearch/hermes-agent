import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $guidanceLedger } from '@/store/guidance-ledger'

import { type GatewayFrame, playFrames } from './test-harness'

// A newcomer's ideas state retires after two chats in which a task finished (store/starter-ideas.ts).
// "Finished" is the gateway's own verdict on the turn: `message.complete` with status `complete` and
// an answer. Sending a message, a failed turn, a stopped turn or an empty one is not a finished task.

beforeEach(() => $guidanceLedger.set({ renameSplash: 'not_applicable' }))
afterEach(cleanup)

const turn = (complete: Record<string, unknown>): GatewayFrame[] => [
  ['message.start', {}],
  ['message.delta', { text: 'Working on it.' }],
  ['message.complete', complete]
]

const counted = () => $guidanceLedger.get().finishedTaskChats ?? []

it('counts a chat whose turn ended complete with an answer', async () => {
  await playFrames('chat-done', turn({ status: 'complete', text: 'Here is your briefing.' }))

  expect(counted()).toEqual(['chat-done'])
})

it.each([
  ['failed', { status: 'error', text: 'The provider refused.' }],
  ['stopped', { status: 'interrupted', text: 'Partial.' }],
  ['empty', { status: 'complete', text: '' }]
])('does not count a %s turn', async (_label, complete) => {
  await playFrames('chat-other', turn(complete))

  expect(counted()).toEqual([])
})

it('does not count a message that only started a turn', async () => {
  await playFrames('chat-sent', [
    ['message.start', {}],
    ['message.delta', { text: 'Thinking' }]
  ])

  expect(counted()).toEqual([])
})
