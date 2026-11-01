import { cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type GatewayFrame, playFrames } from './test-harness'

// A turn's pre-tool commentary was painted TWICE — two separate bubbles with
// nothing between them — when a superseded attempt sealed that segment as a
// standalone interim before the turn's own `message.start`, and the turn then
// re-streamed the same text and sealed it a second time. Every collapse guard
// keys on the turn's FINAL text, so a commentary whose final answer is
// different text matched none of them (#144609).
//
// `interim-redelivery-duplicate.test.tsx` covers the neighbouring shape, where
// the interim text IS the final reply. This file covers the pre-tool segment.

const SID = 'dup-commentary-across-boundary'

const COMMENTARY =
  "For fidelity the decisive knob isn't calibration — it's `lm_head`. Let me verify what each checkpoint does with it."

const FINAL = 'Clear answer: **RadixArk/nvfp4-mix** — the rest of a long, different final answer.'

afterEach(cleanup)

const toolRound = (id: string, name: string): GatewayFrame[] => [
  ['tool.start', { name, tool_id: id, args: {} }],
  ['tool.complete', { name, tool_id: id, result: 'ok' }]
]

const visible = async (frames: GatewayFrame[]) =>
  (await playFrames(SID, frames)).map(message => chatMessageText(message).trim()).filter(Boolean)

it('paints the commentary once when the turn has no boundary in it', async () => {
  const painted = await visible([
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ['message.interim', { text: COMMENTARY, already_streamed: true }],
    ...toolRound('t1', 'execute_code'),
    ['message.delta', { text: FINAL }],
    ['message.complete', { text: FINAL }]
  ])

  expect(painted.filter(text => text === COMMENTARY)).toHaveLength(1)
})

it('paints the commentary once when a message.start falls mid-turn', async () => {
  const painted = await visible([
    ['message.start', {}],
    ['message.interim', { text: COMMENTARY, already_streamed: true }],
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ['message.interim', { text: COMMENTARY, already_streamed: true }],
    ...toolRound('t1', 'execute_code'),
    ['message.delta', { text: FINAL }],
    ['message.complete', { text: FINAL }]
  ])

  expect(painted.filter(text => text === COMMENTARY)).toHaveLength(1)
  expect(painted.filter(text => text === FINAL)).toHaveLength(1)
})

it('paints the commentary once when the interim lands before any delta', async () => {
  const painted = await visible([
    ['message.start', {}],
    ['message.interim', { text: COMMENTARY, already_streamed: false }],
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ['message.interim', { text: COMMENTARY, already_streamed: true }],
    ...toolRound('t1', 'execute_code'),
    ['message.delta', { text: FINAL }],
    ['message.complete', { text: FINAL }]
  ])

  expect(painted.filter(text => text === COMMENTARY)).toHaveLength(1)
})

it('keeps an earlier interim that owns tool rows', async () => {
  // The removable copy must be TEXT-ONLY. When a tool round lands inside the
  // interim bubble before it is sealed, that bubble is the transcript's own
  // record of an earlier round and must survive even when the text repeats.
  const painted = await visible([
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ...toolRound('t0', 'terminal'),
    ['message.interim', { text: COMMENTARY, already_streamed: true }],
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ['message.interim', { text: COMMENTARY, already_streamed: true }],
    ...toolRound('t1', 'execute_code'),
    ['message.delta', { text: FINAL }],
    ['message.complete', { text: FINAL }]
  ])

  expect(painted.filter(text => text === COMMENTARY)).toHaveLength(2)
})
