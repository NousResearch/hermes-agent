import { cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type GatewayFrame, playFrames } from './test-harness'

// A turn that streams prose BEFORE its tool round can receive a terminal frame
// whose text is cumulative — the pre-tool prose plus the final answer. Bounding
// the completion merge at the last tool call used to keep the pre-tool prose as
// its own part AND re-include it in the cumulative final, so the paragraph
// painted twice (visible in a silent tool row where nothing sits between the
// copies). mergeCurrentResponseText must strip the pre-boundary prefix before
// delegating to mergeFinalAssistantText.

const SID = 'cumulative-final-after-tool'
const PROSE = 'Let me check the files.'
const ANSWER = 'Everything looks good.'

afterEach(cleanup)

const toolRound = (id: string, name = 'terminal'): GatewayFrame[] => [
  ['tool.start', { name, tool_id: id, args: {} }],
  ['tool.complete', { name, tool_id: id, result: 'ok' }]
]

const textParts = async (frames: GatewayFrame[]) =>
  (await playFrames(SID, [['message.start', {}], ...frames])).flatMap(message =>
    message.parts.filter((part): part is Extract<(typeof message.parts)[number], { type: 'text' }> => part.type === 'text')
  )

describe('a cumulative final after a pre-tool prose round paints the prose once', () => {
  it.each([
    {
      label: 'prose → tool → streamed answer, cumulative final',
      frames: [
        ['message.delta', { text: PROSE }],
        ...toolRound('t1'),
        ['message.delta', { text: `\n\n${ANSWER}` }],
        ['message.complete', { text: `${PROSE}\n\n${ANSWER}` }]
      ]
    },
    {
      label: 'prose → tool → no streamed answer, cumulative final',
      frames: [
        ['message.delta', { text: PROSE }],
        ...toolRound('t1'),
        ['message.complete', { text: `${PROSE}\n\n${ANSWER}` }]
      ]
    },
    {
      label: 'silent tool row (todo_list), cumulative final',
      frames: [
        ['message.delta', { text: PROSE }],
        ...toolRound('m', 'todo_list'),
        ['message.complete', { text: `${PROSE}\n\n${ANSWER}` }]
      ]
    }
  ])('$label', async ({ frames }) => {
    const texts = (await textParts(frames)).map(part => part.text)

    const occurrences = texts.filter(text => text.includes(PROSE)).length
    expect(occurrences).toBe(1)
    expect(texts.join('')).toBe(`${PROSE}\n\n${ANSWER}`)
  })

  it('keeps the pre-tool prose when the final is NOT cumulative', async () => {
    const texts = (
      await textParts([
        ['message.delta', { text: PROSE }],
        ...toolRound('t1'),
        ['message.delta', { text: `\n\n${ANSWER}` }],
        ['message.complete', { text: ANSWER }]
      ])
    ).map(part => part.text)

    expect(texts.join('')).toBe(`${PROSE}\n\n${ANSWER}`)
    expect(texts.filter(text => text.includes(PROSE)).length).toBe(1)
  })

  it('leaves a cumulative final that only touches the last round unchanged', async () => {
    // Two tool rounds; the final is cumulative over the whole turn, so only the
    // first round's prose is the shared prefix that must not be re-painted.
    const messages = await playFrames(SID, [
      ['message.start', {}],
      ['message.delta', { text: 'Round one.' }],
      ...toolRound('t1'),
      ['message.delta', { text: `\n\n${ANSWER}` }],
      ...toolRound('t2'),
      ['message.complete', { text: `Round one.\n\n${ANSWER}` }]
    ])

    const text = messages.map(message => chatMessageText(message)).join('')
    expect(text.split('Round one.').length - 1).toBe(1)
    expect(text).toContain(ANSWER)
  })
})
