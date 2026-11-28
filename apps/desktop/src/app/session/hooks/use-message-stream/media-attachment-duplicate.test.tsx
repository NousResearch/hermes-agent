import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

const SID = 'media-attachment-duplicate'
const ANSWER = 'The model page confirms the H29.'
const MEDIA = 'MEDIA:D:\\deliverables\\report.pdf'
const STREAMED = `${ANSWER}\n\n${MEDIA}`

afterEach(cleanup)

/**
 * The shape that rendered one reply twice (#129993): the model writes its
 * answer and calls a tool in the same step, so the gateway seals that segment
 * as an interim whose text carries the `MEDIA:` line; after the tool round the
 * same prose arrives again, and the completion carries it WITHOUT the line
 * (delivery extracts the tag — 1337 chars in, 1255 out).
 */
async function play(options: { final?: string; streamed?: string }) {
  const stream = renderMessageStream(SID)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  const streamed = options.streamed ?? STREAMED

  await send('message.start')
  await send('message.delta', { text: streamed })
  await send('tool.start', { name: 'terminal', tool_id: 'before', args: { command: 'python -c import pymupdf' } })
  await send('tool.complete', { name: 'terminal', tool_id: 'before', result: 'ok' })
  await send('message.interim', { text: streamed, already_streamed: true })
  await send('tool.start', { name: 'terminal', tool_id: 'after', args: { command: 'cp report.pdf' } })
  await send('tool.complete', { name: 'terminal', tool_id: 'after', result: 'ok' })
  await send('message.delta', { text: streamed })
  await send('message.complete', { text: options.final ?? ANSWER })

  return stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)
}

it('collapses one reply whose streamed text carries a MEDIA line the completion strips', async () => {
  const messages = await play({})

  expect(messages).toHaveLength(1)
  // The attachment the sealed segment rendered stays, and both tool rounds are kept.
  expect(chatMessageText(messages[0])).toContain('#media:')
  expect(
    messages[0].parts.filter(part => part.type === 'tool-call').map(part => part.toolCallId)
  ).toEqual(['before', 'after'])
  expect(messages[0]).toMatchObject({ interim: false, pending: false })
})

it('still keeps two bubbles when the reply genuinely differs from the final', async () => {
  const messages = await play({ final: 'The model page confirms the H29, and the sink is included.' })

  expect(messages).toHaveLength(2)
})

it('leaves an attachment-free turn unchanged', async () => {
  const messages = await play({ streamed: ANSWER })

  expect(messages).toHaveLength(1)
  expect(chatMessageText(messages[0])).toBe(ANSWER)
})
