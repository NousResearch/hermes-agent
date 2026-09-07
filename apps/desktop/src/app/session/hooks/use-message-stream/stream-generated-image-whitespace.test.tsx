import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'

// Mirrors the fixture the "retain Markdown boundary and code-copy whitespace"
// saga (55ad5afa12) proved for the hydration path: an assistant reply whose
// generated-image echo sits at the very end of the text, leaving a trailing
// blank line once the echo is stripped. chat-messages.test.ts and
// generated-images.test.ts now assert the stripped result keeps that
// trailing '\n\n' — stripGeneratedImageEchoes no longer trims it away.
const IMAGE_URL = 'https://cdn.example/cat.png'
const RAW_TEXT = `Here you go.\n\n![Generated image](${IMAGE_URL})`
const EXPECTED_VISIBLE_TEXT = 'Here you go.\n\n'

function mountStream(): MessageStreamHarness {
  return renderMessageStream(SID)
}

const start = (stream: MessageStreamHarness) =>
  act(() => stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' }))

const toolComplete = (stream: MessageStreamHarness) =>
  act(() =>
    stream.handleEvent({
      payload: {
        name: 'image_generate',
        result: { host_image: IMAGE_URL, image: IMAGE_URL, success: true }
      },
      session_id: SID,
      type: 'tool.complete'
    })
  )

const interim = (stream: MessageStreamHarness, text: string) =>
  act(() =>
    stream.handleEvent({ payload: { already_streamed: true, text }, session_id: SID, type: 'message.interim' })
  )

const complete = (stream: MessageStreamHarness, text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.complete' }))

function lastAssistantText(stream: MessageStreamHarness): string {
  const last = [...stream.state().messages].reverse().find(m => m.role === 'assistant' && !m.hidden)

  return last ? chatMessageText(last) : ''
}

describe('useMessageStream generated-image echo whitespace parity with hydration', () => {
  afterEach(() => {
    cleanup()
  })

  it('preserves the trailing whitespace a stripped generated-image echo leaves behind on message.complete', async () => {
    const stream = mountStream()
    await start(stream)
    await toolComplete(stream)

    await complete(stream, RAW_TEXT)

    // Before the fix, completeAssistantMessage re-trimmed the output of
    // stripGeneratedImageEchoes, collapsing this to 'Here you go.' — the
    // exact destructive behavior the hydration-path saga eliminated.
    expect(lastAssistantText(stream)).toBe(EXPECTED_VISIBLE_TEXT)
  })

  it('preserves the trailing whitespace a stripped generated-image echo leaves behind on message.interim', async () => {
    const stream = mountStream()
    await start(stream)
    await toolComplete(stream)

    await interim(stream, RAW_TEXT)

    // Same parity check for finalizeInterimAssistantMessage — it re-trimmed
    // stripGeneratedImageEchoes's output the same way completeAssistantMessage
    // did.
    expect(lastAssistantText(stream)).toBe(EXPECTED_VISIBLE_TEXT)
  })
})
