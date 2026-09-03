import { describe, expect, it } from 'vitest'

import { selectFirstPaintTranscriptProof, waitForSessionTranscriptDom } from './session-open-dom-proof'

describe('session-open transcript DOM proof', () => {
  it('selects the newest user turn that the tail-first paint always mounts', async () => {
    const transcript = Array.from({ length: 80 }, (_, index) => [
      {
        id: `user-${index}`,
        parts: [{ text: `question ${index}`, type: 'text' as const }],
        role: 'user' as const
      },
      {
        id: `assistant-${index}`,
        parts: [{ text: `answer ${index}`, type: 'text' as const }],
        role: 'assistant' as const
      }
    ]).flat()

    const proof = selectFirstPaintTranscriptProof(transcript)

    expect(proof).toEqual({ expectedMessageId: 'user-79', expectedText: 'question 79' })

    // Model the real FIRST_PAINT_BUDGET=20 tail slice: the oldest user row is
    // absent, while the newest user/assistant group is guaranteed to mount.
    document.body.innerHTML = `
      <main data-hermes-perf-session="stored-1">
        <article data-message-id="user-79">question 79</article>
        <article>answer 79</article>
      </main>
    `

    await expect(
      waitForSessionTranscriptDom({
        ...proof!,
        expectedStoredSessionId: 'stored-1',
        root: document.body,
        timeoutMs: 20
      })
    ).resolves.toEqual(expect.any(Number))
  })

  it('resolves only when the expected session and transcript message are committed', async () => {
    const promise = waitForSessionTranscriptDom({
      expectedMessageId: 'message-1',
      expectedText: 'authoritative answer',
      expectedStoredSessionId: 'stored-1',
      root: document.body,
      timeoutMs: 100
    })

    document.body.innerHTML = `
      <main data-hermes-perf-session="stored-1">
        <article data-message-id="message-1">authoritative answer</article>
      </main>
    `

    await expect(promise).resolves.toEqual(expect.any(Number))
  })

  it('checks every mounted surface for the expected session before rejecting the transcript proof', async () => {
    document.body.innerHTML = `
      <main data-hermes-perf-session="stored-1"></main>
      <main data-hermes-perf-session="stored-1">
        <article data-message-id="message-1">authoritative answer</article>
      </main>
    `

    await expect(
      waitForSessionTranscriptDom({
        expectedMessageId: 'message-1',
        expectedText: 'authoritative answer',
        expectedStoredSessionId: 'stored-1',
        root: document.body,
        timeoutMs: 20
      })
    ).resolves.toEqual(expect.any(Number))
  })

  it('rejects a blank transcript instead of treating animation frames as a commit', async () => {
    document.body.innerHTML = '<main data-hermes-perf-session="stored-1"></main>'

    await expect(
      waitForSessionTranscriptDom({
        expectedMessageId: 'message-1',
        expectedText: 'authoritative answer',
        expectedStoredSessionId: 'stored-1',
        root: document.body,
        timeoutMs: 10
      })
    ).rejects.toThrow(/transcript DOM/i)
  })

  it('rejects a matching message rendered under the wrong session', async () => {
    document.body.innerHTML = `
      <main data-hermes-perf-session="stored-other">
        <article data-message-id="message-1">authoritative answer</article>
      </main>
    `

    await expect(
      waitForSessionTranscriptDom({
        expectedMessageId: 'message-1',
        expectedText: 'authoritative answer',
        expectedStoredSessionId: 'stored-1',
        root: document.body,
        timeoutMs: 10
      })
    ).rejects.toThrow(/transcript DOM/i)
  })

  it('reports mounted session and message ids without copying transcript text into the timeout', async () => {
    document.body.innerHTML = `
      <main data-hermes-perf-session="stored-1">
        <article data-message-id="other-message">secret transcript text</article>
      </main>
    `

    await expect(
      waitForSessionTranscriptDom({
        expectedMessageId: 'message-1',
        expectedText: 'authoritative answer',
        expectedStoredSessionId: 'stored-1',
        root: document.body,
        timeoutMs: 10
      })
    ).rejects.toThrow(/sessions=stored-1; matchingMessageIds=other-message/)

    await expect(
      waitForSessionTranscriptDom({
        expectedMessageId: 'message-1',
        expectedText: 'authoritative answer',
        expectedStoredSessionId: 'stored-1',
        root: document.body,
        timeoutMs: 10
      })
    ).rejects.not.toThrow(/secret transcript text/)
  })
})
