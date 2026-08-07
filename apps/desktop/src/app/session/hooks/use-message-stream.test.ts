import { describe, expect, it } from 'vitest'

import { assistantTextPart, resolveCompletionText, textPart } from '@/lib/chat-messages'
import type { ChatMessagePart } from '@/lib/chat-messages'
import { stripGeneratedImageEchoes, generatedImageEchoSources } from '@/lib/generated-images'

/**
 * Regression test for the streaming-text preservation fix.
 *
 * When the gateway truncates the payload in message.complete, the desktop
 * must keep the longer streaming-accumulated text instead of replacing it
 * with the shorter (truncated) completion text.
 */

describe('resolveCompletionText', () => {
  it('preserves streaming text when completion is a truncated prefix', () => {
    const streamed = 'Hello world, this is a long AI reply with lots of content'
    const completionTruncated = 'Hello world, this is'

    const result = resolveCompletionText(streamed, completionTruncated)

    expect(result).toBe(streamed)
  })

  it('preserves streaming text when completion prefix has whitespace differences', () => {
    const streamed = 'Line one\n\nLine two\n\nLine three with more details'
    const completionTruncated = 'Line one  Line two'

    const result = resolveCompletionText(streamed, completionTruncated)

    expect(result).toBe(streamed)
  })

  it('uses completion text when it is NOT a prefix of streamed text (control case)', () => {
    const streamed = 'Hello world'
    const completionDifferent = 'Completely different text that is longer'

    const result = resolveCompletionText(streamed, completionDifferent)

    expect(result).toBe(completionDifferent)
  })

  it('uses completion text when completion is longer than streamed', () => {
    const streamed = 'Short'
    const completionLonger = 'Short message with additional content from server'

    const result = resolveCompletionText(streamed, completionLonger)

    expect(result).toBe(completionLonger)
  })

  it('uses completion text when both are equal', () => {
    const text = 'Same text on both sides'

    const result = resolveCompletionText(text, text)

    expect(result).toBe(text)
  })

  it('returns streamed text when completion is empty', () => {
    const streamed = 'Some streamed content'

    const result = resolveCompletionText(streamed, '')

    expect(result).toBe(streamed)
  })

  it('returns completion text when streamed is empty', () => {
    const completion = 'Final text from completion'

    const result = resolveCompletionText('', completion)

    expect(result).toBe(completion)
  })
})

describe('replaceTextPart truncation recovery (integration)', () => {
  // Simulates the full replaceTextPart logic as used in completeAssistantMessage
  function replaceTextPart(parts: ChatMessagePart[], finalText: string): ChatMessagePart[] {
    const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()

    const visibleFinalText = stripGeneratedImageEchoes(finalText, generatedImageEchoSources(parts)).trim()

    const streamedText = parts
      .filter((p): p is Extract<ChatMessagePart, { type: 'text' }> => p.type === 'text')
      .map(p => p.text)
      .join('')
      .trim()

    const effectiveText = resolveCompletionText(streamedText, visibleFinalText)
    const effectiveDedupe = normalize(effectiveText)

    const kept = parts.filter(part => {
      if (part.type === 'text') {
        return false
      }

      if (part.type !== 'reasoning' || !effectiveDedupe) {
        return true
      }

      const r = normalize(part.text)

      return !(r && (effectiveDedupe.startsWith(r) || r.startsWith(effectiveDedupe)))
    })

    return effectiveText ? [...kept, assistantTextPart(effectiveText)] : kept
  }

  it('retains full streamed text when message.complete carries a truncated prefix', () => {
    const streamedParts: ChatMessagePart[] = [
      textPart('Hello world, this is a complete response from the AI. '),
      textPart('It includes multiple paragraphs and detailed explanations.')
    ]
    const truncatedCompletion = 'Hello world, this is a complete'

    const result = replaceTextPart(streamedParts, truncatedCompletion)

    const textContent = result
      .filter(p => p.type === 'text')
      .map(p => p.text)
      .join('')

    expect(textContent).toContain('multiple paragraphs and detailed explanations')
    expect(textContent.length).toBeGreaterThan(truncatedCompletion.length)
  })

  it('uses completion text when it is NOT a prefix of streamed content (control)', () => {
    const streamedParts: ChatMessagePart[] = [
      textPart('Original streamed content here')
    ]
    const differentCompletion = 'A completely different and longer final text from the server'

    const result = replaceTextPart(streamedParts, differentCompletion)

    const textContent = result
      .filter(p => p.type === 'text')
      .map(p => p.text)
      .join('')

    expect(textContent).toContain(differentCompletion)
  })

  it('preserves non-text parts (reasoning) alongside the effective text', () => {
    const streamedParts: ChatMessagePart[] = [
      { type: 'reasoning', text: 'Thinking about this...' } as ChatMessagePart,
      textPart('Full detailed response with many paragraphs of content.')
    ]
    const truncatedCompletion = 'Full detailed response'

    const result = replaceTextPart(streamedParts, truncatedCompletion)

    const textContent = result.filter(p => p.type === 'text')
    const reasoningContent = result.filter(p => p.type === 'reasoning')

    expect(textContent.length).toBe(1)
    expect(textContent[0].text).toContain('many paragraphs of content')
    // Reasoning that overlaps with effective text is deduplicated
    expect(reasoningContent.length).toBe(1)
  })
})
