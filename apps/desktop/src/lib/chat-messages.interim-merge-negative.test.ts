/** Negative cases adapted from erict16's #69202 and JoaoMarcos44's #119729.
 * Keep #107386's exact tool-boundary algorithm, not the predecessor heuristics. */
import { expect, it } from 'vitest'
import { assistantTextPart, chatMessageText, mergeFinalAssistantText, toChatMessages, upsertToolPart } from './chat-messages'

const toolCallPart = (id: string) => upsertToolPart([], { tool_id: id, name: 'terminal', args: {}, result: 'ok' }, 'complete')[0]

it('does not erase pre-tool prose merely quoted by a final or sharing a short opener', () => {
  const prefix = [assistantTextPart('OK.'), toolCallPart('first'), assistantTextPart('Detailed finding.'), toolCallPart('second')]
  for (const final of ['The earlier response said OK. and nothing else.', 'OK. The final result is different.']) {
    const merged = mergeFinalAssistantText([...prefix, assistantTextPart('Draft')], final)
    expect(merged.slice(0, prefix.length)).toEqual(prefix)
    expect(merged.filter(part => part.type === 'text').map(part => part.text)).toEqual(['OK.', 'Detailed finding.', final])
  }
})

it('does not authorize raw commentary by whitespace or media-tag resemblance to display text', () => {
  const row = { role: 'assistant' as const, content: 'Canonical final.', display_commentary: [], display_reasoning: 'Real summary.',
    codex_message_items: [{ type: 'message', role: 'assistant', phase: 'commentary',
      content: [{ type: 'output_text', text: 'MEDIA:/private/hidden.png\n  Canonical   final.' }] }] }
  for (const codex_message_items of [row.codex_message_items, JSON.stringify(row.codex_message_items)]) {
    const [message] = toChatMessages([{ ...row, codex_message_items }])
    expect(chatMessageText(message)).toBe('Canonical final.')
    expect(JSON.stringify(message.parts)).not.toContain('hidden.png')
  }
})
