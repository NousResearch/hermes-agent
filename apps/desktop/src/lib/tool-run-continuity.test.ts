import { describe, expect, it } from 'vitest'

import { isToolCallPart, type ToolCallLike } from '@/components/assistant-ui/tool/run-summary'
import type { SessionMessage } from '@/types/hermes'

import type { ChatMessage, ChatMessagePart } from './chat-messages'
import {
  appendAssistantTextPart,
  appendReasoningPart,
  assistantTextPart,
  mergeFinalAssistantText,
  toChatMessages,
  upsertToolPart
} from './chat-messages'
import { coalesceToolOnlyAssistants, createToolMergeCache } from './chat-runtime'

/**
 * A turn described once, replayed two ways: as the gateway event stream the
 * live view builds bubbles from, and as the persisted rows `toChatMessages`
 * rehydrates on resume. Grouping is only stable if both produce the same runs.
 */
type TurnStep =
  | { kind: 'interim'; text: string }
  | { kind: 'final'; text: string }
  | { kind: 'reasoning'; text: string }
  | { kind: 'text'; text: string }
  | { kind: 'tool'; id: string; name: string }

// Mirrors use-message-stream: deltas and tool events accumulate on one pending
// bubble; `message.interim` seals it in place and starts a fresh one; and
// `message.complete` merges the final text onto whatever bubble is open.
function replayLive(steps: TurnStep[]): ChatMessage[] {
  const messages: ChatMessage[] = []
  let streamIndex = 0
  let open: ChatMessage | null = null

  const openBubble = (): ChatMessage => {
    if (open) {
      return open
    }

    streamIndex += 1
    open = { id: `assistant-stream-${streamIndex}`, role: 'assistant', parts: [], pending: true }
    messages.push(open)

    return open
  }

  const seal = (text: string, interim: boolean) => {
    const bubble = open ?? openBubble()

    bubble.parts = mergeFinalAssistantText(bubble.parts, text)
    bubble.pending = false
    bubble.interim = interim
    open = null
  }

  for (const step of steps) {
    switch (step.kind) {
      case 'interim':
        seal(step.text, true)

        break

      case 'final':
        seal(step.text, false)

        break
      case 'reasoning': {
        const bubble = openBubble()

        bubble.parts = appendReasoningPart(bubble.parts, step.text)

        break
      }

      case 'text': {
        const bubble = openBubble()

        bubble.parts = appendAssistantTextPart(bubble.parts, step.text)

        break
      }

      case 'tool': {
        const bubble = openBubble()

        bubble.parts = upsertToolPart(bubble.parts, { tool_id: step.id, name: step.name }, 'complete')

        break
      }
    }
  }

  return coalesceToolOnlyAssistants(messages, createToolMergeCache())
}

// The same turn as the gateway persists it: one row per agent iteration. A row
// is a single API response, so its reasoning and content always precede its own
// tool_calls — anything the agent says after a tool ran belongs to the next row.
function replayStored(steps: TurnStep[]): ChatMessage[] {
  const rows: SessionMessage[] = []
  let timestamp = 0
  let row: (SessionMessage & { tool_calls?: unknown[] }) | null = null

  const openRow = (afterTools: boolean) => {
    if (row && !(afterTools && row.tool_calls)) {
      return row
    }

    timestamp += 1
    row = { role: 'assistant', content: '', timestamp }
    rows.push(row)

    return row
  }

  for (const step of steps) {
    switch (step.kind) {
      case 'interim':

      case 'final':
      case 'text': {
        const current = openRow(true)

        current.content = `${current.content ?? ''}${step.text}`

        if (step.kind !== 'text') {
          row = null
        }

        break
      }

      case 'reasoning':
        openRow(true).reasoning = step.text

        break
      case 'tool': {
        const current = openRow(false)

        current.tool_calls = [
          ...(current.tool_calls ?? []),
          { id: step.id, function: { name: step.name, arguments: '{}' } }
        ]
        timestamp += 1
        rows.push({ role: 'tool', tool_call_id: step.id, tool_name: step.name, content: '{}', timestamp })

        break
      }
    }
  }

  return coalesceToolOnlyAssistants(toChatMessages(rows), createToolMergeCache())
}

/**
 * Maximal spans of back-to-back tool calls — the same rule assistant-ui applies
 * when it hands `ToolGroupSlot` a range, restated here so these tests check our
 * two part streams against each other rather than against the renderer.
 */
function toolRuns(parts: ChatMessagePart[]): ToolCallLike[][] {
  const runs: ToolCallLike[][] = []
  let previousWasTool = false

  for (const part of parts) {
    if (!isToolCallPart(part)) {
      previousWasTool = false

      continue
    }

    if (previousWasTool) {
      runs[runs.length - 1].push(part)
    } else {
      runs.push([part])
    }

    previousWasTool = true
  }

  return runs
}

function runsAcross(messages: ChatMessage[]): string[][] {
  return messages
    .filter(message => message.role === 'assistant')
    .flatMap(message => toolRuns(message.parts))
    .map(run => run.map(tool => tool.toolCallId ?? ''))
}

const TURNS: Record<string, TurnStep[]> = {
  'narration between two tool runs': [
    { kind: 'interim', text: 'Let me check the config.' },
    { kind: 'tool', id: 'a', name: 'read_file' },
    { kind: 'tool', id: 'b', name: 'read_file' },
    { kind: 'interim', text: 'Now let me edit it.' },
    { kind: 'tool', id: 'c', name: 'write_file' },
    { kind: 'final', text: 'Done.' }
  ],
  'reasoning between two tool runs': [
    { kind: 'tool', id: 'a', name: 'terminal' },
    { kind: 'reasoning', text: 'That failed, try the other path.' },
    { kind: 'tool', id: 'b', name: 'terminal' },
    { kind: 'final', text: 'Fixed.' }
  ],
  'unbroken run of tool calls': [
    { kind: 'tool', id: 'a', name: 'read_file' },
    { kind: 'tool', id: 'b', name: 'read_file' },
    { kind: 'tool', id: 'c', name: 'search_files' },
    { kind: 'final', text: 'Here is what I found.' }
  ],
  'reasoning then tools then final': [
    { kind: 'reasoning', text: 'The user wants the lint config.' },
    { kind: 'tool', id: 'a', name: 'search_files' },
    { kind: 'tool', id: 'b', name: 'read_file' },
    { kind: 'final', text: 'It lives in eslint.config.js.' }
  ],
  'tools with no narration at all': [
    { kind: 'tool', id: 'a', name: 'terminal' },
    { kind: 'final', text: 'Clean.' }
  ]
}

describe('tool run segmentation survives rehydration', () => {
  for (const [name, steps] of Object.entries(TURNS)) {
    it(name, () => {
      expect(runsAcross(replayLive(steps))).toEqual(runsAcross(replayStored(steps)))
    })
  }
})

describe('run identity', () => {
  const tool = (id: string, name: string): ChatMessagePart =>
    ({ args: {}, toolCallId: id, toolName: name, type: 'tool-call' }) as ChatMessagePart

  it('keeps the first tool call at the head as the run grows', () => {
    const [run] = toolRuns([tool('a', 'read_file'), tool('b', 'read_file')])
    const [grown] = toolRuns([tool('a', 'read_file'), tool('b', 'read_file'), tool('c', 'terminal')])

    expect(grown[0].toolCallId).toBe(run[0].toolCallId)
    expect(grown).toHaveLength(3)
  })

  it('breaks a run on any non-tool part', () => {
    const runs = toolRuns([tool('a', 'read_file'), assistantTextPart('Now editing.'), tool('b', 'write_file')])

    expect(runs.map(run => run.map(t => t.toolCallId))).toEqual([['a'], ['b']])
  })
})
