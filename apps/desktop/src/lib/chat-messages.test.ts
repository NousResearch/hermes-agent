import { describe, expect, it } from 'vitest'

import type { ChatMessage, ChatMessagePart } from './chat-messages'
import {
  appendAssistantTextPart,
  appendReasoningPart,
  chatMessageText,
  collectUnspokenTurnSpeech,
  mergeFinalAssistantText,
  preserveLocalAssistantErrors,
  reasoningPart,
  renderMediaTags,
  toChatMessages,
  upsertToolPart
} from './chat-messages'

describe('toChatMessages', () => {
  it('keeps a turn with interleaved tool-only rows in a single bubble', () => {
    const messages = toChatMessages([
      { role: 'assistant', content: 'Planning.', timestamp: 1 },
      {
        role: 'assistant',
        content: '',
        timestamp: 2,
        tool_calls: [{ id: 'tc', function: { name: 'terminal', arguments: '{}' } }]
      },
      { role: 'assistant', content: 'Done.', timestamp: 3 }
    ])

    expect(messages).toHaveLength(1)
    expect(messages[0].parts.map(p => p.type)).toEqual(['text', 'tool-call', 'text'])
    expect(chatMessageText(messages[0])).toBe('Planning.Done.')
  })

  it('keeps assistant tool-call iterations in one loaded assistant bubble', () => {
    const messages = toChatMessages([
      { role: 'user', content: 'check this repo', timestamp: 1 },
      {
        role: 'assistant',
        content: "Let me also check if there's a top-level lint workflow.",
        timestamp: 2,
        tool_calls: [{ id: 'tc-1', function: { name: 'search_files', arguments: '{"path":".github"}' } }]
      },
      {
        role: 'tool',
        tool_call_id: 'tc-1',
        tool_name: 'search_files',
        content: '{"error":"Path not found: /repo/.github"}',
        timestamp: 3
      },
      {
        role: 'assistant',
        content: 'No CI in this repo. Build is enough.',
        timestamp: 4,
        tool_calls: [{ id: 'tc-2', function: { name: 'terminal', arguments: '{"command":"git status --short"}' } }]
      },
      {
        role: 'tool',
        tool_call_id: 'tc-2',
        tool_name: 'terminal',
        content: '{"output":"M src/ui/components/image-distortion.tsx\\n","exit_code":0}',
        timestamp: 5
      },
      { role: 'assistant', content: 'Now let me check git status and commit.', timestamp: 6 }
    ])

    const assistantMessages = messages.filter(message => message.role === 'assistant')

    expect(assistantMessages).toHaveLength(1)
    expect(assistantMessages[0].parts.filter(part => part.type === 'tool-call')).toHaveLength(2)
    expect(chatMessageText(assistantMessages[0])).toContain("Let me also check if there's a top-level lint workflow.")
    expect(chatMessageText(assistantMessages[0])).toContain('Now let me check git status and commit.')
  })

  it('hides attached context payloads from user message display', () => {
    const [message] = toChatMessages([
      {
        role: 'user',
        content:
          'what is this file\n\n--- Attached Context ---\n\n📄 @file:tsconfig.tsbuildinfo (981 tokens)\n```json\n{"root":["./src/main.tsx"]}\n```',
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('@file:tsconfig.tsbuildinfo\n\nwhat is this file')
  })

  it('renders MEDIA tags as assistant attachment links', () => {
    const [message] = toChatMessages([
      {
        role: 'assistant',
        content: "MEDIA:/Users/brooklyn/.hermes/cache/audio/tts_20260501_222725.mp3\n\nhow's that sound?",
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe(
      "[Audio: tts_20260501_222725.mp3](#media:%2FUsers%2Fbrooklyn%2F.hermes%2Fcache%2Faudio%2Ftts_20260501_222725.mp3)\n\nhow's that sound?"
    )
  })

  it('keeps the generated image on the tool row while preserving agent prose', () => {
    const [message] = toChatMessages([
      {
        content: '',
        role: 'assistant',
        timestamp: 1,
        tool_calls: [{ id: 'img-1', function: { name: 'image_generate', arguments: '{"prompt":"draw a cat"}' } }]
      },
      {
        content: '{"success":true,"image":"https://cdn.example/cat.png"}',
        role: 'tool',
        timestamp: 2,
        tool_call_id: 'img-1',
        tool_name: 'image_generate'
      },
      {
        content: 'Here you go.\n\n![Generated image](https://cdn.example/cat.png)',
        role: 'assistant',
        timestamp: 3
      }
    ])

    const toolPart = message.parts.find(
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
        part.type === 'tool-call' && part.toolName === 'image_generate'
    )

    expect(toolPart?.result).toMatchObject({ image: 'https://cdn.example/cat.png', success: true })
    // The duplicated image is stripped, but the agent's words survive.
    expect(chatMessageText(message)).toBe('Here you go.')
  })

  it('coerces non-string message content without throwing', () => {
    const [message] = toChatMessages([
      {
        content: {
          text: 'hello from object content'
        },
        role: 'assistant',
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('hello from object content')
  })

  it('applies attached-context filtering when user content is object-shaped', () => {
    const [message] = toChatMessages([
      {
        content: {
          text: 'look\n\n--- Attached Context ---\n\n📄 @file:foo.ts (10 tokens)\n```ts\nconst x = 1\n```'
        },
        role: 'user',
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('@file:foo.ts\n\nlook')
  })
})

describe('renderMediaTags', () => {
  it('renders standalone and inline MEDIA tags as links', () => {
    expect(renderMediaTags('here\nMEDIA:/tmp/voice.mp3\nthere')).toBe(
      'here\n[Audio: voice.mp3](#media:%2Ftmp%2Fvoice.mp3)\nthere'
    )
    expect(renderMediaTags('audio: MEDIA:/tmp/voice.mp3 done')).toBe(
      'audio: [Audio: voice.mp3](#media:%2Ftmp%2Fvoice.mp3) done'
    )
    expect(renderMediaTags('MEDIA:/tmp/demo.mp4')).toBe('[Video: demo.mp4](#media:%2Ftmp%2Fdemo.mp4)')
  })

  it('renders streamed assistant media once the tag is complete', () => {
    const parts = appendAssistantTextPart(appendAssistantTextPart([], 'ok\nMEDIA:'), '/tmp/voice.mp3')
    const text = chatMessageText({ id: 'a', role: 'assistant', parts })

    expect(text).toBe('ok\n[Audio: voice.mp3](#media:%2Ftmp%2Fvoice.mp3)')
  })
})

describe('interleaved reasoning/text coalescing', () => {
  it('keeps narration contiguous when reasoning interrupts mid-sentence', () => {
    // Models that interleave reasoning_content + content deltas emit
    // text → reasoning → text within one tool-bounded segment. The two text
    // fragments are really one sentence and must not be split by the
    // "Thinking" block between them.
    let parts: ChatMessagePart[] = appendAssistantTextPart([], 'Let me ')
    parts = appendReasoningPart(parts, 'checking the file...')
    parts = appendAssistantTextPart(parts, 'verify the full file is correct:')

    expect(parts.map(p => p.type)).toEqual(['text', 'reasoning'])
    expect((parts[0] as { text: string }).text).toBe('Let me verify the full file is correct:')
    expect((parts[1] as { text: string }).text).toBe('checking the file...')
  })

  it('merges reasoning bursts that straddle a narration fragment', () => {
    let parts: ChatMessagePart[] = appendReasoningPart([], 'first thought ')
    parts = appendAssistantTextPart(parts, 'Working on it.')
    parts = appendReasoningPart(parts, 'second thought')

    expect(parts.map(p => p.type)).toEqual(['reasoning', 'text'])
    expect((parts[0] as { text: string }).text).toBe('first thought second thought')
    expect((parts[1] as { text: string }).text).toBe('Working on it.')
  })

  it('starts a fresh text part after a tool call (segment boundary)', () => {
    let parts: ChatMessagePart[] = appendAssistantTextPart([], 'Let me check.')
    parts = upsertToolPart(parts, { name: 'read_file', tool_id: 'tc-1' }, 'running')
    parts = appendAssistantTextPart(parts, 'Now editing.')

    expect(parts.map(p => p.type)).toEqual(['text', 'tool-call', 'text'])
    expect((parts[0] as { text: string }).text).toBe('Let me check.')
    expect((parts[2] as { text: string }).text).toBe('Now editing.')
  })

  it('does not merge reasoning across a tool call', () => {
    let parts: ChatMessagePart[] = appendReasoningPart([], 'before tool')
    parts = upsertToolPart(parts, { name: 'read_file', tool_id: 'tc-1' }, 'running')
    parts = appendReasoningPart(parts, 'after tool')

    expect(parts.map(p => p.type)).toEqual(['reasoning', 'tool-call', 'reasoning'])
    expect((parts[0] as { text: string }).text).toBe('before tool')
    expect((parts[2] as { text: string }).text).toBe('after tool')
  })
})

describe('preserveLocalAssistantErrors', () => {
  it('preserves a local user+error pair when hydration omits the failed turn', () => {
    const nextMessages: ChatMessage[] = [
      {
        id: 'stored-user',
        parts: [{ text: 'earlier', type: 'text' }],
        role: 'user'
      }
    ]

    const currentMessages: ChatMessage[] = [
      {
        id: 'stored-user',
        parts: [{ text: 'earlier', type: 'text' }],
        role: 'user'
      },
      {
        id: 'user-123',
        parts: [{ text: 'new prompt', type: 'text' }],
        role: 'user'
      },
      {
        error: 'OpenRouter 403',
        id: 'assistant-error-1',
        parts: [],
        role: 'assistant'
      }
    ]

    const merged = preserveLocalAssistantErrors(nextMessages, currentMessages)

    expect(merged.map(message => message.id)).toEqual(['stored-user', 'user-123', 'assistant-error-1'])
    expect(merged[2]?.error).toBe('OpenRouter 403')
  })

  it('does not keep orphan local user turns when there is no inline assistant error', () => {
    const nextMessages: ChatMessage[] = [
      {
        id: 'stored-user',
        parts: [{ text: 'earlier', type: 'text' }],
        role: 'user'
      }
    ]

    const currentMessages: ChatMessage[] = [
      ...nextMessages,
      {
        id: 'user-123',
        parts: [{ text: 'new prompt', type: 'text' }],
        role: 'user'
      }
    ]

    const merged = preserveLocalAssistantErrors(nextMessages, currentMessages)

    expect(merged.map(message => message.id)).toEqual(['stored-user'])
  })

  it('does not duplicate local user when stored history already has equivalent text', () => {
    const nextMessages: ChatMessage[] = [
      {
        id: 'stored-user',
        parts: [{ text: 'hi', type: 'text' }],
        role: 'user'
      }
    ]

    const currentMessages: ChatMessage[] = [
      {
        id: 'optimistic-user',
        parts: [{ text: 'hi', type: 'text' }],
        role: 'user'
      },
      {
        error: 'OpenRouter 403',
        id: 'assistant-error-1',
        parts: [],
        role: 'assistant'
      }
    ]

    const merged = preserveLocalAssistantErrors(nextMessages, currentMessages)

    expect(merged.map(message => message.id)).toEqual(['stored-user', 'assistant-error-1'])
  })

  it('keeps local user when only older history has equivalent text', () => {
    const nextMessages: ChatMessage[] = [
      {
        id: 'older-user',
        parts: [{ text: 'hi', type: 'text' }],
        role: 'user'
      },
      {
        id: 'older-assistant',
        parts: [{ text: 'hello', type: 'text' }],
        role: 'assistant'
      },
      {
        id: 'tail-user',
        parts: [{ text: 'different prompt', type: 'text' }],
        role: 'user'
      }
    ]

    const currentMessages: ChatMessage[] = [
      {
        id: 'optimistic-user',
        parts: [{ text: 'hi', type: 'text' }],
        role: 'user'
      },
      {
        error: 'OpenRouter 403',
        id: 'assistant-error-1',
        parts: [],
        role: 'assistant'
      }
    ]

    const merged = preserveLocalAssistantErrors(nextMessages, currentMessages)

    expect(merged.map(message => message.id)).toEqual([
      'older-user',
      'older-assistant',
      'tail-user',
      'optimistic-user',
      'assistant-error-1'
    ])
  })

  it('keeps local assistant error when hydrated message reuses same id', () => {
    const nextMessages: ChatMessage[] = [
      {
        id: 'user-1',
        parts: [{ text: 'new prompt', type: 'text' }],
        role: 'user'
      },
      {
        id: 'assistant-stream-1',
        parts: [{ text: '', type: 'text' }],
        role: 'assistant'
      }
    ]

    const currentMessages: ChatMessage[] = [
      {
        id: 'user-1',
        parts: [{ text: 'new prompt', type: 'text' }],
        role: 'user'
      },
      {
        error: 'OpenRouter 403',
        id: 'assistant-stream-1',
        parts: [],
        role: 'assistant'
      }
    ]

    const merged = preserveLocalAssistantErrors(nextMessages, currentMessages)

    const assistant = merged.find(message => message.id === 'assistant-stream-1')

    expect(assistant?.error).toBe('OpenRouter 403')
    expect(assistant?.pending).toBe(false)
  })
})

describe('upsertToolPart', () => {
  it('preserves inline diffs from tool completion events', () => {
    const parts = upsertToolPart(
      [],
      {
        inline_diff: '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new',
        name: 'patch',
        tool_id: 'tool-1'
      },
      'complete'
    )

    const [part] = parts

    expect(part?.type).toBe('tool-call')
    expect(part && 'result' in part ? part.result : undefined).toMatchObject({
      inline_diff: '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    })
  })

  it('keeps live todo rows stable across sparse progress payloads', () => {
    const first = upsertToolPart(
      [],
      {
        name: 'todo',
        todos: [{ content: 'Boil water', id: 'boil', status: 'in_progress' }],
        tool_id: 'todo-1'
      },
      'running'
    )

    const progressed = upsertToolPart(
      first,
      {
        name: 'todo',
        preview: 'updating plan',
        tool_id: 'todo-1'
      },
      'running'
    )

    const [part] = progressed
    const args = part && 'args' in part ? (part.args as Record<string, unknown>) : {}

    expect(args.todos).toEqual([{ content: 'Boil water', id: 'boil', status: 'in_progress' }])
  })

  it('archives todo state on completion and accepts explicit empty clears', () => {
    const started = upsertToolPart(
      [],
      {
        name: 'todo',
        todos: [{ content: 'Boil water', id: 'boil', status: 'in_progress' }],
        tool_id: 'todo-1'
      },
      'running'
    )

    const completed = upsertToolPart(
      started,
      {
        name: 'todo',
        tool_id: 'todo-1'
      },
      'complete'
    )

    const cleared = upsertToolPart(
      completed,
      {
        name: 'todo',
        todos: [],
        tool_id: 'todo-1'
      },
      'complete'
    )

    const completedResult =
      completed[0] && 'result' in completed[0] ? (completed[0].result as Record<string, unknown>) : {}

    const clearedResult = cleared[0] && 'result' in cleared[0] ? (cleared[0].result as Record<string, unknown>) : {}

    expect(completedResult.todos).toEqual([{ content: 'Boil water', id: 'boil', status: 'in_progress' }])
    expect(clearedResult.todos).toEqual([])
  })

  it('keeps parallel same-name tools distinct without explicit ids', () => {
    const startedTokyo = upsertToolPart(
      [],
      {
        context: 'tokyo weather',
        name: 'web_search'
      },
      'running'
    )

    const startedReykjavik = upsertToolPart(
      startedTokyo,
      {
        context: 'reykjavik weather',
        name: 'web_search'
      },
      'running'
    )

    const completedTokyo = upsertToolPart(
      startedReykjavik,
      {
        context: 'tokyo weather',
        message: 'tokyo done',
        name: 'web_search',
        summary: 'Did 5 searches'
      },
      'complete'
    )

    const completedBoth = upsertToolPart(
      completedTokyo,
      {
        context: 'reykjavik weather',
        message: 'reykjavik done',
        name: 'web_search',
        summary: 'Did 5 searches'
      },
      'complete'
    )

    const webParts = completedBoth.filter(
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
        part.type === 'tool-call' && part.toolName === 'web_search'
    )

    const contexts = webParts.map(part => String((part.args as Record<string, unknown>)?.context || ''))

    const summaries = webParts.map(part => {
      if (!('result' in part) || !part.result || typeof part.result !== 'object') {
        return ''
      }

      return String((part.result as Record<string, unknown>).summary || '')
    })

    expect(webParts).toHaveLength(2)
    expect(contexts).toEqual(['tokyo weather', 'reykjavik weather'])
    expect(summaries).toEqual(['Did 5 searches', 'Did 5 searches'])
  })

  it('pairs a terminal completion with its context-only start when event IDs differ', () => {
    const started = upsertToolPart(
      [],
      { context: 'echo "Hello from the terminal"', name: 'terminal', tool_id: 'terminal-start' },
      'running'
    )

    const completed = upsertToolPart(
      started,
      {
        args: { command: 'echo "Hello from the terminal"' },
        name: 'terminal',
        result: { exit_code: 0, stdout: 'Hello from the terminal' },
        tool_id: 'terminal-complete'
      },
      'complete'
    )

    const terminalParts = completed.filter(
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> => part.type === 'tool-call' && part.toolName === 'terminal'
    )

    expect(terminalParts).toHaveLength(1)
    expect(terminalParts[0]?.toolCallId).toBe('terminal-complete')
    expect(terminalParts[0] && 'result' in terminalParts[0] ? terminalParts[0].result : undefined).toMatchObject({
      exit_code: 0,
      stdout: 'Hello from the terminal'
    })
  })

  it('preserves query args when completion payload omits context', () => {
    const started = upsertToolPart(
      [],
      {
        context: 'auckland weather today and tomorrow forecast',
        name: 'web_search',
        tool_id: 'search-1'
      },
      'running'
    )

    const completed = upsertToolPart(
      started,
      {
        duration_s: 1.1,
        name: 'web_search',
        summary: 'Did 5 searches in 1.1s',
        tool_id: 'search-1'
      },
      'complete'
    )

    const [part] = completed

    expect(part?.type).toBe('tool-call')
    expect((part as Extract<ChatMessagePart, { type: 'tool-call' }>).args).toMatchObject({
      context: 'auckland weather today and tomorrow forecast'
    })
    expect((part as Extract<ChatMessagePart, { type: 'tool-call' }>).result).toMatchObject({
      summary: 'Did 5 searches in 1.1s'
    })
  })

  it('does not append phantom same-name tool rows for id-less progress updates', () => {
    const startedA = upsertToolPart(
      [],
      {
        context: 'reykjavik weather today and tomorrow forecast',
        name: 'web_search'
      },
      'running'
    )

    const startedB = upsertToolPart(
      startedA,
      {
        context: 'kathmandu weather today and tomorrow forecast',
        name: 'web_search'
      },
      'running'
    )

    const progressed = upsertToolPart(
      startedB,
      {
        name: 'web_search'
      },
      'running'
    )

    const webParts = progressed.filter(
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
        part.type === 'tool-call' && part.toolName === 'web_search'
    )

    expect(webParts).toHaveLength(2)
  })

  it('matches id-less live starts with later identified completions', () => {
    const started = upsertToolPart(
      [],
      {
        context: 'asuncion paraguay weather today and tomorrow forecast',
        name: 'web_search'
      },
      'running'
    )

    const completed = upsertToolPart(
      started,
      {
        context: 'asuncion paraguay weather today and tomorrow forecast',
        duration_s: 1.1,
        name: 'web_search',
        summary: 'Did 5 searches in 1.1s',
        tool_id: 'search-asuncion'
      },
      'complete'
    )

    const webParts = completed.filter(
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
        part.type === 'tool-call' && part.toolName === 'web_search'
    )

    expect(webParts).toHaveLength(1)
    expect(webParts[0].toolCallId).toBe('search-asuncion')
    expect(webParts[0].result).toMatchObject({ summary: 'Did 5 searches in 1.1s' })
  })

  it('matches id-less live starts with later identified progress updates', () => {
    const started = upsertToolPart(
      [],
      {
        context: 'reykjavik tashkent uzbekistan weather today and tomorrow forecast',
        name: 'web_search'
      },
      'running'
    )

    const progressed = upsertToolPart(
      started,
      {
        context: 'reykjavik tashkent uzbekistan weather today and tomorrow forecast',
        name: 'web_search',
        tool_id: 'search-reykjavik'
      },
      'running'
    )

    const webParts = progressed.filter(
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
        part.type === 'tool-call' && part.toolName === 'web_search'
    )

    expect(webParts).toHaveLength(1)
    expect(webParts[0].toolCallId).toBe('search-reykjavik')
  })

  it('reconciles preview-first progress rows with later stable-id starts', () => {
    const progressA = upsertToolPart(
      [],
      {
        name: 'web_search',
        preview: 'tokyo weather'
      },
      'running'
    )

    const progressB = upsertToolPart(
      progressA,
      {
        name: 'web_search',
        preview: 'reykjavik weather'
      },
      'running'
    )

    const startedA = upsertToolPart(
      progressB,
      {
        args: { query: 'tokyo weather' },
        name: 'web_search',
        tool_id: 'search-tokyo'
      },
      'running'
    )

    const startedB = upsertToolPart(
      startedA,
      {
        args: { query: 'reykjavik weather' },
        name: 'web_search',
        tool_id: 'search-reykjavik'
      },
      'running'
    )

    const completedA = upsertToolPart(
      startedB,
      {
        name: 'web_search',
        summary: 'Did 5 searches',
        tool_id: 'search-tokyo'
      },
      'complete'
    )

    const completedB = upsertToolPart(
      completedA,
      {
        name: 'web_search',
        summary: 'Did 5 searches',
        tool_id: 'search-reykjavik'
      },
      'complete'
    )

    const webParts = completedB
      .filter(
        (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
          part.type === 'tool-call' && part.toolName === 'web_search'
      )
      .map(part => ({
        id: part.toolCallId,
        query: String((part.args as Record<string, unknown>)?.query || ''),
        summary:
          part.result && typeof part.result === 'object'
            ? String((part.result as Record<string, unknown>).summary || '')
            : ''
      }))

    expect(webParts).toEqual([
      { id: 'search-tokyo', query: 'tokyo weather', summary: 'Did 5 searches' },
      { id: 'search-reykjavik', query: 'reykjavik weather', summary: 'Did 5 searches' }
    ])
  })

  it('uses structured live tool args for titles before hydrate', () => {
    const started = upsertToolPart(
      [],
      {
        args: { search_term: 'reykjavik bishkek kyrgyzstan weather today and tomorrow forecast' },
        name: 'web_search',
        tool_id: 'search-bishkek'
      },
      'running'
    )

    const [part] = started

    expect(part?.type).toBe('tool-call')
    expect((part as Extract<ChatMessagePart, { type: 'tool-call' }>).args).toMatchObject({
      search_term: 'reykjavik bishkek kyrgyzstan weather today and tomorrow forecast'
    })
  })

  it('keeps structured live tool results before hydrate', () => {
    const completed = upsertToolPart(
      [],
      {
        args: { query: 'suva weather' },
        name: 'web_search',
        result: { data: { web: [{ title: 'Suva forecast', url: 'https://example.test', description: 'Sunny' }] } },
        summary: 'Did 1 search in 0.5s',
        tool_id: 'search-suva'
      },
      'complete'
    )

    const [part] = completed

    expect(part?.type).toBe('tool-call')
    expect((part as Extract<ChatMessagePart, { type: 'tool-call' }>).result).toMatchObject({
      data: { web: [{ title: 'Suva forecast' }] },
      summary: 'Did 1 search in 0.5s'
    })
  })
})

describe('mergeFinalAssistantText', () => {
  it('removes all text parts and appends the final text', () => {
    const parts = [
      { type: 'text' as const, text: 'streamed delta 1' },
      { type: 'text' as const, text: 'streamed delta 2' },
      { type: 'tool-call' as const, toolCallId: 'tc1', toolName: 'terminal', args: {} as never, argsText: '{}' }
    ]

    const result = mergeFinalAssistantText(parts, 'final answer')

    expect(result.filter(p => p.type === 'text')).toHaveLength(1)
    expect(result.filter(p => p.type === 'text')[0]).toMatchObject({ text: 'final answer' })
    expect(result.some(p => p.type === 'tool-call')).toBe(true)
  })

  it('drops reasoning that the final text fully covers (reasoning ⊆ final)', () => {
    const parts = [reasoningPart('Let me check the files.'), { type: 'text' as const, text: 'streamed' }]

    const result = mergeFinalAssistantText(parts, 'Let me check the files. Everything looks good.')

    expect(result.filter(p => p.type === 'reasoning')).toHaveLength(0)
    expect(result.filter(p => p.type === 'text')).toHaveLength(1)
  })

  it('keeps a longer reasoning block when the final text is only a short prefix', () => {
    // #61447: a short final ("Done.") must NOT swallow a longer reasoning block
    // that merely starts with it.
    const parts = [
      reasoningPart(
        'Done. The root cause was a bare catch block swallowing Stripe errors. The fix adds proper error logging.'
      ),
      { type: 'text' as const, text: 'streamed' }
    ]

    const result = mergeFinalAssistantText(parts, 'Done.')

    expect(result.filter(p => p.type === 'reasoning')).toHaveLength(1)
    expect(result.filter(p => p.type === 'text')[0]).toMatchObject({ text: 'Done.' })
  })

  it('keeps non-restating reasoning', () => {
    const parts = [
      reasoningPart('I analyzed the issue and found a race condition in the event loop.'),
      { type: 'text' as const, text: 'streamed' }
    ]

    const result = mergeFinalAssistantText(parts, 'Fixed the race condition.')

    expect(result.filter(p => p.type === 'reasoning')).toHaveLength(1)
    expect(result.filter(p => p.type === 'text')).toHaveLength(1)
  })

  it('handles empty final text', () => {
    const parts = [{ type: 'text' as const, text: 'streamed' }, reasoningPart('some reasoning')]

    const result = mergeFinalAssistantText(parts, '')

    expect(result.filter(p => p.type === 'text')).toHaveLength(0)
    expect(result.filter(p => p.type === 'reasoning')).toHaveLength(1)
  })
})

describe('collectUnspokenTurnSpeech', () => {
  const assistant = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
    id,
    role: 'assistant',
    parts: text ? [{ type: 'text', text }] : [],
    ...extra
  })

  const user = (id: string, text: string): ChatMessage => ({
    id,
    role: 'user',
    parts: [{ type: 'text', text }]
  })

  it('includes sealed interim narration AND the final answer of a tool-calling turn', () => {
    const messages = [
      user('u1', 'what time is it?'),
      assistant('a1', 'Let me check the clock.', { interim: true }),
      assistant('a2', 'It is 9 PM.')
    ]

    const speech = collectUnspokenTurnSpeech(messages, null)

    expect(speech).not.toBeNull()
    expect(speech?.id).toBe('a1')
    expect(speech?.text).toBe('Let me check the clock.\n\nIt is 9 PM.')
    expect(speech?.pending).toBe(false)
  })

  it('keeps the binding id stable while later bubbles stream in', () => {
    const turnStart = [user('u1', 'go'), assistant('a1', 'Let me check.', { interim: true })]
    const first = collectUnspokenTurnSpeech(turnStart, null)

    const turnLater = [...turnStart, assistant('a2', 'Still work', { pending: true })]
    const later = collectUnspokenTurnSpeech(turnLater, null)

    expect(first?.id).toBe('a1')
    expect(later?.id).toBe('a1')
    // The earlier snapshot's text is a prefix of the later one — the live
    // session appends by length, so aggregation must be append-only.
    expect(later?.text.startsWith(first?.text ?? '')).toBe(true)
    expect(later?.pending).toBe(true)
  })

  it('starts after the last spoken message and skips hidden/empty bubbles', () => {
    const messages = [
      assistant('a0', 'Spoken last turn.'),
      user('u1', 'next'),
      assistant('a1', '', { pending: false }),
      assistant('a2', 'hidden note', { hidden: true }),
      assistant('a3', 'The real reply.')
    ]

    const speech = collectUnspokenTurnSpeech(messages, 'a0')

    expect(speech?.id).toBe('a3')
    expect(speech?.text).toBe('The real reply.')
  })

  it('reports pending from the newest assistant bubble even when it has no text yet', () => {
    const messages = [
      assistant('a1', 'Narration done.', { interim: true }),
      assistant('a2', '', { pending: true })
    ]

    const speech = collectUnspokenTurnSpeech(messages, null)

    expect(speech?.id).toBe('a1')
    expect(speech?.text).toBe('Narration done.')
    expect(speech?.pending).toBe(true)
  })

  it('returns null when everything is spoken or there is no assistant text', () => {
    expect(collectUnspokenTurnSpeech([], null)).toBeNull()
    expect(collectUnspokenTurnSpeech([assistant('a1', 'Done.')], 'a1')).toBeNull()
    expect(collectUnspokenTurnSpeech([user('u1', 'hello'), assistant('a1', '')], null)).toBeNull()
  })
})
