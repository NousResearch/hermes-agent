import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import type { ChatMessage, ChatMessagePart } from './chat-messages'
import {
  appendAssistantTextPart,
  appendReasoningPart,
  chatMessageText,
  collectUnspokenTurnSpeech,
  completeOpenTimelineParts,
  mergeFinalAssistantText,
  preserveLocalAssistantErrors,
  reasoningPart,
  renderMediaTags,
  sealOpenToolParts,
  stripPendingClarifyProjectionForCache,
  toChatMessages,
  upsertToolPart,
  withUniqueToolCallIdsWithinMessage
} from './chat-messages'

const toolCallPart = (toolCallId: string): ChatMessagePart =>
  ({
    type: 'tool-call' as const,
    toolCallId,
    toolName: 'read_file',
    args: {} as never,
    argsText: '{}'
  }) as ChatMessagePart

const assistantWith = (parts: ChatMessagePart[]): ChatMessage =>
  ({ id: 'm1', role: 'assistant', parts, timestamp: 0 }) as unknown as ChatMessage

describe('withUniqueToolCallIdsWithinMessage', () => {
  it('renames a duplicate toolCallId within one message (#87857)', () => {
    const message = assistantWith([toolCallPart('call_x'), toolCallPart('call_x')])

    const result = withUniqueToolCallIdsWithinMessage(message)

    const ids = result.parts
      .filter((part): part is Extract<ChatMessagePart, { type: 'tool-call' }> => part.type === 'tool-call')
      .map(part => part.toolCallId)

    expect(new Set(ids).size).toBe(ids.length)
    expect(ids[0]).toBe('call_x')
    expect(ids[1]).not.toBe('call_x')
  })

  it('returns the same reference when there is no duplicate', () => {
    const message = assistantWith([toolCallPart('call_a'), toolCallPart('call_b')])

    expect(withUniqueToolCallIdsWithinMessage(message)).toBe(message)
  })

  it('does not touch parts without a toolCallId', () => {
    const textPart = { type: 'text' as const, text: 'hi' } as ChatMessagePart
    const message = assistantWith([textPart, toolCallPart('call_a')])

    expect(withUniqueToolCallIdsWithinMessage(message)).toBe(message)
  })
})

describe('toChatMessages', () => {
  it('rebuilds the full command from a gateway tool row carrying args', () => {
    // Gateway watch-window hydration projects tool rows as
    // {role:'tool', name, context, args?}. `context` is an 80-char preview;
    // the backend also ships the full args, and the part must carry them so
    // the expanded `$` transcript shows the whole command.
    const longCommand = `echo ${'x'.repeat(200)}`

    const messages = toChatMessages([
      { role: 'user', content: 'run it', timestamp: 1 },
      {
        role: 'tool',
        name: 'terminal',
        content: '',
        context: `${longCommand.slice(0, 79)}…`,
        args: { command: longCommand },
        timestamp: 2
      }
    ])

    const toolPart = messages.flatMap(m => m.parts).find(part => part.type === 'tool-call')

    expect(toolPart).toBeDefined()
    expect((toolPart as { args: { command?: string } }).args.command).toBe(longCommand)
  })

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
    expect(messages[0].timestamp).toBe(1)
    expect(messages[0].parts.map(p => p.timestamp)).toEqual([1, 2, 3])
  })

  it('starts a hydrated bubble at an earlier leading tool call', () => {
    const messages = toChatMessages([
      {
        role: 'assistant',
        content: '',
        timestamp: 1,
        tool_calls: [{ id: 'tc', function: { name: 'terminal', arguments: '{}' } }]
      },
      { role: 'tool', tool_call_id: 'tc', content: 'ok', timestamp: 2 },
      { role: 'assistant', content: 'Done.', timestamp: 3 }
    ])

    expect(messages).toHaveLength(1)
    expect(messages[0].timestamp).toBe(1)
    expect(messages[0].parts.map(part => part.timestamp)).toEqual([1, 3])
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
    expect(assistantMessages[0].timestamp).toBe(2)
    expect(assistantMessages[0].parts.map(part => part.timestamp)).toEqual([2, 2, 4, 4, 6])
    expect(assistantMessages[0].parts.filter(part => part.type === 'tool-call').map(part => part.completedAt)).toEqual([
      3, 5
    ])
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

  it('lifts @image directive lines into attachmentRefs instead of inline text', () => {
    const [message] = toChatMessages([
      {
        role: 'user',
        content: '@image:/tmp/cat.png\nwhat is in this photo?',
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('what is in this photo?')
    expect((message as { attachmentRefs?: string[] }).attachmentRefs).toEqual(['@image:/tmp/cat.png'])
  })

  it('keeps a user turn that carried only an attached image (no caption)', () => {
    const [message] = toChatMessages([
      {
        role: 'user',
        content: '@image:/tmp/cat.png',
        timestamp: 1
      }
    ])

    // The bubble has no visible text, but must survive the empty-turn filter
    // because it carries attachment refs — otherwise a stand-alone attachment
    // vanishes from the transcript after a session switch / restart.
    expect(chatMessageText(message)).toBe('')
    expect((message as { attachmentRefs?: string[] }).attachmentRefs).toEqual(['@image:/tmp/cat.png'])
  })

  it('renders a native-vision turn as caption plus thumbnail, not raw placeholder text', () => {
    // How a turn sent to a natively-vision-capable model comes back out of the
    // session store: a backtick-quoted ref (the path has spaces) and the
    // `[screenshot]` stand-in left by flattening the parts list.
    const ref = '@image:`/Users/me/Library/Application Support/Hermes/composer-images/a.png`'

    const [message] = toChatMessages([
      {
        role: 'user',
        content: `${ref}\nwhat is in this photo?\n[screenshot]`,
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('what is in this photo?')
    expect((message as { attachmentRefs?: string[] }).attachmentRefs).toEqual([ref])
  })

  it('leaves a plain user prompt without attachment refs untouched', () => {
    const [message] = toChatMessages([
      {
        role: 'user',
        content: 'just a question',
        timestamp: 1
      }
    ])

    expect((message as { attachmentRefs?: string[] }).attachmentRefs).toBeUndefined()
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

  it('leaves an inline @ ref in place instead of hoisting a duplicate', () => {
    const [message] = toChatMessages([
      {
        role: 'user',
        content:
          'summarize @file:`src/main.ts` for me\n\n--- Attached Context ---\n\n📄 @file:`src/main.ts` (10 tokens)\n```ts\nconst x = 1\n```',
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('summarize @file:`src/main.ts` for me')
  })

  it('never paints redirect scaffolding as an assistant bubble', () => {
    // What the desktop actually receives after a mid-stream steer: the runtime
    // keeps the interrupt scaffolding in a server-only api_content sidecar
    // (never shipped to the client) so content is already clean, and marks a
    // prose-free checkpoint display_kind:'hidden'. The transcript must show the
    // partial reply and the user's correction — never
    // "[This response was interrupted by a user correction.]".
    const messages = toChatMessages([
      { role: 'user', content: 'go', timestamp: 1 },
      {
        role: 'assistant',
        content: 'Hey. I was mid-Figma MCP fix when we paused.',
        timestamp: 2
      },
      { role: 'user', content: 'i love you', timestamp: 3 },
      {
        // Nothing had reached the screen — checkpoint exists only for the model.
        role: 'assistant',
        content: '[This response was interrupted by a user correction.]',
        display_kind: 'hidden',
        timestamp: 4
      },
      { role: 'user', content: 'keep going', timestamp: 5 }
    ])

    expect(messages.map(chatMessageText)).toEqual([
      'go',
      'Hey. I was mid-Figma MCP fix when we paused.',
      'i love you',
      'keep going'
    ])

    for (const message of messages) {
      expect(chatMessageText(message)).not.toContain('This response was interrupted')
      expect(chatMessageText(message)).not.toContain('Visible response before the interruption')
      expect(chatMessageText(message)).not.toContain('Context from the interrupted assistant response')
    }
  })

  it('projects persisted composite compaction carriers to their live user turn', () => {
    const messages = toChatMessages([
      {
        id: 71,
        role: 'user',
        content: 'internal summary scaffold\n\nREAL ASK',
        display_content: 'REAL ASK',
        timestamp: 1
      },
      {
        id: 72,
        role: 'user',
        content: 'prior live ask\n\ninternal summary scaffold',
        display_content: 'prior live ask',
        timestamp: 2
      }
    ])

    expect(messages.map(chatMessageText)).toEqual(['REAL ASK', 'prior live ask'])
    expect(messages.map(message => message.rowId)).toEqual([71, 72])
  })

  it('projects durable timeline kinds without inspecting their text', () => {
    const messages = toChatMessages([
      { role: 'user', content: 'real user turn', timestamp: 1 },
      { role: 'assistant', content: 'real assistant reply', timestamp: 2 },
      {
        role: 'user',
        content: 'opaque compaction payload',
        display_kind: 'hidden',
        timestamp: 3
      },
      {
        role: 'user',
        content: 'opaque model context payload',
        display_kind: 'model_switch',
        timestamp: 4
      },
      {
        role: 'user',
        content: 'opaque delegation context payload',
        display_kind: 'async_delegation_complete',
        timestamp: 5
      },
      {
        role: 'user',
        content: '[System note: Your previous turn was interrupted mid-run…]\n\noriginal prompt',
        display_kind: 'auto_continue',
        timestamp: 6
      },
      {
        role: 'user',
        content: "[System: The user has changed the assistant's personality…]",
        display_kind: 'personality_switch',
        timestamp: 7
      }
    ])

    expect(messages.map(message => message.role)).toEqual(['user', 'assistant', 'system', 'system', 'system', 'system'])
    expect(messages.map(chatMessageText)).toEqual([
      'real user turn',
      'real assistant reply',
      'model changed',
      'background agent work finished',
      'resumed interrupted turn',
      'personality changed'
    ])
  })

  // A backend older than this app serves display_metadata as unparsed JSON
  // text. Indexing into that string used to throw and fail the whole resume.
  it.each([
    ['an object', { delegation_id: 'deleg_1', task_count: 2 }, '2 background agents finished'],
    ['JSON text', JSON.stringify({ delegation_id: 'deleg_1', task_count: 1 }), '1 background agent finished'],
    ['unparseable text', '{not-json', 'background agent work finished'],
    ['text that is not an object', '"deleg_1"', 'background agent work finished'],
    ['a missing task count', { delegation_id: 'deleg_1' }, 'background agent work finished']
  ])('labels a delegation event given %s', (_case, displayMetadata, expected) => {
    const read = () =>
      toChatMessages([
        {
          role: 'user',
          content: 'opaque delegation context payload',
          display_kind: 'async_delegation_complete',
          display_metadata: displayMetadata as SessionMessage['display_metadata'],
          timestamp: 1
        }
      ])

    expect(read).not.toThrow()
    expect(chatMessageText(read()[0])).toBe(expected)
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

  it('keeps bare paths intact when they contain spaces', () => {
    expect(
      renderMediaTags(
        'MEDIA:/Users/richardtang/MusicProduction/Ricktai/2026-08-29 - Not Today/audio/NotToday.wav'
      )
    ).toBe(
      '[Audio: NotToday.wav](#media:%2FUsers%2Frichardtang%2FMusicProduction%2FRicktai%2F2026-08-29%20-%20Not%20Today%2Faudio%2FNotToday.wav)'
    )
    // spaces + non-media extension → file card path survives too
    expect(renderMediaTags('MEDIA:/tmp/my folder/notes.lrc')).toBe(
      '[File: notes.lrc](#media:%2Ftmp%2Fmy%20folder%2Fnotes.lrc)'
    )
  })

  it('strips trailing sentence punctuation only from bare paths', () => {
    expect(renderMediaTags('MEDIA:/tmp/voice.mp3.')).toBe('[Audio: voice.mp3](#media:%2Ftmp%2Fvoice.mp3)')
    expect(renderMediaTags('MEDIA:/tmp/voice.mp3,')).toBe('[Audio: voice.mp3](#media:%2Ftmp%2Fvoice.mp3)')
    // a quoted path is explicit: punctuation inside quotes is preserved
    expect(renderMediaTags('MEDIA:"/tmp/file (1).mp4"')).toBe(
      '[Video: file (1).mp4](#media:%2Ftmp%2Ffile%20(1).mp4)'
    )
    // a bare path ending in ")" is untouched — only [.,;:] is stripped
    expect(renderMediaTags('MEDIA:/tmp/file (1).mp4')).toBe(
      '[Video: file (1).mp4](#media:%2Ftmp%2Ffile%20(1).mp4)'
    )
  })

  it('does not swallow same-line prose after a bare MEDIA tag', () => {
    expect(renderMediaTags('MEDIA:/a/b.mp4 and here is the rest')).toBe(
      '[Video: b.mp4](#media:%2Fa%2Fb.mp4) and here is the rest'
    )
    expect(renderMediaTags('MEDIA:/tmp/voice.mp3 — see below')).toBe(
      '[Audio: voice.mp3](#media:%2Ftmp%2Fvoice.mp3) — see below'
    )
    expect(renderMediaTags('MEDIA:"/tmp/file (1).mp4" and prose')).toBe(
      '[Video: file (1).mp4](#media:%2Ftmp%2Ffile%20(1).mp4) and prose'
    )
  })
})

describe('interleaved reasoning/text boundaries', () => {
  it('preserves text → reasoning → text as three ordered activities', () => {
    let parts: ChatMessagePart[] = appendAssistantTextPart([], 'Let me ', 1)
    parts = appendReasoningPart(parts, 'checking the file...', 2)
    parts = appendAssistantTextPart(parts, 'verify the full file is correct:', 3)

    expect(parts.map(p => p.type)).toEqual(['text', 'reasoning', 'text'])
    expect(parts.map(p => p.timestamp)).toEqual([1, 2, 3])
    expect(parts.map(p => p.completedAt)).toEqual([2, 3, undefined])
  })

  it('preserves reasoning → text → reasoning as three ordered activities', () => {
    let parts: ChatMessagePart[] = appendReasoningPart([], 'first thought ', 1)
    parts = appendAssistantTextPart(parts, 'Working on it.', 2)
    parts = appendReasoningPart(parts, 'second thought', 3)

    expect(parts.map(p => p.type)).toEqual(['reasoning', 'text', 'reasoning'])
    expect(parts.map(p => p.timestamp)).toEqual([1, 2, 3])
    expect(parts.map(p => p.completedAt)).toEqual([2, 3, undefined])
  })

  it('starts a fresh text part after a tool call (segment boundary)', () => {
    let parts: ChatMessagePart[] = appendAssistantTextPart([], 'Let me check.', 10.125)
    parts = upsertToolPart(parts, { name: 'read_file', tool_id: 'tc-1' }, 'running', 11.25)
    parts = appendAssistantTextPart(parts, 'Now editing.', 12.5)

    expect(parts.map(p => p.type)).toEqual(['text', 'tool-call', 'text'])
    expect((parts[0] as { text: string }).text).toBe('Let me check.')
    expect((parts[2] as { text: string }).text).toBe('Now editing.')
    expect(parts.map(p => p.timestamp)).toEqual([10.125, 11.25, 12.5])
    expect(parts[0].completedAt).toBe(11.25)
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
  it('preserves richer live boundaries when the durable row has only its completion time', () => {
    const durable = toChatMessages([{ role: 'assistant', content: 'Done.', timestamp: 3 }])

    const live: ChatMessage[] = [
      {
        completedAt: 3,
        id: 'assistant-stream',
        parts: [{ completedAt: 3, text: 'Done.', timestamp: 1, type: 'text' }],
        role: 'assistant',
        timestamp: 1
      }
    ]

    const [message] = preserveLocalAssistantErrors(durable, live)

    expect([message.timestamp, message.completedAt]).toEqual([1, 3])
    expect(message.parts[0]).toMatchObject({ completedAt: 3, timestamp: 1, type: 'text' })
  })

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
  it('preserves call time through progress and records completion time', () => {
    const started = upsertToolPart([], { name: 'read_file', tool_id: 'call-1' }, 'running', 100.125)

    const progressed = upsertToolPart(
      started,
      { name: 'read_file', preview: 'still reading', tool_id: 'call-1' },
      'running',
      101.5
    )

    const completed = upsertToolPart(
      progressed,
      { name: 'read_file', result: { content: 'done' }, tool_id: 'call-1' },
      'complete',
      102.875
    )

    expect(completed).toHaveLength(1)
    expect(completed[0].timestamp).toBe(100.125)
    expect(completed[0].completedAt).toBe(102.875)
  })

  it('closes active commentary when a tool starts', () => {
    const text = appendAssistantTextPart([], 'Checking first.', 20)
    const withTool = upsertToolPart(text, { name: 'read_file', tool_id: 'call-2' }, 'running', 21)

    expect(withTool[0].completedAt).toBe(21)
  })

  it('closes active commentary when only a tool completion arrives', () => {
    const text = appendAssistantTextPart([], 'Checking first.', 20)
    const withTool = upsertToolPart(text, { name: 'read_file', tool_id: 'call-2' }, 'complete', 21)

    expect(withTool[0].completedAt).toBe(21)
  })

  it('closes unfinished timeline segments when a turn completes', () => {
    const parts = appendAssistantTextPart([], 'Done.', 30.25)

    expect(completeOpenTimelineParts(parts, 31.75)[0].completedAt).toBe(31.75)
  })

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
      (part): part is Extract<ChatMessagePart, { type: 'tool-call' }> =>
        part.type === 'tool-call' && part.toolName === 'terminal'
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
  it('keeps confirmed text-reasoning-text boundaries in the final response', () => {
    let parts: ChatMessagePart[] = appendAssistantTextPart([], 'First. ', 1)
    parts = appendReasoningPart(parts, 'Think.', 2)
    parts = appendAssistantTextPart(parts, 'Last.', 3)

    const result = mergeFinalAssistantText(parts, 'First. Last.', 4)

    expect(result.map(part => part.type)).toEqual(['text', 'reasoning', 'text'])
    expect(result.map(part => part.timestamp)).toEqual([1, 2, 3])
  })

  it('timestamps completion-only text when no streamed text preceded it', () => {
    const result = mergeFinalAssistantText([], 'final answer', 12.5)

    expect(result[0]).toMatchObject({ text: 'final answer', timestamp: 12.5, type: 'text' })
  })

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

  it('does not erase streamed text when the final completion is empty (#95514)', () => {
    const parts = [{ type: 'text' as const, text: 'streamed' }, reasoningPart('some reasoning')]

    const result = mergeFinalAssistantText(parts, '')

    expect(result.filter(p => p.type === 'text')).toHaveLength(1)
    expect(result.filter(p => p.type === 'text')[0]).toMatchObject({ text: 'streamed' })
    expect(result.filter(p => p.type === 'reasoning')).toHaveLength(1)
  })

  it('treats whitespace-only final text as non-authoritative (#95514)', () => {
    const parts = [{ type: 'text' as const, text: 'already on screen' }]

    const result = mergeFinalAssistantText(parts, '   \n\t')

    expect(result.filter(p => p.type === 'text')).toHaveLength(1)
    expect(result.filter(p => p.type === 'text')[0]).toMatchObject({ text: 'already on screen' })
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
    const messages = [assistant('a1', 'Narration done.', { interim: true }), assistant('a2', '', { pending: true })]

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

describe('stripPendingClarifyProjectionForCache', () => {
  const clarifyPart = (toolCallId: string): ChatMessagePart => ({
    type: 'tool-call',
    toolCallId,
    toolName: 'clarify',
    args: { choices: ['a'], question: 'Pick' },
    argsText: '{"question":"Pick","choices":["a"]}'
  })

  it('drops a synthetic request-id-only clarify row from the durable cache', () => {
    const messages: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'choose' }] },
      { id: 'synthetic', role: 'assistant', parts: [clarifyPart('req-1')], pending: true }
    ]

    expect(stripPendingClarifyProjectionForCache(messages, 'req-1')).toEqual([messages[0]])
  })

  it('keeps a provider-authored clarify in position but strips its local running bit', () => {
    const messages: ChatMessage[] = [
      {
        id: 'provider',
        role: 'assistant',
        parts: [{ type: 'text', text: 'Choose.' }, clarifyPart('call-provider')],
        pending: true
      }
    ]

    const [cached] = stripPendingClarifyProjectionForCache(messages, 'req-1')
    expect(cached.pending).toBe(false)
    expect(cached.parts.map(part => part.type)).toEqual(['text', 'tool-call'])
  })
})

describe('sealOpenToolParts', () => {
  const toolPart = (over: Partial<ChatMessagePart> = {}): ChatMessagePart =>
    ({
      type: 'tool-call',
      toolCallId: 'call-1',
      toolName: 'terminal',
      args: {},
      argsText: '{}',
      ...over
    }) as ChatMessagePart

  const assistantWithParts = (parts: ChatMessagePart[], over: Partial<ChatMessage> = {}): ChatMessage =>
    ({
      id: 'a1',
      role: 'assistant',
      parts,
      ...over
    }) as ChatMessage

  it('seals open tool-call parts in settled assistant messages', () => {
    const messages = [assistantWithParts([toolPart()])]

    const next = sealOpenToolParts(messages)

    expect(next[0].parts[0]).toHaveProperty('result')
  })

  it('leaves already-completed tool parts untouched', () => {
    const done = toolPart({ result: { code: 0 } })
    const messages = [assistantWithParts([done])]

    const next = sealOpenToolParts(messages)

    expect(next[0].parts[0]).toBe(done)
  })

  it('leaves pending messages alone', () => {
    const messages = [assistantWithParts([toolPart()], { pending: true })]

    const next = sealOpenToolParts(messages)

    expect(next[0].parts[0]).not.toHaveProperty('result')
  })

  it('leaves non-tool parts untouched', () => {
    const text = { type: 'text', text: 'hello' } as ChatMessagePart
    const messages = [assistantWithParts([text, toolPart()])]

    const next = sealOpenToolParts(messages)

    expect(next[0].parts[0]).toBe(text)
    expect(next[0].parts[1]).toHaveProperty('result')
  })

  it('returns the same array reference when nothing needs sealing', () => {
    const done = toolPart({ result: { code: 0 } })
    const messages = [assistantWithParts([done])]

    expect(sealOpenToolParts(messages)).toBe(messages)
  })
})
