import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import type { SessionResumeResult } from '@/types/hermes'

import {
  appendLiveSessionProjection,
  dedupeInflightUserAgainstTranscript,
  removeRepresentedLocalLiveProjection
} from './live-session-projection'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

describe('appendLiveSessionProjection', () => {
  it('hydrates client identity and authored time from live send envelopes', () => {
    const inflight = appendLiveSessionProjection([], {
      session_id: 'runtime-live',
      inflight: {
        user: 'live prompt',
        assistant: '',
        streaming: true,
        client_message_id: 'client-live',
        user_timestamp: 1_790_594_548.125
      }
    })

    const queued = appendLiveSessionProjection([], {
      session_id: 'runtime-queued',
      queued: {
        user: 'queued prompt',
        client_message_id: 'client-queued',
        user_timestamp: 1_790_594_549.25
      }
    })

    expect(inflight.find(message => message.role === 'user')).toMatchObject({
      clientMessageId: 'client-live',
      timestamp: 1_790_594_548.125
    })
    expect(queued.find(message => message.role === 'user')).toMatchObject({
      clientMessageId: 'client-queued',
      timestamp: 1_790_594_549.25
    })
  })

  it('uses client identity before repeated prose for an ordinary in-flight user row', () => {
    const stored = [{ ...msg('stored-user', 'user', 'same repeated prompt'), clientMessageId: 'client-old', rowId: 10 }]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'same repeated prompt',
        assistant: '',
        streaming: true,
        display_metadata: { client_message_id: 'client-new' }
      }
    })

    expect(restored.map(message => [message.role, message.clientMessageId])).toEqual([
      ['user', 'client-old'],
      ['user', 'client-new'],
      ['assistant', undefined]
    ])
  })

  // A synthetic starting prompt keeps the display typing its persisted row
  // will get: on reconnect it renders as the same timeline event as history,
  // never as a user bubble; a real user quoting the marker text stays a user
  // bubble because the gateway typed nothing (#112144).
  it('renders a typed synthetic in-flight prompt as its timeline event, not a user bubble', () => {
    const typed = appendLiveSessionProjection([], {
      session_id: 'runtime-1',
      inflight: {
        user: '[IMPORTANT: Background process finished] fixture',
        display_kind: 'process_complete',
        display_metadata: { display_text: 'Background Process Finished: fixture' },
        assistant: '',
        streaming: true
      }
    })

    const inflightRow = (message: ChatMessage) => message.id === 'user-inflight-runtime-1'

    expect(typed.filter(inflightRow).map(message => [message.role, chatMessageText(message)])).toEqual([
      ['system', 'Background Process Finished: fixture']
    ])

    const quoted = appendLiveSessionProjection([], {
      session_id: 'runtime-1',
      inflight: { user: '[IMPORTANT: Background process finished] fixture', assistant: '', streaming: true }
    })

    expect(quoted.filter(inflightRow).map(message => [message.role, chatMessageText(message)])).toEqual([
      ['user', '[IMPORTANT: Background process finished] fixture']
    ])
  })

  it('omits a hidden synthetic in-flight prompt but keeps its streaming reply', () => {
    const restored = appendLiveSessionProjection([], {
      session_id: 'runtime-1',
      inflight: { user: 'scaffolding the model must see', display_kind: 'hidden', assistant: 'On it.', streaming: true }
    })

    expect(restored.map(message => [message.role, chatMessageText(message)])).toEqual([['assistant', 'On it.']])
  })
  // Corrections typed while a turn ran are their own user bubbles on the same
  // turn, ordered by ARRIVAL. Without boundary offsets (older gateway) the
  // whole dump precedes them — never the old prompt → corrections → reply
  // order that spliced them above output the user had already read (#73793).
  it('projects mid-turn redirect corrections after the assistant output that predates them', () => {
    const restored = appendLiveSessionProjection([], {
      session_id: 'runtime-1',
      inflight: {
        user: 'remove the session counts',
        corrections: ['hurry up', 'and the worktree ones'],
        assistant: 'Moving.',
        streaming: true
      }
    })

    expect(restored.map(message => message.parts.map(part => ('text' in part ? part.text : '')).join(''))).toEqual([
      'remove the session counts',
      'Moving.',
      'hurry up',
      'and the worktree ones'
    ])
  })

  // With correction_offsets the flat dump is split at each accepted-correction
  // boundary, so every correction lands after exactly the output it followed
  // and before the output it redirected — arrival order end to end (#73793).
  it('interleaves corrections into the assistant dump at their arrival offsets', () => {
    const restored = appendLiveSessionProjection([], {
      session_id: 'runtime-1',
      inflight: {
        user: 'remove the session counts',
        corrections: ['hurry up', 'and the worktree ones'],
        correction_offsets: [7, 13],
        assistant: 'Moving.Still.Done soon.',
        streaming: true
      }
    })

    expect(restored.map(message => message.parts.map(part => ('text' in part ? part.text : '')).join(''))).toEqual([
      'remove the session counts',
      'Moving.',
      'hurry up',
      'Still.',
      'and the worktree ones',
      'Done soon.'
    ])
    expect(restored.map(message => message.role)).toEqual([
      'user',
      'assistant',
      'user',
      'assistant',
      'user',
      'assistant'
    ])
    // Only the live tail streams; sealed pre-correction segments are settled.
    expect(restored.at(-1)).toMatchObject({ id: 'assistant-stream-runtime-1', pending: true })
    expect(restored[1]).toMatchObject({ pending: false, interim: true })
    expect(restored[3]).toMatchObject({ pending: false, interim: true })
  })

  it('keeps the live stream row even when every offset points at the dump tail', () => {
    const restored = appendLiveSessionProjection([], {
      session_id: 'runtime-1',
      inflight: {
        user: 'prompt',
        corrections: ['nudge'],
        correction_offsets: [4],
        assistant: 'text',
        streaming: true
      }
    })

    // The whole dump precedes the correction, and the still-streaming turn
    // keeps its (empty for now) live row at the tail so future deltas land
    // BELOW the correction, not above it.
    expect(restored.map(message => message.parts.map(part => ('text' in part ? part.text : '')).join(''))).toEqual([
      'prompt',
      'text',
      'nudge',
      ''
    ])
    expect(restored.at(-1)).toMatchObject({ id: 'assistant-stream-runtime-1', pending: true })
    expect(restored.at(-1)?.role).toBe('assistant')
  })

  it('does not re-project a correction the transcript already persisted', () => {
    const stored = [msg('stored-user', 'user', 'remove the session counts'), msg('stored-fix', 'user', 'hurry up')]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'remove the session counts',
        corrections: ['hurry up'],
        assistant: 'Moving.',
        streaming: true
      }
    })

    expect(restored.filter(message => message.role === 'user').map(message => message.id)).toEqual([
      'stored-user',
      'stored-fix'
    ])
  })

  it('does not duplicate the inflight user when the persisted turn carries @image refs', () => {
    // By the time a stored transcript reaches appendLiveSessionProjection it
    // has already been run through toChatMessages, so the @image directive has
    // been lifted into attachmentRefs and the visible text is the bare caption.
    const stored = [
      msg('stored-user', 'user', 'current running prompt', {
        attachmentRefs: ['@image:/tmp/cat.png']
      }),
      msg('stored-assistant', 'assistant', 'earlier answer')
    ]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'current running prompt',
        assistant: 'partial answer',
        streaming: true
      }
    })

    // The persisted user already carries the same visible text (the attachment
    // lives in attachmentRefs on both sides), so the inflight *user* projection
    // must be suppressed — exactly one user row, no duplicated bubble stacked
    // on top of the persisted one. The live assistant tail is still projected.
    const userRows = restored.filter(message => message.role === 'user')
    expect(userRows).toHaveLength(1)
    expect(userRows[0].id).toBe('stored-user')

    const userText = userRows[0].parts.map(part => ('text' in part ? part.text : '')).join('')

    expect(userText).toBe('current running prompt')
  })

  it('restores the running turn and accepted queued prompt after a renderer restart', () => {
    const stored = [msg('stored-user', 'user', 'earlier'), msg('stored-assistant', 'assistant', 'earlier answer')]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'current prompt',
        assistant: 'partial answer',
        streaming: true
      },
      queued: { user: 'newest prompt' }
    })

    expect(restored.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'assistant', 'user'])
    expect(restored.map(message => message.parts.map(part => ('text' in part ? part.text : '')).join(''))).toEqual([
      'earlier',
      'earlier answer',
      'current prompt',
      'partial answer',
      'newest prompt'
    ])
    expect(restored[3]).toMatchObject({ id: 'assistant-stream-runtime-1', pending: true })
  })

  it('does not duplicate a persisted inflight user after consecutive canceled user turns', () => {
    const stored = [
      msg('stored-user-1', 'user', 'canceled prompt one'),
      msg('stored-user-2', 'user', 'canceled prompt two'),
      msg('stored-user-3', 'user', 'current running prompt')
    ]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'current running prompt',
        assistant: 'partial answer',
        streaming: true
      }
    })

    expect(restored.map(message => message.role)).toEqual(['user', 'user', 'user', 'assistant'])
    expect(restored.map(message => message.parts.map(part => ('text' in part ? part.text : '')).join(''))).toEqual([
      'canceled prompt one',
      'canceled prompt two',
      'current running prompt',
      'partial answer'
    ])
    expect(restored[3]).toMatchObject({ id: 'assistant-stream-runtime-1', pending: true })
  })

  it('preserves the original array when no live projection exists', () => {
    const stored = [msg('stored-user', 'user', 'earlier')]

    expect(appendLiveSessionProjection(stored, { session_id: 'runtime-1' })).toBe(stored)
  })

  it('does not sandwich a structured mid-turn row with the inflight flat dump (#76444)', () => {
    const stored: ChatMessage[] = [
      msg('stored-user', 'user', 'do the work'),
      {
        id: 'live-assistant',
        role: 'assistant',
        pending: true,
        parts: [
          { type: 'reasoning', text: 'thinking about tools' },
          { type: 'tool-call', toolCallId: 'c1', toolName: 'terminal', args: {} },
          { type: 'text', text: 'partial' }
        ]
      }
    ]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'do the work',
        // Flat dump includes thinking chatter + tool narration — longer than
        // the answer text alone, which is how the sandwich used to grow.
        assistant: 'thinking about tools\nRan terminal\npartial and more dump',
        streaming: true
      }
    })

    const assistants = restored.filter(message => message.role === 'assistant')
    expect(assistants).toHaveLength(1)
    expect(assistants[0].id).toBe('live-assistant')
    expect(assistants[0].parts.some(part => part.type === 'reasoning')).toBe(true)
    expect(assistants[0].parts.some(part => part.type === 'tool-call')).toBe(true)
    // Answer text stays the structured row's text, not the dump.
    expect(
      assistants[0].parts.filter(part => part.type === 'text').map(part => ('text' in part ? part.text : ''))
    ).toEqual(['partial'])
  })

  it('still projects inflight when only a completed historical tool reply has structure', () => {
    // Older completed assistants keep reasoning/tool parts in the full
    // transcript; they must not suppress a new turn's text projection.
    const stored: ChatMessage[] = [
      msg('old-user', 'user', 'previous task'),
      {
        id: 'old-assistant',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'old', toolName: 'terminal', args: {} },
          { type: 'text', text: 'done earlier' }
        ]
      },
      msg('new-user', 'user', 'new task')
    ]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      inflight: {
        user: 'new task',
        assistant: 'working on it',
        streaming: true
      }
    })

    expect(restored.map(message => message.id)).toContain('assistant-stream-runtime-1')
    expect(restored.at(-1)).toMatchObject({
      id: 'assistant-stream-runtime-1',
      pending: true
    })
  })

  // #121122: switching away mid-turn and back. REST already holds this
  // turn's partial assistant row (text + tool blocks committed as the turn
  // progressed) while `inflight` still streams the fuller dump. Appending
  // the dump paints the turn twice: the frozen partial with its action bar
  // plus the live copy repeating it. Fold the dump into the tail row.
  it('folds a still-streaming dump into the same-turn committed partial instead of doubling it', () => {
    const stored: ChatMessage[] = [
      msg('1-user', 'user', 'Fais X'),
      {
        id: '111-1-assistant',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'call-1', toolName: 'terminal', result: 'done' },
          { type: 'text', text: 'Tu as raison. Je les regarde' }
        ],
        timestamp: 111,
        rowId: 13
      } as ChatMessage
    ]

    const inflight = {
      user: 'Fais X',
      assistant: 'Tu as raison. Je les regarde vraiment cette fois. + more',
      streaming: true
    }

    const restored = appendLiveSessionProjection(stored, { session_id: 's1', turn_started_at: 100, inflight })

    const assistants = restored.filter(message => message.role === 'assistant')
    expect(assistants).toHaveLength(1)
    expect(assistants[0].id).toBe('assistant-stream-s1')
    expect(assistants[0].pending).toBe(true)
    // The committed row's tool structure and row id survive; the fuller live text wins.
    expect(assistants[0].parts.some(part => part.type === 'tool-call')).toBe(true)
    expect(assistants[0].rowId).toBe(13)
    expect(chatMessageText(assistants[0])).toBe('Tu as raison. Je les regarde vraiment cette fois. + more')

    // Without turn_started_at (older runtime) the tail may be the PREVIOUS
    // turn's answer to a resent prompt: keep both rows rather than drop it.
    const untimed = appendLiveSessionProjection(stored, { session_id: 's1', inflight })

    expect(untimed.filter(message => message.role === 'assistant')).toHaveLength(2)
  })
})

const runningProjection = (user: string): SessionResumeResult =>
  ({
    session_id: 'runtime-1',
    session_key: 'stored-1',
    resumed: 'stored-1',
    message_count: 2,
    messages: [],
    running: true,
    inflight: { user, assistant: 'partial answer', streaming: true }
  }) as SessionResumeResult

describe('dedupeInflightUserAgainstTranscript', () => {
  it('retains the in-flight user source only when it already exists after the runtime anchor', () => {
    const runtime = [
      msg('runtime-user', 'user', 'earlier prompt', { timestamp: 1 }),
      msg('runtime-assistant', 'assistant', 'earlier answer', { timestamp: 2 })
    ]

    const persisted = [...runtime, msg('persisted-current', 'user', 'current prompt', { timestamp: 3 })]

    const deduped = dedupeInflightUserAgainstTranscript(persisted, runtime, runningProjection('current prompt'))

    expect(deduped.inflight?.user).toBe('current prompt')
    expect(deduped.inflight?.assistant).toBe('partial answer')
  })

  it('does not let older equal prose with a different client id hide the live occurrence', () => {
    const runtime = [
      msg('runtime-user', 'user', 'earlier prompt', { timestamp: 1 }),
      msg('runtime-assistant', 'assistant', 'earlier answer', { timestamp: 2 })
    ]

    const persisted = [
      ...runtime,
      msg('persisted-repeat', 'user', 'same repeated prompt', {
        clientMessageId: 'client-old',
        rowId: 10,
        timestamp: 3
      })
    ]

    const projection = {
      ...runningProjection('same repeated prompt'),
      inflight: {
        user: 'same repeated prompt',
        assistant: 'partial answer',
        streaming: true,
        client_message_id: 'client-new'
      }
    }

    const deduped = dedupeInflightUserAgainstTranscript(persisted, runtime, projection)
    const restored = appendLiveSessionProjection(persisted, deduped)

    expect(restored.filter(message => message.role === 'user').map(message => message.clientMessageId)).toEqual([
      undefined,
      'client-old',
      'client-new'
    ])
  })

  it('preserves the assistant boundary before a queued turn when the persisted in-flight user has no delta', () => {
    const runtime = [
      msg('runtime-user', 'user', 'earlier prompt', { timestamp: 1 }),
      msg('runtime-assistant', 'assistant', 'earlier answer', { timestamp: 2 })
    ]

    const persisted = [...runtime, msg('persisted-current', 'user', 'current prompt', { timestamp: 3 })]

    const projection = {
      ...runningProjection('current prompt'),
      inflight: { user: 'current prompt', assistant: '', streaming: false },
      queued: { user: 'queued prompt' }
    }

    const deduped = dedupeInflightUserAgainstTranscript(persisted, runtime, projection)
    const restored = appendLiveSessionProjection(persisted, deduped)

    expect(restored.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'assistant', 'user'])
    expect(restored.slice(-2).map(message => message.id)).toEqual([
      'assistant-stream-runtime-1',
      'user-queued-runtime-1'
    ])
  })

  it('preserves an intentionally repeated prompt when the match is before the runtime anchor', () => {
    const runtime = [
      msg('runtime-user', 'user', 'repeat this', { timestamp: 1 }),
      msg('runtime-assistant', 'assistant', 'finished answer', { timestamp: 2 })
    ]

    const projection = runningProjection('repeat this')
    const unchanged = dedupeInflightUserAgainstTranscript(runtime, runtime, projection)

    expect(unchanged).toBe(projection)
    expect(unchanged.inflight?.user).toBe('repeat this')
  })

  it('preserves a repeated in-flight prompt when the persisted match already has an answer', () => {
    const runtime = [
      msg('runtime-user', 'user', 'earlier prompt', { timestamp: 1 }),
      msg('runtime-assistant', 'assistant', 'earlier answer', { timestamp: 2 })
    ]

    const persisted = [
      ...runtime,
      msg('persisted-repeat', 'user', 'repeat this', { timestamp: 3 }),
      msg('persisted-repeat-answer', 'assistant', 'finished repeat answer', { timestamp: 4 })
    ]

    const projection = runningProjection('repeat this')
    const unchanged = dedupeInflightUserAgainstTranscript(persisted, runtime, projection)

    expect(unchanged).toBe(projection)
    expect(unchanged.inflight?.user).toBe('repeat this')
  })
})

describe('removeRepresentedLocalLiveProjection', () => {
  it('removes only matched synthetic rows from the open local tail', () => {
    const previous = [
      msg('user-old-optimistic', 'user', 'current prompt'),
      msg('assistant-complete', 'assistant', 'finished answer'),
      msg('user-current', 'user', 'current prompt'),
      msg('assistant-stream-current', 'assistant', 'partial answer', { pending: true }),
      msg('user-queued-runtime', 'user', 'queued prompt'),
      msg('user-racing', 'user', 'new racing prompt')
    ]

    const projection = {
      ...runningProjection('current prompt'),
      queued: { user: 'queued prompt' }
    }

    const remaining = removeRepresentedLocalLiveProjection(previous, projection)

    expect(remaining.map(message => message.id)).toEqual(['user-old-optimistic', 'assistant-complete', 'user-racing'])
  })

  it('removes a local stream row whose text has advanced past the activation snapshot', () => {
    const previous = [
      msg('user-current', 'user', 'current prompt'),
      msg('assistant-stream-current', 'assistant', 'partial answer and more', { pending: true })
    ]

    const projection = runningProjection('current prompt')

    const remaining = removeRepresentedLocalLiveProjection(previous, projection)

    expect(remaining).toEqual([])
  })

  it('preserves an ambiguous text-identical local race prompt without a matching stream boundary', () => {
    const previous = [
      msg('runtime-assistant', 'assistant', 'finished answer'),
      msg('user-racing', 'user', 'repeat this')
    ]

    const projection = runningProjection('repeat this')

    expect(removeRepresentedLocalLiveProjection(previous, projection)).toBe(previous)
  })

  it('does not consume a generic racing user as the activation-owned queued row', () => {
    const previous = [
      msg('runtime-assistant', 'assistant', 'finished answer'),
      msg('user-current', 'user', 'current prompt'),
      msg('assistant-stream-current', 'assistant', 'partial answer', { pending: true }),
      msg('user-racing', 'user', 'repeat this')
    ]

    const projection = {
      ...runningProjection('current prompt'),
      queued: { user: 'repeat this' }
    }

    const remaining = removeRepresentedLocalLiveProjection(previous, projection)

    expect(remaining.map(message => message.id)).toEqual(['runtime-assistant', 'user-racing'])
  })
})
