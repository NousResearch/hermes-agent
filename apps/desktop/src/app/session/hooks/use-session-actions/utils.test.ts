import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { textWithoutReferenceLines, WIRE_REFERENCE_KINDS } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, type ChatMessagePart, chatMessageText } from '@/lib/chat-messages'
import { $approvalModes, approvalModeForProfile } from '@/store/approval-mode'
import { $desktopOnboarding } from '@/store/onboarding'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $currentBranch,
  $currentCwd,
  setCurrentBranch,
  setCurrentCwd,
  setSelectedStoredSessionId,
  workspaceCwdBelongsToSelectedSession
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import {
  appendLiveSessionProjection,
  applyRuntimeInfo,
  applyStoredSessionPreviewRuntimeInfo,
  chatMessageArraysEquivalent,
  chatMessagesEquivalent,
  chatPartsEquivalent,
  isSessionGoneError,
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages,
  sessionMatchesStoredId,
  sessionShouldHaveTranscript,
  toBranchMessages
} from './utils'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

// A live assistant row carrying the structure the gateway's text-only inflight
// snapshot cannot: reasoning and tool calls, with or without any text yet.
const streamingMsg = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({
    id,
    role: 'assistant',
    parts: [
      { type: 'reasoning', text: 'planning' },
      { type: 'tool-call', toolCallId: 'call-1', toolName: 'terminal', result: 'done' },
      ...(text ? [{ type: 'text', text } as ChatMessagePart] : [])
    ],
    pending: true,
    ...extra
  }) as ChatMessage

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

describe('applyRuntimeInfo approval mode', () => {
  beforeEach(() => {
    $approvalModes.set({})
    $activeGatewayProfile.set('work')
  })

  it('reconciles session.info against the gateway profile', () => {
    applyRuntimeInfo({ approval_mode: 'smart', desktop_contract: 3 })

    expect(approvalModeForProfile('work')).toBe('smart')
    expect(approvalModeForProfile('default')).toBe('smart')
  })
})

const initialOnboardingState = $desktopOnboarding.get()

describe('applyRuntimeInfo credential warnings', () => {
  beforeEach(() => {
    $desktopOnboarding.set({ ...initialOnboardingState, reason: null, requested: false })
  })

  afterEach(() => {
    $desktopOnboarding.set(initialOnboardingState)
  })

  it('requests setup for the exact empty-key warning returned by the server', () => {
    const warning = "No API key configured for provider 'openrouter'. First message will fail."

    applyRuntimeInfo({ credential_warning: warning })

    expect($desktopOnboarding.get()).toMatchObject({ reason: warning, requested: true })
  })

  it('ignores an auxiliary-provider warning', () => {
    applyRuntimeInfo({ credential_warning: 'OPENROUTER_API_KEY not set' })

    expect($desktopOnboarding.get()).toMatchObject({ reason: null, requested: false })
  })
})

describe('applyRuntimeInfo foreground scoping', () => {
  beforeEach(() => {
    setCurrentCwd('/main-repo')
    setCurrentBranch('main')
  })

  afterEach(() => {
    setCurrentCwd('')
    setCurrentBranch('')
  })

  it('publishes a foreground runtime into the composer atoms', () => {
    const patch = applyRuntimeInfo({ branch: 'bb/feature', cwd: '/main-repo/worktree' })

    expect($currentCwd.get()).toBe('/main-repo/worktree')
    expect($currentBranch.get()).toBe('bb/feature')
    expect(patch).toMatchObject({ branch: 'bb/feature', cwd: '/main-repo/worktree' })
  })

  it('keeps a background runtime out of the composer atoms but still returns its patch', () => {
    const patch = applyRuntimeInfo({ branch: 'bb/tile', cwd: '/other-worktree' }, { foreground: false })

    // The main pane's rail must stay on its own tree.
    expect($currentCwd.get()).toBe('/main-repo')
    expect($currentBranch.get()).toBe('main')
    // ...while the caller still gets everything it needs for its own session.
    expect(patch).toMatchObject({ branch: 'bb/tile', cwd: '/other-worktree' })
  })

  // #71254: `if (info.cwd)` treated '' as "no opinion", so a detached session
  // never released the previous project and the Files pane stayed on it forever.
  it('treats an empty runtime cwd as authoritative and releases ownership', () => {
    setSelectedStoredSessionId('session-detached')
    const patch = applyRuntimeInfo({ cwd: '' })

    expect(patch).toMatchObject({ cwd: '' })
    expect(workspaceCwdBelongsToSelectedSession()).toBe(false)
  })

  // The release must NOT blank the path: setCurrentCwd persists, so writing ''
  // would also wipe the remembered workspace that seeds $currentCwd on boot.
  it('leaves the path in place when releasing, so panes do not collapse', () => {
    setSelectedStoredSessionId('session-detached')
    applyRuntimeInfo({ cwd: '' })

    expect($currentCwd.get()).toBe('/main-repo')
  })

  it('claims ownership for the selected session when a real cwd arrives', () => {
    setSelectedStoredSessionId('session-b')
    applyRuntimeInfo({ cwd: '/project-b' })

    expect($currentCwd.get()).toBe('/project-b')
    expect(workspaceCwdBelongsToSelectedSession()).toBe(true)
  })
})

describe('applyStoredSessionPreviewRuntimeInfo workspace paint', () => {
  beforeEach(() => {
    setCurrentCwd('/previous-project')
    setSelectedStoredSessionId(null)
  })

  afterEach(() => {
    setCurrentCwd('')
    setSelectedStoredSessionId(null)
  })

  // The core of the report: cold resume paints before session.resume returns.
  it('rebinds the workspace from the selected session row before resume settles', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '/next-project', model: 'gpt' }, 'session-next')
    setSelectedStoredSessionId('session-next')

    expect($currentCwd.get()).toBe('/next-project')
    expect(workspaceCwdBelongsToSelectedSession()).toBe(true)
  })

  it('releases ownership when the selected session row reports no workspace', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '', model: 'gpt' }, 'session-detached')
    setSelectedStoredSessionId('session-detached')

    expect(workspaceCwdBelongsToSelectedSession()).toBe(false)
  })

  // Regression guard: a session outside the loaded sidebar page has no row at
  // all. Blanking $currentCwd here would drop file-tree state on every switch
  // into older history, so the path must survive and ownership carry the signal.
  it('does not blank the pane when the session row is not loaded', () => {
    applyStoredSessionPreviewRuntimeInfo(undefined, 'session-off-page')
    setSelectedStoredSessionId('session-off-page')

    expect($currentCwd.get()).toBe('/previous-project')
    expect(workspaceCwdBelongsToSelectedSession()).toBe(false)
  })

  // Regression guard: git_repo_root is documented null for non-git workspaces
  // and not-yet-backfilled rows, so it must never stand in for a real cwd —
  // doing so reads as "no workspace" and blanks a pane that was correct.
  it('uses the row cwd for a non-git workspace with no repo root', () => {
    applyStoredSessionPreviewRuntimeInfo(
      { cwd: '/plain/folder', git_repo_root: null, model: 'gpt' } as never,
      'session-nongit'
    )
    setSelectedStoredSessionId('session-nongit')

    expect($currentCwd.get()).toBe('/plain/folder')
    expect(workspaceCwdBelongsToSelectedSession()).toBe(true)
  })

  it('clears the branch label so the previous project does not leak across a switch', () => {
    setCurrentBranch('bb/previous')
    applyStoredSessionPreviewRuntimeInfo({ cwd: '/next-project', model: 'gpt' }, 'session-next')

    expect($currentBranch.get()).toBe('')
  })
})

describe('isSessionGoneError', () => {
  it('is true for 404 / session-not-found, false otherwise', () => {
    expect(isSessionGoneError(new Error('Request failed 404'))).toBe(true)
    expect(isSessionGoneError(new Error('Session not found'))).toBe(true)
    expect(isSessionGoneError(new Error('ECONNREFUSED'))).toBe(false)
    expect(isSessionGoneError(null)).toBe(false)
  })
})

describe('sessionMatchesStoredId', () => {
  it('matches on live id or lineage root', () => {
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'a')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'live', _lineage_root_id: 'root' }), 'root')).toBe(true)
    expect(sessionMatchesStoredId(session({ id: 'a' }), 'b')).toBe(false)
  })
})

describe('sessionShouldHaveTranscript', () => {
  it('is true only when the session has messages', () => {
    expect(sessionShouldHaveTranscript(session({ message_count: 3 }))).toBe(true)
    expect(sessionShouldHaveTranscript(session({ message_count: 0 }))).toBe(false)
    expect(sessionShouldHaveTranscript(undefined)).toBe(false)
  })
})

describe('toBranchMessages', () => {
  it('keeps only user/assistant turns that carry text', () => {
    const out = toBranchMessages([
      msg('u', 'user', 'hi'),
      msg('blank', 'assistant', '   '),
      msg('sys', 'system', 'ignored'),
      msg('a', 'assistant', 'hello')
    ])

    expect(out.map(b => b.source.id)).toEqual(['u', 'a'])
    expect(out[0]).toMatchObject({ content: 'hi', role: 'user' })
  })
})

describe('chatPartsEquivalent', () => {
  it('returns true for identical text parts', () => {
    const partA = { type: 'text' as const, text: 'Hello world' }
    const partB = { type: 'text' as const, text: 'Hello world' }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns false for text parts with different content', () => {
    const partA = { type: 'text' as const, text: 'Hello' }
    const partB = { type: 'text' as const, text: 'World' }

    expect(chatPartsEquivalent(partA, partB)).toBe(false)
  })

  it('returns true for identical reasoning parts', () => {
    const partA = { type: 'reasoning' as const, text: 'Thinking...' }
    const partB = { type: 'reasoning' as const, text: 'Thinking...' }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns true for tool-call parts with same identity and both have no result', () => {
    const partA = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}'
    }

    const partB = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}'
    }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns true for tool-call parts with same identity and both have results', () => {
    const partA = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}',
      result: { content: 'file data' },
      isError: false
    }

    const partB = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}',
      result: { content: 'file data' },
      isError: false
    }

    expect(chatPartsEquivalent(partA, partB)).toBe(true)
  })

  it('returns false when only one tool-call part has a result', () => {
    const partA = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}'
    }

    const partB = {
      type: 'tool-call' as const,
      toolCallId: 'tc-1',
      toolName: 'read_file',
      args: {} as never,
      argsText: '{}',
      result: { content: 'file data' },
      isError: false
    }

    expect(chatPartsEquivalent(partA, partB)).toBe(false)
  })

  it('uses reference equality fast-path for identical part objects', () => {
    const part = { type: 'text' as const, text: 'Same reference' }

    expect(chatPartsEquivalent(part, part)).toBe(true)
  })
})

describe('chatMessagesEquivalent', () => {
  it('returns true for structurally identical messages', () => {
    expect(chatMessagesEquivalent(msg('1', 'user', 'Hello'), msg('1', 'user', 'Hello'))).toBe(true)
  })

  it('returns false when text part content differs', () => {
    expect(chatMessagesEquivalent(msg('1', 'user', 'Hello'), msg('1', 'user', 'World'))).toBe(false)
  })

  it('returns false when tool result presence differs', () => {
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [{ type: 'tool-call', toolCallId: 'tc-1', toolName: 'read_file', args: {} as never, argsText: '{}' }]
    }

    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        {
          type: 'tool-call',
          toolCallId: 'tc-1',
          toolName: 'read_file',
          args: {} as never,
          argsText: '{}',
          result: { content: 'data' },
          isError: false
        }
      ]
    }

    expect(chatMessagesEquivalent(messageA, messageB)).toBe(false)
  })

  it('returns false when message IDs differ', () => {
    expect(chatMessagesEquivalent(msg('msg-1', 'user', 'Hello'), msg('msg-2', 'user', 'Hello'))).toBe(false)
  })

  it('compares large messages with embedded images structurally without JSON.stringify', () => {
    // Verifies that two structurally identical messages (that would be equal
    // via stringify) are also equal via the new cheap structural compare.
    const messageA: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'Here are the images:' },
        {
          type: 'tool-call',
          toolCallId: 'img-1',
          toolName: 'image_generate',
          args: { prompt: 'a cat' } as never,
          argsText: '{"prompt":"a cat"}',
          result: { image: 'data:image/png;base64,iVBORw0KG...(large base64)' },
          isError: false
        }
      ]
    }

    const messageB: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'Here are the images:' },
        {
          type: 'tool-call',
          toolCallId: 'img-1',
          toolName: 'image_generate',
          args: { prompt: 'a cat' } as never,
          argsText: '{"prompt":"a cat"}',
          result: { image: 'data:image/png;base64,iVBORw0KG...(large base64)' },
          isError: false
        }
      ]
    }

    // The structural compare treats these as equal (both have result defined,
    // same toolCallId/toolName), without comparing the full result object.
    expect(chatMessagesEquivalent(messageA, messageB)).toBe(true)
  })
})

describe('chatMessageArraysEquivalent', () => {
  it('returns true for identical arrays via identity fast-path', () => {
    const messages: ChatMessage[] = [msg('1', 'user', 'x')]

    expect(chatMessageArraysEquivalent(messages, messages)).toBe(true)
  })

  it('compares length and per-message equivalence', () => {
    const a = [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')]
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'y')])).toBe(true)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x')])).toBe(false)
    expect(chatMessageArraysEquivalent(a, [msg('1', 'user', 'x'), msg('2', 'assistant', 'changed')])).toBe(false)
  })
})

describe('reconcileResumeMessages', () => {
  it('returns next untouched when there is no previous transcript', () => {
    const next = [msg('1', 'user', 'hi')]
    expect(reconcileResumeMessages(next, [])).toBe(next)
  })

  it('re-grafts reasoning parts onto a matching assistant turn', () => {
    const next = [msg('a', 'assistant', 'answer')]

    const previous = [
      msg('a', 'assistant', 'answer', {
        parts: [
          { type: 'reasoning', text: 'thinking' },
          { type: 'text', text: 'answer' }
        ]
      } as Partial<ChatMessage>)
    ]

    const [out] = reconcileResumeMessages(next, previous)
    expect(out.parts.some(p => p.type === 'reasoning')).toBe(true)
  })

  it('preserves attachment refs for a matching user turn', () => {
    const next = [msg('stored-user', 'user', 'describe this image')]

    const previous = [
      msg('live-user', 'user', 'describe this image', {
        attachmentRefs: ['@image:/tmp/photo.png']
      })
    ]

    const [out] = reconcileResumeMessages(next, previous)

    expect(out.attachmentRefs).toEqual(['@image:/tmp/photo.png'])
  })

  it('does not overwrite attachment refs already present on the resumed message', () => {
    const next = [
      msg('stored-user', 'user', 'describe this image', {
        attachmentRefs: ['@image:/tmp/authoritative.png']
      })
    ]

    const previous = [
      msg('live-user', 'user', 'describe this image', {
        attachmentRefs: ['@image:/tmp/cached.png']
      })
    ]

    const [out] = reconcileResumeMessages(next, previous)

    expect(out.attachmentRefs).toEqual(['@image:/tmp/authoritative.png'])
  })

  it('does not preserve attachment refs when the user text differs', () => {
    const next = [msg('stored-user', 'user', 'a different prompt')]

    const previous = [
      msg('live-user', 'user', 'describe this image', {
        attachmentRefs: ['@image:/tmp/photo.png']
      })
    ]

    const [out] = reconcileResumeMessages(next, previous)

    expect(out.attachmentRefs).toBeUndefined()
  })

  // #75825: switching sessions mid-stream can re-hydrate an empty inflight shell
  // at the same ordinal as the live stream row that still holds the full reply.
  it('prefers a richer local pending assistant over an empty projection shell', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'hello from stream', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('assistant-stream-sess', 'assistant', '', { pending: true })]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(reconciled[1]).toMatchObject({ id: 'assistant-stream-live', pending: true })
    expect(chatMessageText(reconciled[1])).toBe('hello from stream')
  })

  it('prefers a richer local pending assistant when the projection lags mid-stream', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'hello world', { pending: true })
    ]

    const next = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'hello', { pending: true })
    ]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(chatMessageText(reconciled[1])).toBe('hello world')
    expect(reconciled[1].id).toBe('assistant-stream-live')
  })

  it('does not override when the authoritative assistant has advanced further', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'hello', { pending: true })
    ]

    const next = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'hello world', { pending: true })
    ]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(chatMessageText(reconciled[1])).toBe('hello world')
    expect(reconciled[1].id).toBe('assistant-stream-sess')
  })

  // The reported "no inference traces or tool calls": mid tool-work, the local
  // row holds reasoning + tool calls and NO text yet, so both bodies are empty
  // text and a text-length comparison cannot tell them apart.
  it('prefers a traces-only local pending row over an empty shell', () => {
    const previous = [msg('1-user', 'user', 'run the tools'), streamingMsg('assistant-stream-live', '')]

    const next = [
      msg('1-user', 'user', 'run the tools'),
      msg('assistant-stream-sess', 'assistant', '', { pending: true })
    ]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(reconciled[1].id).toBe('assistant-stream-live')
    expect(reconciled[1].parts.map(part => part.type)).toEqual(['reasoning', 'tool-call'])
  })

  // A longer local body that is NOT an extension of the authoritative text is a
  // different turn at the same ordinal (compression rewrites history) and must
  // not hijack the slot.
  it('leaves a shorter non-prefix authoritative assistant intact', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'a long local reply about something else entirely', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('9-assistant', 'assistant', 'short authoritative answer')]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(reconciled[1].id).toBe('9-assistant')
    expect(chatMessageText(reconciled[1])).toBe('short authoritative answer')
  })

  // A retained failure snapshot (`inflight.error`) is projected with empty text.
  // Preferring the local partial over it would erase the error and repaint the
  // turn as healthy.
  it('does not treat an errored authoritative row as an empty shell', () => {
    const previous = [
      msg('1-user', 'user', 'do the thing'),
      msg('assistant-stream-live', 'assistant', 'partial answer before the failure', { pending: true })
    ]

    const next = [
      msg('1-user', 'user', 'do the thing'),
      msg('assistant-stream-sess', 'assistant', '', { error: 'model call failed: 500' })
    ]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(reconciled[1].error).toBe('model call failed: 500')
  })

  // Content comes from the renderer; liveness stays the backend's call. A
  // settled shell (queued turn behind a finished inflight one) must not leave
  // the preserved reply spinning forever.
  it('takes the local body but the authoritative settled state', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'streamed body', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('assistant-stream-sess', 'assistant', '', { pending: false })]

    const reconciled = reconcileResumeMessages(next, previous)

    expect(reconciled[1]).toMatchObject({ id: 'assistant-stream-live', pending: false })
    expect(chatMessageText(reconciled[1])).toBe('streamed body')
  })
})

describe('preserveLocalPendingTurnMessages', () => {
  it('keeps an optimistic user turn and pending assistant when the server projection is behind', () => {
    const next = [msg('1-user', 'user', 'first'), msg('2-assistant', 'assistant', 'first answer')]

    const previous = [
      ...next,
      msg('user-optimistic', 'user', 'new question'),
      msg('assistant-stream-1', 'assistant', 'partial answer', { pending: true })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user',
      '2-assistant',
      'user-optimistic',
      'assistant-stream-1'
    ])
  })

  it('drops the local copies once the same role ordinals are authoritative', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'new question'),
      msg('assistant-stream-1', 'assistant', 'partial answer', { pending: true })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', 'new question'),
      msg('4-assistant-stored', 'assistant', 'complete answer')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('drops stale optimistic history after compression and keeps only the live tail', () => {
    const compressedAuthority = [
      msg('stored-user', 'user', 'first turn that survived compression'),
      msg('stored-assistant', 'assistant', 'latest authoritative reply')
    ]

    const pollutedWarmCache = [
      msg('user-old-1', 'user', 'compressed-away prompt one'),
      msg('assistant-old-1', 'assistant', 'compressed-away reply one'),
      msg('user-old-2', 'user', 'compressed-away prompt two'),
      msg('assistant-old-2', 'assistant', 'compressed-away reply two'),
      msg('user-inflight', 'user', 'the one genuinely in-flight prompt')
    ]

    expect(preserveLocalPendingTurnMessages(compressedAuthority, pollutedWarmCache).map(message => message.id)).toEqual(
      ['stored-user', 'stored-assistant', 'user-inflight']
    )
  })

  it('drops the live tail once the latest authoritative user has persisted it after compression', () => {
    const compressedAuthority = [
      msg('stored-user', 'user', 'the one genuinely in-flight prompt'),
      msg('stored-assistant', 'assistant', 'its authoritative reply')
    ]

    const pollutedWarmCache = [
      msg('user-old-1', 'user', 'compressed-away prompt one'),
      msg('assistant-old-1', 'assistant', 'compressed-away reply one'),
      msg('user-inflight', 'user', 'the one genuinely in-flight prompt')
    ]

    expect(preserveLocalPendingTurnMessages(compressedAuthority, pollutedWarmCache)).toBe(compressedAuthority)
  })

  // A mid-turn redirect inserts its correction as a SECOND optimistic user row
  // for the same turn. Keeping only the newest dropped the prompt that started
  // it, so a resume repainted the thread with the user's message missing.
  it('keeps every optimistic user row in the live run after a mid-turn redirect', () => {
    const previous = [
      msg('user-1000', 'user', 'remove the session counts'),
      msg('user-2000', 'user', 'hurry up'),
      msg('assistant-stream-1', 'assistant', 'Moving.', { pending: true })
    ]

    expect(preserveLocalPendingTurnMessages([], previous).map(message => message.id)).toEqual([
      'user-1000',
      'user-2000',
      'assistant-stream-1'
    ])
  })

  it('still drops optimistic rows separated from the live run by an assistant reply', () => {
    const previous = [
      msg('user-stale', 'user', 'compressed-away prompt'),
      msg('assistant-stale', 'assistant', 'compressed-away reply'),
      msg('user-1000', 'user', 'the live prompt'),
      msg('user-2000', 'user', 'the correction'),
      msg('assistant-stream-1', 'assistant', 'Moving.', { pending: true })
    ]

    expect(preserveLocalPendingTurnMessages([], previous).map(message => message.id)).toEqual([
      'user-1000',
      'user-2000',
      'assistant-stream-1'
    ])
  })

  // #67603: the gateway persists model-switch / personality notices as role=user
  // ([System: …], tui_gateway/server.py). A single trailing marker is already
  // handled by the latestAuthoritativeUser guard above, but TWO switches around
  // one turn put a marker BEFORE the committed prompt (shifting its ordinal) and
  // another AFTER it (so the prompt is no longer the last user row, so the text
  // guard can't rescue it). Naive ordinal pairing then pairs the optimistic row
  // against a marker, treats it as uncommitted, and re-appends it — the
  // duplicated user bubble stacked at the bottom of the chat.
  it('does not duplicate the optimistic prompt when markers bracket it (two model switches)', () => {
    const marker = (name: string) => `[System: The active model for this chat has changed to ${name}.]`

    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'second question')
    ]

    const next = [
      msg('s1-user', 'user', 'first'),
      msg('s2-assistant', 'assistant', 'first answer'),
      msg('s3-marker', 'user', marker('k2')),
      msg('s4-user', 'user', 'second question'),
      msg('s5-assistant', 'assistant', 'second answer'),
      msg('s6-marker', 'user', marker('k3'))
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('still keeps a genuinely uncommitted optimistic turn when a marker is present', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'new question')
    ]

    // The marker is persisted but the new prompt has not committed yet — the
    // optimistic row must survive (marker exclusion must not over-correct).
    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-marker-stored', 'user', '[System: The active model for this chat has changed to k3.]')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-marker-stored',
      'user-optimistic'
    ])
  })

  // #70720: the gateway persists an attached image as a leading `@image:<path>`
  // directive line, while the local optimistic composer keeps it as separate
  // `attachmentRefs`. A naive text compare (chatMessageText a === b) therefore
  // always mismatched whenever an image was attached and re-appended the
  // optimistic row as a distinct, duplicate user bubble. Both sides must now
  // reduce to the same visible text via textWithoutReferenceLines.
  it('does not duplicate the optimistic image turn when the persisted turn carries @image refs', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'what is in this photo?', {
        attachmentRefs: ['@image:/tmp/cat.png']
      })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', '@image:/tmp/cat.png\nwhat is in this photo?')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('does not duplicate the optimistic file turn when the persisted turn carries @file refs', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'text', {
        attachmentRefs: ['@file:X']
      })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', '@file:X\n\ntext')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it.each(WIRE_REFERENCE_KINDS.filter(kind => kind !== 'file' && kind !== 'image'))(
    'does not duplicate the optimistic %s turn when the persisted turn carries its directive',
    kind => {
      const ref = `@${kind}:X`

      const previous = [
        msg('1-user', 'user', 'first'),
        msg('2-assistant', 'assistant', 'first answer'),
        msg('user-optimistic', 'user', 'text', {
          attachmentRefs: [ref]
        })
      ]

      const next = [
        msg('1-user-stored', 'user', 'first'),
        msg('2-assistant-stored', 'assistant', 'first answer'),
        msg('3-user-stored', 'user', `${ref}\n\ntext`)
      ]

      expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
    }
  )

  it('does not duplicate a directive-only file turn', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', '', {
        attachmentRefs: ['@file:X']
      })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', '@file:X')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('does not duplicate a turn with multiple CRLF directives and Unicode payloads', () => {
    const refs = ['@file:`資料/über notes.md`', '@url:`https://example.com/café?q=✓`']

    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'text', {
        attachmentRefs: refs
      })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', `${refs.join('\r\n')}\r\n\r\ntext`)
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('strips only complete reference lines from visible text', () => {
    expect(textWithoutReferenceLines('see @file:X here')).toBe('see @file:X here')
    expect(textWithoutReferenceLines('@file:X trailing prose')).toBe('@file:X trailing prose')
    expect(textWithoutReferenceLines('  @file:X')).toBe('@file:X')
  })

  it('still keeps a genuinely uncommitted optimistic image turn when the persisted text differs', () => {
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-optimistic', 'user', 'a different caption', {
        attachmentRefs: ['@image:/tmp/cat.png']
      })
    ]

    // Persisted turn has a different caption — the optimistic row is still
    // uncommitted and must survive (image-aware compare must not over-correct).
    const next = [
      msg('1-user-stored', 'user', 'first'),
      msg('2-assistant-stored', 'assistant', 'first answer'),
      msg('3-user-stored', 'user', '@image:/tmp/cat.png\nwhat is in this photo?')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-user-stored',
      'user-optimistic'
    ])
  })

  // #75825: an empty inflight projection shell at the same ordinal must not
  // discard the local pending assistant that still holds the streamed content.
  // Replace the shell (do not append) so the transcript shows one reply.
  it('replaces an empty inflight shell with a fuller local pending assistant', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'partial answer so far', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('assistant-stream-sess', 'assistant', '', { pending: true })]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.map(message => message.id)).toEqual(['1-user', 'assistant-stream-live'])
    expect(chatMessageText(preserved[1])).toBe('partial answer so far')
    expect(preserved[1].pending).toBe(true)
  })

  it('replaces a lagging same-id shell with the fuller local pending body', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'full streamed content', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('assistant-stream-sess', 'assistant', '', { pending: true })]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.map(message => message.id)).toEqual(['1-user', 'assistant-stream-sess'])
    expect(chatMessageText(preserved[1])).toBe('full streamed content')
  })

  it('still drops local pending when authoritative text is at least as complete', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'partial', { pending: true })
    ]

    const next = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'partial and more', { pending: true })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  // Mid tool-work both bodies are empty text, so only the parts distinguish the
  // live row from the shell — the reported "no inference traces or tool calls".
  it('replaces an empty shell with a traces-only local pending row', () => {
    const previous = [msg('1-user', 'user', 'run the tools'), streamingMsg('assistant-stream-live', '')]

    const next = [
      msg('1-user', 'user', 'run the tools'),
      msg('assistant-stream-sess', 'assistant', '', { pending: true })
    ]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved).toHaveLength(2)
    expect(preserved[1].parts.map(part => part.type)).toEqual(['reasoning', 'tool-call'])
  })

  // Length alone is not identity: a longer local row that does not extend the
  // authoritative text belongs to another turn and must not take its slot — by
  // ordinal or by reusing the stream id.
  it('leaves a shorter non-prefix authoritative assistant intact', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'a long local reply about something else entirely', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('9-assistant', 'assistant', 'short authoritative answer')]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.map(message => message.id)).toEqual(['1-user', '9-assistant'])
    expect(chatMessageText(preserved[1])).toBe('short authoritative answer')
  })

  it('leaves a shorter non-prefix authoritative assistant intact on the same stream id', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'a long local reply about something else entirely', { pending: true })
    ]

    const next = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'short authoritative answer')
    ]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(chatMessageText(preserved[1])).toBe('short authoritative answer')
  })

  it('does not erase a retained failure with the local partial', () => {
    const previous = [
      msg('1-user', 'user', 'do the thing'),
      msg('assistant-stream-live', 'assistant', 'partial answer before the failure', { pending: true })
    ]

    const next = [
      msg('1-user', 'user', 'do the thing'),
      msg('assistant-stream-sess', 'assistant', '', { error: 'model call failed: 500' })
    ]

    const assistant = preserveLocalPendingTurnMessages(next, previous).find(message => message.role === 'assistant')

    expect(assistant?.error).toBe('model call failed: 500')
    expect(assistant?.pending).not.toBe(true)
  })

  it('takes the local body but the authoritative settled state', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-live', 'assistant', 'streamed body', { pending: true })
    ]

    const next = [msg('1-user', 'user', 'question'), msg('assistant-stream-sess', 'assistant', '', { pending: false })]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved[1]).toMatchObject({ id: 'assistant-stream-live', pending: false })
    expect(chatMessageText(preserved[1])).toBe('streamed body')
  })

  // #70209: history committed the reply under its own id, so the settled local
  // stream row sits at a later ordinal, pairs with nothing, and gets appended —
  // the same answer twice.
  it('does not re-append a settled stream row the authoritative history already carries', () => {
    const next = [msg('1-user-stored', 'user', 'question'), msg('2-assistant-stored', 'assistant', 'answer')]
    const settledLocalStream = msg('assistant-stream-runtime-1', 'assistant', 'answer', { pending: false })

    expect(preserveLocalPendingTurnMessages(next, [...next, settledLocalStream])).toBe(next)
  })

  // The reply finished locally but the gateway had not committed it when the
  // session was reopened — the local row is the only copy and must survive.
  it('keeps a settled stream row the authoritative history has not committed', () => {
    const previous = [
      msg('1-user', 'user', 'question'),
      msg('assistant-stream-sess', 'assistant', 'the finished reply', { pending: false })
    ]

    const next = [msg('1-user', 'user', 'question')]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user',
      'assistant-stream-sess'
    ])
  })

  // The whole point of replacing rather than appending: one reply on screen,
  // and the committed history around the live turn untouched.
  it('does not duplicate or rewrite committed history around the live turn', () => {
    const history = [
      msg('1-user', 'user', 'first question'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('3-user', 'user', 'run the tools')
    ]

    const previous = [...history, streamingMsg('assistant-stream-live', 'here is the full reply')]
    const next = [...history, msg('assistant-stream-sess', 'assistant', '', { pending: true })]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved).toHaveLength(4)
    expect(chatMessageText(preserved[1])).toBe('first answer')
    expect(preserved.filter(message => message.role === 'assistant')).toHaveLength(2)
  })
})

describe('appendLiveSessionProjection', () => {
  // Corrections typed while a turn ran are their own user bubbles on the same
  // turn. Resume must rebuild the prompt AND every correction, in order.
  it('projects mid-turn redirect corrections after the prompt that started the turn', () => {
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
      'hurry up',
      'and the worktree ones',
      'Moving.'
    ])
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
})
