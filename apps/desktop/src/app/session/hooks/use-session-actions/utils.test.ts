import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, type ChatMessagePart, chatMessageText, textPart } from '@/lib/chat-messages'
import { $approvalModes, approvalModeForProfile } from '@/store/approval-mode'
import { $desktopOnboarding, consumePendingCredentialWarning } from '@/store/onboarding'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $currentBranch,
  $currentCwd,
  $currentModel,
  $currentProvider,
  $currentUsage,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentModel,
  setCurrentProvider,
  setCurrentUsage,
  setSelectedStoredSessionId,
  workspaceCwdBelongsToSelectedSession
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { appendLiveSessionProjection } from './live-session-projection'
import {
  applyRuntimeInfo,
  applyStoredSessionPreviewRuntimeInfo,
  chatMessageArraysEquivalent,
  chatMessagesEquivalent,
  chatPartsEquivalent,
  goneSessionVerdict,
  isSessionGoneError,
  overlayConcurrentMessageChanges,
  preserveEquivalentTranscript,
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages,
  resolveResumedBusy,
  selectBranchMessages,
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
    consumePendingCredentialWarning()
    $desktopOnboarding.set({ ...initialOnboardingState, reason: null, requested: false })
  })

  afterEach(() => {
    consumePendingCredentialWarning()
    $desktopOnboarding.set(initialOnboardingState)
  })

  it('defers the empty-key warning to submit time instead of popping onboarding on switch', () => {
    const warning = "No API key configured for provider 'openrouter'. First message will fail."

    applyRuntimeInfo({ credential_warning: warning })

    // Merely switching to (or activating a session on) the unconfigured
    // profile must NOT open the blocking overlay…
    expect($desktopOnboarding.get()).toMatchObject({ reason: null, requested: false })
    // …but the warning is staged for the submit path to consume.
    expect(consumePendingCredentialWarning()).toBe(warning)
    // Consuming clears it — the next submit doesn't double-fire.
    expect(consumePendingCredentialWarning()).toBeNull()
  })

  it('a warning-free session event clears the stash (profile healed or switched away)', () => {
    applyRuntimeInfo({
      credential_warning: "No API key configured for provider 'openrouter'. First message will fail."
    })
    applyRuntimeInfo({ model: 'gpt-5' })

    expect(consumePendingCredentialWarning()).toBeNull()
  })

  it('ignores an auxiliary-provider warning', () => {
    applyRuntimeInfo({ credential_warning: 'OPENROUTER_API_KEY not set' })

    expect($desktopOnboarding.get()).toMatchObject({ reason: null, requested: false })
    expect(consumePendingCredentialWarning()).toBeNull()
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

  it('returns authoritative usage for a background runtime snapshot', () => {
    const patch = applyRuntimeInfo(
      { usage: { calls: 3, compressions: 4, input: 100, output: 20, total: 120 } },
      { foreground: false }
    )

    expect(patch?.usage).toEqual({ calls: 3, compressions: 4, input: 100, output: 20, total: 120 })
  })

  it('clears a previous session compression count when the focused snapshot omits it', () => {
    setCurrentUsage({ calls: 2, compressions: 4, input: 10, output: 5, total: 15 })

    applyRuntimeInfo({ usage: { calls: 0, input: 0, output: 0, total: 0 } })

    expect($currentUsage.get().compressions).toBeUndefined()
  })

  it('does not let a background runtime clear the focused compression count', () => {
    setCurrentUsage({ calls: 2, compressions: 4, input: 10, output: 5, total: 15 })

    applyRuntimeInfo({ usage: { calls: 0, input: 0, output: 0, total: 0 } }, { foreground: false })

    expect($currentUsage.get().compressions).toBe(4)
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

  it('clears live-only compression usage as soon as a cold session switch starts', () => {
    setCurrentUsage({ calls: 2, compressions: 4, input: 10, output: 5, total: 15 })

    applyStoredSessionPreviewRuntimeInfo({ cwd: '/next-project', model: 'gpt' }, 'session-next')

    expect($currentUsage.get().compressions).toBeUndefined()
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

describe('goneSessionVerdict', () => {
  it('drafts only when the id is verifiably gone in calm conditions', () => {
    expect(goneSessionVerdict({ createdThisRun: false, stillListed: false, switchInFlight: false })).toBe('draft')
  })

  it('retries when a profile/connection switch is in flight (#88540 route revert)', () => {
    expect(goneSessionVerdict({ createdThisRun: false, stillListed: false, switchInFlight: true })).toBe('retry')
  })

  it('retries when the session is still listed on some profile', () => {
    expect(goneSessionVerdict({ createdThisRun: false, stillListed: true, switchInFlight: false })).toBe('retry')
  })

  it('never discards a session created by this window in this run', () => {
    expect(goneSessionVerdict({ createdThisRun: true, stillListed: false, switchInFlight: false })).toBe('retry')
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

describe('selectBranchMessages', () => {
  it('uses the complete authoritative transcript for a whole-chat branch', () => {
    const local = [msg('summary', 'assistant', 'compact summary'), msg('tail', 'assistant', 'latest answer')]

    const authoritative = [
      msg('old-user', 'user', 'first question', { rowId: 11 }),
      msg('old-assistant', 'assistant', 'first answer', { rowId: 12 }),
      msg('tail-user', 'user', 'latest question', { rowId: 13 }),
      msg('tail-assistant', 'assistant', 'latest answer', { rowId: 14 })
    ]

    expect(selectBranchMessages(local, authoritative).map(message => message.content)).toEqual([
      'first question',
      'first answer',
      'latest question',
      'latest answer'
    ])
  })

  it('maps a clicked local bubble to the authoritative row before slicing', () => {
    const local = [
      msg('tail-user', 'user', 'latest question', { rowId: 13 }),
      msg('tail-assistant', 'assistant', 'latest answer', { rowId: 14 })
    ]

    const authoritative = [
      msg('old-user', 'user', 'first question', { rowId: 11 }),
      msg('old-assistant', 'assistant', 'first answer', { rowId: 12 }),
      msg('tail-user', 'user', 'latest question', { rowId: 13 }),
      msg('tail-assistant', 'assistant', 'latest answer', { rowId: 14 })
    ]

    expect(selectBranchMessages(local, authoritative, 'tail-assistant').map(message => message.content)).toEqual([
      'first question',
      'first answer',
      'latest question',
      'latest answer'
    ])
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

  it('returns false when visible timeline boundaries change', () => {
    const started = { type: 'text' as const, text: 'Hello', timestamp: 10 }
    const completed = { ...started, completedAt: 11 }

    expect(chatPartsEquivalent(started, completed)).toBe(false)
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
})

describe('chatMessagesEquivalent', () => {
  it('returns true for structurally identical messages', () => {
    expect(chatMessagesEquivalent(msg('1', 'user', 'Hello'), msg('1', 'user', 'Hello'))).toBe(true)
  })

  it('returns false when a visible message timestamp changes', () => {
    const before = { ...msg('1', 'user', 'Hello'), timestamp: 10 }
    const after = { ...before, timestamp: 11 }

    expect(chatMessagesEquivalent(before, after)).toBe(false)
  })

  it('returns false when text part content differs', () => {
    expect(chatMessagesEquivalent(msg('1', 'user', 'Hello'), msg('1', 'user', 'World'))).toBe(false)
  })

  it('returns false when message IDs differ', () => {
    expect(chatMessagesEquivalent(msg('msg-1', 'user', 'Hello'), msg('msg-2', 'user', 'Hello'))).toBe(false)
  })
})

describe('chatMessageArraysEquivalent', () => {
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

  it('matches a windowed repeated user prompt by row id then client id before role ordinal', () => {
    const previous = [
      msg('local-old', 'user', 'same prompt', {
        attachmentRefs: ['@file:old.txt'],
        clientMessageId: 'client-old'
      }),
      msg('local-new', 'user', 'same prompt', {
        attachmentRefs: ['@file:new.txt'],
        clientMessageId: 'client-new'
      })
    ]

    const next = [
      msg('stored-new', 'user', 'same prompt', {
        clientMessageId: 'client-new',
        rowId: 42
      })
    ]

    const [out] = reconcileResumeMessages(next, previous)

    expect(out.attachmentRefs).toEqual(['@file:new.txt'])
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
  it('does not re-append a durably completed reply that compaction re-inserted under a new row id', () => {
    const previous = [
      msg('u1', 'user', 'q1', { rowId: 11340 }),
      msg('a1', 'assistant', 'r1', { rowId: 11345, durableComplete: true }),
      msg('user-9-x', 'user', 'q2', { rowId: 11350 }),
      msg('assistant-stream-9-0', 'assistant', 'r2', { pending: false, rowId: 11359, durableComplete: true })
    ]

    const next = [
      msg('s-u1', 'user', 'q1', { rowId: 11440 }),
      msg('s-a1', 'assistant', 'r1', { rowId: 11444 }),
      msg('s-u2', 'user', 'q2', { rowId: 11450 }),
      msg('s-a2', 'assistant', 'r2', { rowId: 11459 })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous)).toEqual(next)
  })

  // A turn that compressed mid-flight only earns a partial receipt
  // (`complete: false`), but its rows are still proven committed. When a later
  // compaction rewrites every one of them, the local copy is stale history.
  const partialReceipt = { row_ids: [11350, 11355, 11359], complete: false, final_assistant_row_id: 11359 }

  it('does not re-append a partially receipted reply once compaction rewrote all of its rows', () => {
    const previous = [
      msg('u1', 'user', 'q1', { rowId: 11340 }),
      msg('user-9-x', 'user', 'q2', { rowId: 11350 }),
      msg('assistant-stream-9-0', 'assistant', 'r2', {
        pending: false,
        rowId: 11359,
        durableComplete: false,
        persistedTurn: partialReceipt
      })
    ]

    const reinserted = [
      msg('s-summary', 'assistant', '[summary]', { rowId: 11440 }),
      msg('s-u2', 'user', 'q2', { rowId: 11450 }),
      msg('s-a2', 'assistant', 'r2', { rowId: 11459 })
    ]

    const summarizedAway = [
      msg('s-summary', 'assistant', '[summary]', { rowId: 11440 }),
      msg('s-u3', 'user', 'q3', { rowId: 11450 }),
      msg('s-a3', 'assistant', 'r3', { rowId: 11459 })
    ]

    expect(preserveLocalPendingTurnMessages(reinserted, previous)).toEqual(reinserted)
    expect(preserveLocalPendingTurnMessages(summarizedAway, previous)).toEqual(summarizedAway)
  })

  it('keeps a partially receipted reply the store has not reached or still partly holds', () => {
    const reply = msg('assistant-stream-9-0', 'assistant', 'r2 with unpersisted tail', {
      pending: false,
      rowId: 11359,
      durableComplete: false,
      persistedTurn: partialReceipt
    })

    const previous = [msg('u1', 'user', 'q1', { rowId: 11340 }), msg('user-9-x', 'user', 'q2', { rowId: 11350 }), reply]
    const behind = [msg('s-u1', 'user', 'q1', { rowId: 11340 })]

    const partlyHeld = [
      msg('s-u1', 'user', 'q1', { rowId: 11340 }),
      msg('s-u2', 'user', 'q2', { rowId: 11350 }),
      msg('s-a2', 'assistant', 'r2', { rowId: 11460 })
    ]

    expect(preserveLocalPendingTurnMessages(behind, previous).map(message => message.id)).toContain(reply.id)
    expect(preserveLocalPendingTurnMessages(partlyHeld, previous).map(message => message.id)).toContain(reply.id)
  })

  it('does not append acknowledged local history after a shifted newest page', () => {
    const previous = [
      msg('user-first', 'user', 'Original request', { timestamp: 1 }),
      msg('assistant-stream-first', 'assistant', 'Working.', { pending: false, timestamp: 2 }),
      msg('user-followup', 'user', 'Follow-up request', { timestamp: 3 }),
      msg('assistant-stream-final', 'assistant', 'Completed.', { pending: false, rowId: 30, durableComplete: true })
    ]

    const answer = msg('stored-answer', 'assistant', 'Completed.', { rowId: 30, timestamp: 5 })

    const folded = { ...answer, rowId: 20, parts: [{ ...textPart('Completed.'), sourceRowId: 30 }] }

    for (const next of [[answer], [msg('stored-followup', 'user', 'Follow-up request'), answer], [folded]]) {
      expect(preserveLocalPendingTurnMessages(next, previous)).toEqual(next)
    }

    const unacknowledged = msg('user-new', 'user', 'A new request', { timestamp: 6 })
    expect(preserveLocalPendingTurnMessages([answer], [...previous, unacknowledged])).toEqual([answer, unacknowledged])
  })

  it('drops the acknowledged prompt when a compaction handoff precedes its committed copy', () => {
    // #121088: an in-place compaction handoff (or preserved-task notice) is a
    // synthetic USER-role row, so the committed copy of the prompt is no
    // longer the newest user row — and one lands after the reply too. A
    // newest-only compare misses the committed copy and re-appends the
    // optimistic row below the whole refreshed turn.
    const previous = [
      msg('1-user', 'user', 'first', { rowId: 100 }),
      msg('2-assistant', 'assistant', 'first answer', { rowId: 101 }),
      msg('user-optimistic', 'user', 'unable to publish')
    ]

    const next = [
      msg('1-user-stored', 'user', 'first', { rowId: 100 }),
      msg('2-assistant-stored', 'assistant', 'first answer', { rowId: 101 }),
      msg('3-handoff', 'user', 'Context was compacted; continuing.'),
      msg('4-user-stored', 'user', 'unable to publish', { rowId: 200 }),
      msg('5-assistant-stored', 'assistant', 'stored answer', { rowId: 201 }),
      msg('6-notice', 'user', 'Preserved task notice')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-handoff',
      '4-user-stored',
      '5-assistant-stored',
      '6-notice'
    ])
  })

  it('still keeps a genuinely unacknowledged repetition of an older question', () => {
    // The committed twin of a genuine repeat predates the acknowledged
    // boundary and never enters the newly committed window, so widening
    // the acknowledged-prompt compare must not swallow it.
    const previous = [
      msg('1-user', 'user', 'what time is it?', { rowId: 100 }),
      msg('2-assistant', 'assistant', 'noon', { rowId: 101 }),
      msg('user-optimistic', 'user', 'what time is it?')
    ]

    const next = [
      msg('1-user-stored', 'user', 'what time is it?', { rowId: 100 }),
      msg('2-assistant-stored', 'assistant', 'noon', { rowId: 101 }),
      msg('3-system-user', 'user', 'Preserved task notice')
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-system-user',
      'user-optimistic'
    ])
  })

  it('keeps a newer equal reply and its prompt until that occurrence is persisted', () => {
    const previousAnswer = msg('stored-answer', 'assistant', 'Completed.', { rowId: 10 })
    const prompt = msg('user-new', 'user', 'Repeat the check', { rowId: 11 })
    const reply = msg('assistant-stream-new', 'assistant', 'Completed.', { pending: false, rowId: 12 })
    reply.parts.push({ type: 'reasoning', text: 'Reasoning only from the new occurrence.' })
    expect(reconcileResumeMessages([previousAnswer], [prompt, reply])).toEqual([previousAnswer])

    // Neither equal prose nor missing clocks can make a different persisted
    // occurrence acknowledge this one, even when the older row left the cache.
    for (const previous of [
      [previousAnswer, prompt, reply],
      [prompt, reply]
    ]) {
      expect(preserveLocalPendingTurnMessages([previousAnswer], previous)).toEqual([previousAnswer, prompt, reply])
    }
  })

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

  // Arrival-ordered mid-turn corrections (#73793) seal the live output BETWEEN
  // the prompt and the correction. The sealed live-tail row must not end the
  // optimistic run, or a refresh drops the prompt that started the turn.
  it('keeps the whole live run when sealed live output sits between prompt and correction', () => {
    const previous = [
      msg('user-1000', 'user', 'remove the session counts'),
      msg('assistant-stream-1', 'assistant', 'two screens of output', { interim: true }),
      msg('user-2000', 'user', 'hurry up'),
      msg('assistant-stream-2', 'assistant', 'post-redirect output', { pending: true })
    ]

    expect(preserveLocalPendingTurnMessages([], previous).map(message => message.id)).toEqual([
      'user-1000',
      'assistant-stream-1',
      'user-2000',
      'assistant-stream-2'
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

  it('does not keep a settled final-answer bubble already folded into the tool-round message', () => {
    const folded = {
      id: '1790016993.1043298-1-assistant',
      role: 'assistant' as const,
      parts: [
        { type: 'text' as const, text: 'I will inspect the fixture, then give the final result.' },
        { type: 'tool-call' as const, toolCallId: 'call-1', toolName: 'terminal', result: '71' },
        { type: 'text' as const, text: 'The result is 71.' }
      ]
    }

    const next = [msg('1-user', 'user', 'inspect the fixture'), folded]

    const previous = [
      msg('1-user', 'user', 'inspect the fixture'),
      { ...folded, parts: folded.parts.slice(0, 2) },
      msg('assistant-stream-placeholder', 'assistant', '', { pending: false }),
      msg('assistant-stream-final', 'assistant', 'The result is 71.', { pending: false })
    ]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    const finals = preserved.flatMap(message =>
      message.parts.filter(part => part.type === 'text' && part.text === 'The result is 71.')
    )

    expect(finals).toHaveLength(1)
    expect(preserved.map(message => message.id)).not.toContain('assistant-stream-final')
  })

  // #121613: a completed reply that settled onto a non-stream id (an interim
  // id the completion settled onto, or an appended `assistant-<ts>` bubble)
  // is invisible to the stream-id rule, but when the refreshed page has not
  // committed it the local row is the only copy and must survive.
  it('keeps a settled non-stream reply the authoritative history has not committed', () => {
    const reply = msg('assistant-99', 'assistant', 'the completed reply', { pending: false, interim: false })
    const previous = [msg('1-user', 'user', 'question'), reply]
    const next = [msg('1-user', 'user', 'question')]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user',
      'assistant-99'
    ])
  })

  it('does not re-append a settled non-stream reply the authoritative history already carries', () => {
    const next = [msg('1-user-stored', 'user', 'question'), msg('2-assistant-stored', 'assistant', 'answer')]
    const settledLocal = msg('assistant-99', 'assistant', 'answer', { pending: false, interim: false })

    expect(preserveLocalPendingTurnMessages(next, [...next, settledLocal])).toBe(next)
  })

  it('does not resurrect a superseded interim bubble the refresh rewrote', () => {
    const next = [msg('1-user', 'user', 'question'), msg('2-assistant', 'assistant', 'rewritten final')]
    const interim = msg('assistant-interim-1', 'assistant', 'old interim', { pending: false, interim: true })

    expect(preserveLocalPendingTurnMessages(next, [msg('1-user', 'user', 'question'), interim])).toBe(next)
  })

  it('keeps a settled final-answer bubble the folded tool round has not absorbed', () => {
    const toolRound = {
      id: 'row-1-assistant',
      role: 'assistant' as const,
      parts: [
        { type: 'text' as const, text: 'I will inspect the fixture, then give the final result.' },
        { type: 'tool-call' as const, toolCallId: 'call-1', toolName: 'terminal', result: '71' }
      ]
    }

    const next = [msg('1-user', 'user', 'inspect the fixture'), toolRound]
    const previous = [...next, msg('assistant-stream-final', 'assistant', 'The result is 71.', { pending: false })]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toContain(
      'assistant-stream-final'
    )
  })

  it('keeps an equal final answer that belongs to a later turn history has not stored', () => {
    const folded = {
      id: 'row-1-assistant',
      role: 'assistant' as const,
      parts: [
        { type: 'text' as const, text: 'I will inspect the fixture, then give the final result.' },
        { type: 'tool-call' as const, toolCallId: 'call-1', toolName: 'terminal', result: '71' },
        { type: 'text' as const, text: 'The result is 71.' }
      ]
    }

    const next = [msg('1-user', 'user', 'first'), folded, msg('2-user', 'user', 'again')]
    const previous = [...next, msg('assistant-stream-later', 'assistant', 'The result is 71.', { pending: false })]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toContain(
      'assistant-stream-later'
    )
  })

  // A Codex Responses turn: an acknowledgement, two progress updates between
  // tool rounds, then the answer. Live, each seals as its own bubble; history
  // folds them into one row and may keep the public commentary only in
  // `reasoning` (#119716). Tool call ids are the durable identity either way.
  const tool = (toolCallId: string) =>
    ({ type: 'tool-call', toolCallId, toolName: 'terminal', result: 'ok' }) as ChatMessagePart

  const sealed = (id: string, parts: ChatMessagePart[], extra: Partial<ChatMessage> = {}) =>
    ({ id, role: 'assistant', parts, pending: false, interim: true, ...extra }) as ChatMessage

  const lunaTurn = (prefix: string, callPrefix: string) => [
    sealed(`assistant-stream-${prefix}-ack`, [{ type: 'text', text: `${prefix}: on it, reading the logs.` }]),
    sealed(`assistant-stream-${prefix}-progress-1`, [
      tool(`${callPrefix}-1`),
      { type: 'text', text: `${prefix}: logs clean.` }
    ]),
    sealed(`assistant-stream-${prefix}-progress-2`, [
      tool(`${callPrefix}-2`),
      { type: 'text', text: `${prefix}: config fixed.` }
    ]),
    sealed(
      `assistant-stream-${prefix}-final`,
      [tool(`${callPrefix}-3`), { type: 'text', text: `${prefix}: all done.` }],
      {
        interim: false
      }
    )
  ]

  const lunaFold = (id: string, prefix: string, callPrefix: string, commentary: 'reasoning' | 'text') =>
    ({
      id,
      role: 'assistant',
      parts: [
        ...[`${prefix}: on it, reading the logs.`, `${prefix}: logs clean.`, `${prefix}: config fixed.`].flatMap(
          (text, at) => [
            commentary === 'text'
              ? ({ type: 'text', text } as ChatMessagePart)
              : ({ type: 'reasoning', text: `**Plan**\n\n${text}` } as ChatMessagePart),
            tool(`${callPrefix}-${at + 1}`)
          ]
        ),
        { type: 'text', text: `${prefix}: all done.` }
      ]
    }) as ChatMessage

  it.each(['reasoning', 'text'] as const)(
    'retires every sealed bubble of a folded turn whose commentary hydrated as %s',
    commentary => {
      const user = msg('1-user', 'user', 'fix it', { rowId: 1 })
      const next = [user, lunaFold('2-assistant', 'a', 'call-a', commentary)]

      expect(preserveLocalPendingTurnMessages(next, [user, ...lunaTurn('a', 'call-a')])).toBe(next)
    }
  )

  // The previous turn's bubbles are not owned by the newest prompt, so they
  // must not resurface under it (#119511, and the self-sustaining tail of
  // stale commentary in #119362) — nor may they swallow the live reply.
  it.each(['reasoning', 'text'] as const)(
    'does not re-append an earlier turn under a newer prompt when its commentary hydrated as %s',
    commentary => {
      const next = [
        msg('1-user', 'user', 'fix it', { rowId: 1 }),
        lunaFold('2-assistant', 'a', 'call-a', commentary),
        msg('3-user', 'user', 'and the other one', { rowId: 9 })
      ]

      const previous = [
        msg('user-1-a', 'user', 'fix it', { rowId: 1 }),
        ...lunaTurn('a', 'call-a'),
        // No submit receipt yet, so no acknowledged boundary past turn a.
        msg('user-2-b', 'user', 'and the other one'),
        sealed('assistant-stream-live', [tool('call-b-1'), { type: 'text', text: 'b: still going.' }])
      ]

      expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
        '1-user',
        '2-assistant',
        '3-user',
        'assistant-stream-live'
      ])
    }
  )

  // #118228: narration bubbles still marked pending when the rehydrate lands.
  // A sealed interim's text is final, so the fold carrying it retires it.
  it('retires pending interim narration the merged fold already carries, even with later turns stored', () => {
    const user = msg('1-user', 'user', 'run the build', { rowId: 1 })

    // More live bubbles than stored assistant rows: ordinal pairing runs out.
    const turn = [
      ...lunaTurn('a', 'call-a')
        .slice(0, 3)
        .map(row => ({ ...row, pending: true })),
      msg('assistant-stream-a-tail', 'assistant', 'a: all done.', { pending: true })
    ]

    const next = [
      user,
      lunaFold('2-assistant', 'a', 'call-a', 'text'),
      msg('3-system', 'system', 'Background Process Finished: bash build.sh'),
      msg('4-assistant', 'assistant', 'build verified'),
      msg('5-user', 'user', 'installed it, same problem', { rowId: 20 }),
      msg('6-assistant', 'assistant', 'then it is not the line count')
    ]

    expect(preserveLocalPendingTurnMessages(next, [user, ...turn])).toBe(next)
  })

  // The fold committed the tool rounds but not the answer yet: that bubble is
  // the only copy and must survive, while the carried commentary retires.
  it('keeps the final answer a fold has not committed yet', () => {
    const user = msg('1-user', 'user', 'fix it', { rowId: 1 })
    const fold = lunaFold('2-assistant', 'a', 'call-a', 'reasoning')
    const next = [user, { ...fold, parts: fold.parts.filter(part => part.type !== 'text') }]

    expect(
      preserveLocalPendingTurnMessages(next, [user, ...lunaTurn('a', 'call-a')]).map(message => message.id)
    ).toEqual(['1-user', '2-assistant', 'assistant-stream-a-final'])
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

  // A still-PENDING stream row whose committed twin the authoritative history
  // already carries (ordinal shifted under compaction) used to fall through to
  // `preserved.push` and render the same answer twice — the reported tail
  // duplication (A B C D E C D). The #70209 guard only covers settled local
  // rows (`pending !== true`); these cover the pending ones.
  it('does not re-append a pending stream row the authoritative history already carries', () => {
    const previous = [
      msg('1-user', 'user', '查金价'),
      msg('2-a', 'assistant', 'X'),
      streamingMsg('assistant-stream-live', '面板内容')
    ]

    const next = [msg('1-user', 'user', '查金价'), msg('9-assistant', 'assistant', '面板内容')]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('drops a pending stream row whose text the committed authoritative reply extends', () => {
    const previous = [
      msg('1-user', 'user', '查金价'),
      msg('2-a', 'assistant', 'X'),
      streamingMsg('assistant-stream-live', '面板')
    ]

    const next = [msg('1-user', 'user', '查金价'), msg('9-assistant', 'assistant', '面板内容完整版')]

    expect(preserveLocalPendingTurnMessages(next, previous)).toBe(next)
  })

  it('replaces the committed row with a further-along pending copy instead of appending', () => {
    const previous = [
      msg('1-user', 'user', '查金价'),
      msg('2-a', 'assistant', 'X'),
      streamingMsg('assistant-stream-live', '面板内容完整版')
    ]

    const next = [msg('1-user', 'user', '查金价'), msg('9-assistant', 'assistant', '面板')]

    const preserved = preserveLocalPendingTurnMessages(next, previous)

    expect(preserved.map(message => message.id)).toEqual(['1-user', '9-assistant'])
    expect(chatMessageText(preserved[1])).toBe('面板内容完整版')
  })

  // The authoritative history genuinely does not have this reply yet — the
  // pending row is the only copy and must survive (same contract as the
  // settled-row variant above).
  it('still keeps a pending stream row when the authoritative history has no reply', () => {
    const previous = [msg('1-user', 'user', '查金价'), streamingMsg('assistant-stream-live', '面板内容')]

    const next = [msg('1-user', 'user', '查金价')]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user',
      'assistant-stream-live'
    ])
  })
})

describe('resolveResumedBusy', () => {
  it('keeps a live busy turn when the resume snapshot stalely reports idle (#70449)', () => {
    expect(resolveResumedBusy(false, true)).toBe(true)
    expect(resolveResumedBusy(undefined, true)).toBe(true)
    expect(resolveResumedBusy(null, true)).toBe(true)
  })

  it('clears busy when both the snapshot and the live cache agree the turn ended', () => {
    expect(resolveResumedBusy(false, false)).toBe(false)
    expect(resolveResumedBusy(undefined, false)).toBe(false)
  })

  it('adopts a running turn reported by the snapshot even without live state', () => {
    expect(resolveResumedBusy(true, false)).toBe(true)
    expect(resolveResumedBusy(true, true)).toBe(true)
  })
})

describe('overlayConcurrentMessageChanges', () => {
  it('does not replace an authoritative row with an unchanged baseline cache row', () => {
    const baseline = [msg('shared-assistant', 'assistant', 'stale cached answer')]
    const authoritative = [msg('shared-assistant', 'assistant', 'completed persisted answer')]

    const overlaid = overlayConcurrentMessageChanges(authoritative, baseline, baseline)

    expect(overlaid).toBe(authoritative)
    expect(overlaid[0].parts).toEqual([{ type: 'text', text: 'completed persisted answer' }])
  })

  it('replaces an activation stream placeholder and appends rows created after the baseline', () => {
    const baseline = [msg('assistant-stream-runtime', 'assistant', 'partial A', { pending: true })]
    const authoritative = [msg('assistant-stream-activation', 'assistant', 'partial A', { pending: true })]

    const current = [
      msg('assistant-stream-runtime', 'assistant', 'partial A + delta B', { pending: true }),
      msg('user-racing', 'user', 'racing prompt')
    ]

    const overlaid = overlayConcurrentMessageChanges(authoritative, baseline, current)

    expect(overlaid.map(message => message.id)).toEqual(['assistant-stream-runtime', 'user-racing'])
    expect(overlaid[0].parts).toEqual([{ type: 'text', text: 'partial A + delta B' }])
  })

  it('merges an activation prefix with a baseline-new runtime delta chunk', () => {
    const authoritative = [msg('assistant-stream-activation', 'assistant', 'partial A', { pending: true })]
    const current = [msg('assistant-stream-runtime', 'assistant', ' + delta B', { pending: true })]

    const overlaid = overlayConcurrentMessageChanges(authoritative, [], current)

    expect(overlaid.map(message => message.id)).toEqual(['assistant-stream-runtime'])
    expect(overlaid[0].parts).toEqual([
      { type: 'text', text: 'partial A' },
      { type: 'text', text: ' + delta B' }
    ])
  })

  // Switch back to a chat mid-reply: session.activate snapshots the first
  // chunk, message.complete settles the live row, THEN the gated REST page
  // resolves with the committed reply. Warm-activation composition order.
  it('keeps one reply when message.complete lands while the switch-back hydrate is in flight', () => {
    const snapshot = {
      session_id: 'runtime-b',
      turn_started_at: 100,
      inflight: { user: 'prompt b', assistant: 'A2 ', streaming: true }
    }

    const baseline = [
      msg('user-optimistic', 'user', 'prompt b'),
      msg('assistant-stream-1-2', 'assistant', 'A2 ', { pending: true })
    ]

    const current = [
      baseline[0],
      msg('assistant-stream-1-2', 'assistant', 'A2 finished while away', { pending: false })
    ]

    const persisted = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('4-assistant', 'assistant', 'A2 finished while away', { rowId: 4, timestamp: 105 })
    ]

    const hydrated = appendLiveSessionProjection(persisted, snapshot)
    const overlaid = overlayConcurrentMessageChanges(hydrated, baseline, current)

    expect(overlaid.map(message => [message.id, chatMessageText(message)])).toEqual([
      ['3-user', 'prompt b'],
      ['4-assistant', 'A2 finished while away']
    ])

    // Before the commit the same snapshot still projects the running reply.
    const running = overlayConcurrentMessageChanges(
      appendLiveSessionProjection(persisted.slice(0, 1), snapshot),
      baseline,
      [baseline[0], msg('assistant-stream-1-2', 'assistant', 'A2 finished', { pending: true })]
    )

    expect(running.map(message => [message.id, chatMessageText(message)])).toEqual([
      ['3-user', 'prompt b'],
      ['assistant-stream-1-2', 'A2 finished']
    ])
  })

  // The same prompt sent again (from another client, so the cache lacks its
  // row) streams the same opening as the previous answer. The cached-transcript
  // path must keep that NEXT turn's stream, and an errored settled row keeps
  // its failure instead of folding into identical committed text.
  it('keeps the next turn of a resent prompt and an errored settled row', () => {
    const cached = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('4-assistant', 'assistant', 'Same answer', { rowId: 4, timestamp: 105 })
    ]

    const snapshot = {
      session_id: 'runtime-b',
      turn_started_at: 200,
      inflight: { user: 'prompt b', assistant: 'Same ', streaming: true }
    }

    const hydrated = appendLiveSessionProjection(cached, snapshot)

    expect(hydrated.map(message => [message.id, chatMessageText(message)])).toEqual([
      ['3-user', 'prompt b'],
      ['4-assistant', 'Same answer'],
      ['assistant-stream-runtime-b', 'Same ']
    ])

    const settledNextTurn = msg('assistant-stream-1-3', 'assistant', 'Same answer', { pending: false })

    expect(overlayConcurrentMessageChanges(cached, cached, [...cached, settledNextTurn]).at(-1)).toBe(settledNextTurn)

    const errored = msg('assistant-stream-1-2', 'assistant', 'A2 done', { pending: false, error: 'stream lost' })
    const page = [msg('3-user', 'user', 'prompt b'), msg('4-assistant', 'assistant', 'A2 done')]

    expect(overlayConcurrentMessageChanges(page, [page[0]], [page[0], errored]).at(-1)).toBe(errored)
  })

  // The committed row and the settled live row capture the same reply at two
  // moments while it kept streaming, so one is routinely a prefix of the
  // other (#123993): accept either as a forward extension, as the sibling
  // removeRepresentedLocalLiveProjection already does (2494b95929).
  it('folds a settled live row that lags behind the committed row into one reply', () => {
    const page = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('4-assistant', 'assistant', 'A2 finished while away', { rowId: 4 })
    ]

    const current = [page[0], msg('assistant-stream-1-2', 'assistant', 'A2 finished', { pending: false })]

    const overlaid = overlayConcurrentMessageChanges(page, [], current)

    expect(overlaid.map(message => [message.id, chatMessageText(message)])).toEqual([
      ['3-user', 'prompt b'],
      ['4-assistant', 'A2 finished while away']
    ])
  })

  it('folds a settled live row that ran past the committed row into one reply', () => {
    const page = [
      msg('3-user', 'user', 'prompt b', { rowId: 3 }),
      msg('4-assistant', 'assistant', 'A2 finished', { rowId: 4 })
    ]

    const current = [page[0], msg('assistant-stream-1-2', 'assistant', 'A2 finished while away', { pending: false })]

    const overlaid = overlayConcurrentMessageChanges(page, [], current)

    expect(overlaid.map(message => [message.id, chatMessageText(message)])).toEqual([
      ['3-user', 'prompt b'],
      ['4-assistant', 'A2 finished']
    ])
  })
})

describe('preserveEquivalentTranscript', () => {
  it('keeps the current array BY REFERENCE when the replacement is content-equivalent', () => {
    // The exact warm-resume shape of #95595: fresh objects, identical content.
    const current = [msg('u-1', 'user', 'hello'), msg('a-1', 'assistant', 'const x = 1')]
    const freshObjects = current.map(message => ({ ...message, parts: [...message.parts] }))

    const preserved = preserveEquivalentTranscript(current, freshObjects)

    expect(preserved).toBe(current)
    expect(preserved[0]).toBe(current[0])
  })

  it('accepts the replacement when anything changed', () => {
    const current = [msg('u-1', 'user', 'hello')]
    const next = [msg('u-1', 'user', 'hello'), msg('a-1', 'assistant', 'new turn')]

    expect(preserveEquivalentTranscript(current, next)).toBe(next)
  })

  it('rejects the replacement when a message diverges in content', () => {
    const current = [msg('u-1', 'user', 'hello')]
    const next = [msg('u-1', 'user', 'hello world')]

    expect(preserveEquivalentTranscript(current, next)).toBe(next)
  })

  it('rejects the replacement when metadata a row renders diverges', () => {
    const current = [msg('u-1', 'user', 'hello')]
    const next = [msg('u-1', 'user', 'hello', { pending: true })]

    expect(preserveEquivalentTranscript(current, next)).toBe(next)
  })
})

describe('preserveLocalPendingTurnMessages attachment rewrites (#120978)', () => {
  it('drops the rowId-less pasted-attachment prompt once the rewritten copy commits', () => {
    // A pasted clipboard image is rewritten on the durable side (marker lines,
    // no data: ref) while the optimistic local row keeps the bare caption and
    // the data: ref — exact text/refs equality can never match them and the
    // optimistic row was re-appended below the newest turn.
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-1790168309-ab12cd', 'user', 'unable to publish', {
        attachmentRefs: ['data:image/png;base64,AAAA']
      })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first', { rowId: 1 }),
      msg('2-assistant-stored', 'assistant', 'first answer', { rowId: 2 }),
      msg('3-user-stored', 'user', 'unable to publish\n\n[Image attached at: C:\\img\\shot.png]\n[screenshot]', {
        rowId: 3
      })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-user-stored'
    ])
  })

  it('never tolerance-matches a plain repeat prompt without attachment evidence', () => {
    // The gating invariant: rewrite markers on the stored side AND attachment
    // evidence on the local side. A bare repeated caption is a genuine new
    // question and must survive.
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-plain-repeat', 'user', 'unable to publish')
    ]

    const next = [
      msg('1-user-stored', 'user', 'first', { rowId: 1 }),
      msg('2-assistant-stored', 'assistant', 'first answer', { rowId: 2 }),
      msg('3-user-stored', 'user', 'unable to publish\n\n[Image attached at: C:\\img\\shot.png]', { rowId: 3 })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-user-stored',
      'user-plain-repeat'
    ])
  })

  it('never tolerance-matches a rowId-bearing optimistic row it provably is not (#122079)', () => {
    // The submit receipt binds user_row_id onto the optimistic row while the
    // stored page still ends at the earlier paste, so the row reaches the
    // dedupe compare carrying a rowId none of the committed candidates hold.
    // The tolerant arm must stay inside the identity gate: pasting the same
    // captioned screenshot twice is a genuine new turn, not a duplicate.
    const previous = [
      msg('1-user', 'user', 'first'),
      msg('2-assistant', 'assistant', 'first answer'),
      msg('user-1790168309-ab12cd', 'user', 'unable to publish', {
        rowId: 901,
        attachmentRefs: ['data:image/png;base64,AAAA']
      })
    ]

    const next = [
      msg('1-user-stored', 'user', 'first', { rowId: 1 }),
      msg('2-assistant-stored', 'assistant', 'first answer', { rowId: 2 }),
      msg('3-user-stored', 'user', 'unable to publish\n\n[Image attached at: C:\\img\\shot.png]\n[screenshot]', {
        rowId: 3
      })
    ]

    expect(preserveLocalPendingTurnMessages(next, previous).map(message => message.id)).toEqual([
      '1-user-stored',
      '2-assistant-stored',
      '3-user-stored',
      'user-1790168309-ab12cd'
    ])
  })
})

describe('applyStoredSessionPreviewRuntimeInfo does not persist the preview', () => {
  beforeEach(() => {
    localStorage.clear()
    setCurrentModel('user-pick')
    setCurrentProvider('anthropic')
  })

  afterEach(() => {
    localStorage.clear()
  })

  // The preview is provisional: it paints while session.resume is still in
  // flight. An abandoned resume never repairs the selection afterwards, so a
  // persisting paint strands a manual model with an EMPTY provider in
  // localStorage — every later session.create pairs that model with the
  // profile provider and fails the coherence gate.
  it('moves the visible model/provider without persisting them', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '', model: 'claude-opus-5-5' }, 'session-next')

    // Visible paint happened…
    expect($currentModel.get()).toBe('claude-opus-5-5')
    expect($currentProvider.get()).toBe('')

    // …but nothing was persisted: the composer's sticky selection survives.
    expect(localStorage.getItem('hermes.desktop.composer.model')).toBe('user-pick')
    expect(localStorage.getItem('hermes.desktop.composer.provider')).toBe('anthropic')
  })
})
