import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerSubmit } from './use-composer-submit'

vi.mock('@/lib/chat-runtime', () => ({ SLASH_COMMAND_RE: /^\/\S/ }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))
vi.mock('@/store/composer', () => ({ clearSessionDraft: vi.fn() }))
vi.mock('@/store/composer-input-history', () => ({ resetBrowseState: () => {} }))
vi.mock('@/store/composer-queue', () => ({ enqueueQueuedPrompt: vi.fn() }))
vi.mock('../composer-utils', () => ({ cloneAttachments: (a: unknown[]) => a }))
vi.mock('../focus', () => ({ onComposerSubmitRequest: () => () => {} }))
vi.mock('../rich-editor', () => ({ composerPlainText: () => '' }))
vi.mock('../scope', () => ({ useComposerScope: () => ({ attachments: { clear: () => {} } }) }))

import { clearSessionDraft } from '@/store/composer'

const flushMicrotasks = () => new Promise<void>(resolve => setTimeout(resolve, 0))

function buildArgs(overrides: Partial<Parameters<typeof useComposerSubmit>[0]> = {}) {
  return {
    activeQueueSessionKey: 'session-a',
    activeQueueSessionKeyRef: { current: 'session-a' as string | null },
    attachments: [],
    busy: false,
    canSteer: false,
    clearDraft: vi.fn(),
    disabled: false,
    draftRef: { current: '' },
    drainNextQueued: vi.fn().mockResolvedValue(false),
    editorRef: { current: null },
    exitQueuedEdit: vi.fn().mockReturnValue(false),
    focusInput: vi.fn(),
    inputDisabled: false,
    loadIntoComposer: vi.fn(),
    onCancel: vi.fn(),
    onSteer: undefined,
    onSubmit: vi.fn(),
    queueCurrentDraft: vi.fn().mockReturnValue(false),
    queueEdit: null,
    queuedPrompts: [],
    sessionId: 'session-a',
    setComposerText: vi.fn(),
    stashAt: vi.fn(),
    ...overrides
  }
}

describe('useComposerSubmit dispatchSubmit restore scoping (#66661)', () => {
  it('repaints the composer when the submitting session is still active', async () => {
    const args = buildArgs({ onSubmit: vi.fn().mockResolvedValue(false) })
    const { result } = renderHook(() => useComposerSubmit(args))

    result.current.dispatchSubmit('rejected text')
    await flushMicrotasks()

    expect(args.loadIntoComposer).toHaveBeenCalledWith('rejected text', [])
    expect(args.stashAt).toHaveBeenCalledWith('session-a', 'rejected text', [])
  })

  it('does NOT repaint the live composer after a session switch — only stashes to the submitted scope', async () => {
    const args = buildArgs({ onSubmit: vi.fn().mockResolvedValue(false) })
    const { result } = renderHook(() => useComposerSubmit(args))

    result.current.dispatchSubmit('rejected text')
    // User switches to another session while the gateway is still processing.
    args.activeQueueSessionKeyRef.current = 'session-b'
    await flushMicrotasks()

    // The rejected text must not be painted into session B's live composer…
    expect(args.loadIntoComposer).not.toHaveBeenCalled()
    // …but it must survive in session A's stash.
    expect(args.stashAt).toHaveBeenCalledWith('session-a', 'rejected text', [])
  })

  it('applies the same guard when onSubmit throws instead of returning false', async () => {
    const args = buildArgs({ onSubmit: vi.fn().mockRejectedValue(new Error('gateway down')) })
    const { result } = renderHook(() => useComposerSubmit(args))

    result.current.dispatchSubmit('rejected text')
    args.activeQueueSessionKeyRef.current = 'session-b'
    await flushMicrotasks()

    expect(args.loadIntoComposer).not.toHaveBeenCalled()
    expect(args.stashAt).toHaveBeenCalledWith('session-a', 'rejected text', [])
  })

  it('clears the submitted scope draft on acceptance, without repainting or stashing', async () => {
    const args = buildArgs({ onSubmit: vi.fn().mockResolvedValue(true) })
    const { result } = renderHook(() => useComposerSubmit(args))

    result.current.dispatchSubmit('accepted text')
    await flushMicrotasks()

    expect(clearSessionDraft).toHaveBeenCalledWith('session-a')
    expect(args.loadIntoComposer).not.toHaveBeenCalled()
    expect(args.stashAt).not.toHaveBeenCalled()
  })
})
