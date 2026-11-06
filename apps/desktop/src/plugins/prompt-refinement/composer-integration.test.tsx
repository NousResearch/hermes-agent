/**
 * Prompt Refinement through the REAL composer send path: the draft hook, the
 * submit hook (Enter → clear → dispatch → restore-on-reject), the middleware
 * wrapper that stamps the send origin, and the plugin's fail-closed middleware
 * with a real review store. Only the assistant-ui adapter and the model call
 * are substituted. Proves the draft-integrity contract end to end: a canceled
 * or failed review sends nothing and gives the words and attachments back to
 * the chat they came from — even after a chat switch, and even when the model
 * answers late.
 */

import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { DRAFT_PERSIST_DEBOUNCE_MS, type QueueEditState } from '@/app/chat/composer/composer-utils'
// This test drives the app's real composer hooks, so it reaches past the SDK on purpose.
// eslint-disable-next-line no-restricted-imports
import { COMPOSER_AREAS } from '@/app/chat/composer/contrib'
// eslint-disable-next-line no-restricted-imports
import { useComposerDraft } from '@/app/chat/composer/hooks/use-composer-draft'
// eslint-disable-next-line no-restricted-imports
import { useComposerSubmit } from '@/app/chat/composer/hooks/use-composer-submit'
// eslint-disable-next-line no-restricted-imports
import { useMiddlewareSubmit } from '@/app/chat/composer/hooks/use-middleware-submit'
// eslint-disable-next-line no-restricted-imports
import { composerPlainText, RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
// eslint-disable-next-line no-restricted-imports
import { registry } from '@/contrib/registry'
// eslint-disable-next-line no-restricted-imports
import {
  $composerAttachments,
  clearSessionDraft,
  type ComposerAttachment,
  mainComposerScope,
  takeSessionDraft
} from '@/store/composer'
// eslint-disable-next-line no-restricted-imports
import { composerInputHeld, holdComposerInput } from '@/store/composer-holds'

import { createRefinementMiddleware } from './middleware'
import { createReviewStore, type ReviewStore } from './review-store'
import { createScopeGuard } from './scope-guard'
import { DEFAULT_SETTINGS } from './settings'
import type { RefineResult } from './types'

const runtime = vi.hoisted(() => {
  let text = ''
  const listeners = new Set<() => void>()

  const setText = vi.fn((next: string) => {
    text = next
    listeners.forEach(listener => listener())
  })

  return {
    composer: { setText },
    getState: () => ({ text }),
    reset: () => {
      text = ''
      listeners.clear()
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener)

      return () => listeners.delete(listener)
    }
  }
})

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => runtime.composer }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: runtime.getState() }),
  useComposerRuntime: () => runtime
}))

const CHAT_A = 'refine-int-chat-a'
const CHAT_B = 'refine-int-chat-b'
const PROMPT = 'please refactor the parser in src/parser.ts into small modules'
const REFINED = 'Refactor the parser in src/parser.ts into small, focused modules.'

const ATTACHMENT: ComposerAttachment = { id: 'file:/tmp/spec.md', kind: 'file', label: 'spec.md', path: '/tmp/spec.md' }

const queueEditRef = { current: null as QueueEditState | null }
/** Stands in for the app's submit engine; records what the composer would send. */
type SentOptions = { attachments?: unknown; composerScope?: unknown; typedDraft?: unknown }
const onSubmitProp = vi.fn(async (_text: string, _options?: SentOptions) => true)
let submitDraft: () => void

function Harness({ scope, attachments }: { scope: string; attachments: ComposerAttachment[] }) {
  const draft = useComposerDraft({
    activeQueueSessionKey: scope,
    focusKey: null,
    inputDisabled: false,
    queueEditRef,
    sessionId: scope
  })

  const { onSubmit } = useMiddlewareSubmit(onSubmitProp)

  const submit = useComposerSubmit({
    ...draft,
    activeQueueSessionKey: scope,
    attachments,
    busy: false,
    disabled: false,
    drainNextQueued: async () => false,
    exitQueuedEdit: () => false,
    inputDisabled: false,
    onCancel: () => {},
    onSteer: async () => true,
    onSteerHidden: async () => true,
    onSubmit,
    queueCurrentDraft: () => false,
    queueEdit: null,
    queuedPrompts: [],
    sessionId: scope
  })

  submitDraft = submit.submitDraft

  return (
    <div
      contentEditable
      data-slot={RICH_INPUT_SLOT}
      data-testid="editor"
      ref={draft.editorRef}
      suppressContentEditableWarning
    />
  )
}

function typeAndSave(editor: HTMLElement, text: string) {
  editor.textContent = text
  editor.dispatchEvent(new Event('input', { bubbles: true }))
  act(() => {
    runtime.composer.setText(text)
  })
  act(() => void vi.advanceTimersByTime(DRAFT_PERSIST_DEBOUNCE_MS + 50))
}

const settle = () => act(async () => void (await vi.advanceTimersByTimeAsync(0)))

let store: ReviewStore
let selected: string = CHAT_A
const selectionListeners = new Set<() => void>()
let answer: (result: RefineResult) => void
const disposers: (() => void)[] = []

function selectChat(id: string) {
  selected = id
  selectionListeners.forEach(listener => listener())
}

beforeEach(() => {
  vi.useFakeTimers()
  runtime.reset()
  onSubmitProp.mockClear()
  selected = CHAT_A
  store = createReviewStore()
  disposers.push(store.registerHost())
  disposers.push(store.$review.listen(review => review && store.acknowledge(review.id)))

  const middleware = createRefinementMiddleware({
    settings: () => ({ ...DEFAULT_SETTINGS, mode: 'auto' }),
    review: store.request,
    refine: () => new Promise<RefineResult>(resolve => (answer = resolve)),
    holdComposer: holdComposerInput,
    contextFor: () => '',
    guardScope: (_draft, onDrift) =>
      createScopeGuard(
        {
          current: () => selected,
          listen: listener => (selectionListeners.add(listener), () => selectionListeners.delete(listener))
        },
        onDrift
      ),
    isSlashCommand: text => text.startsWith('/'),
    cancelActiveReview: () => store.cancelActive('scope-changed'),
    notify: () => {}
  })

  disposers.push(registry.register({ id: 'refine-int', area: COMPOSER_AREAS.middleware, data: middleware }))
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  cleanup()
  vi.useRealTimers()
  mainComposerScope.clear()
  clearSessionDraft(CHAT_A)
  clearSessionDraft(CHAT_B)
})

function openReviewIn(attachments: ComposerAttachment[] = [ATTACHMENT]) {
  const view = render(<Harness attachments={attachments} scope={CHAT_A} />)
  const editor = view.getByTestId('editor')
  typeAndSave(editor, PROMPT)
  act(() => submitDraft())

  return { ...view, editor }
}

describe('prompt refinement in the real composer', () => {
  it('refines a typed send first: the composer clears and is held, nothing is submitted yet', async () => {
    const { editor } = openReviewIn()
    await settle()

    expect(store.$review.get()).toMatchObject({
      phase: 'refining',
      target: 'main',
      original: PROMPT,
      attachmentLabels: ['spec.md']
    })
    expect(composerPlainText(editor).trim()).toBe('')
    expect(composerInputHeld('main')).toBe(true)
    expect(onSubmitProp).not.toHaveBeenCalled()
  })

  it('a clean refinement is submitted on its own, once, typed, with the same attachments', async () => {
    openReviewIn()
    await settle()
    answer({ ok: true, text: REFINED, unchanged: false, analysis: null })
    await settle()

    expect(store.$review.get()).toBeNull()
    expect(composerInputHeld('main')).toBe(false)
    expect(onSubmitProp).toHaveBeenCalledTimes(1)
    const [text, options = {}] = onSubmitProp.mock.calls[0]
    expect(text).toBe(REFINED)
    expect(options).toMatchObject({ typedDraft: true, composerScope: CHAT_A })
    expect(options.attachments).toEqual([ATTACHMENT])
  })

  it('Cancel submits nothing and restores the draft and its attachments', async () => {
    const { editor } = openReviewIn()
    await settle()

    store.decide(store.$review.get()!.id, { kind: 'cancel' })
    await settle()

    expect(onSubmitProp).not.toHaveBeenCalled()
    expect(composerInputHeld('main')).toBe(false)
    expect(composerPlainText(editor).trim()).toBe(PROMPT)
    expect($composerAttachments.get()).toEqual([expect.objectContaining({ id: ATTACHMENT.id, label: 'spec.md' })])
    expect(takeSessionDraft(CHAT_A).text).toBe(PROMPT)
  })

  it('a failed refinement submits nothing; Cancel restores the draft and its attachments', async () => {
    const { editor } = openReviewIn()
    await settle()
    answer({ ok: false, failure: 'timeout' })
    await settle()

    expect(store.$review.get()).toMatchObject({ phase: 'failed', attention: 'failed' })
    expect(onSubmitProp).not.toHaveBeenCalled()

    store.decide(store.$review.get()!.id, { kind: 'cancel' })
    await settle()

    expect(onSubmitProp).not.toHaveBeenCalled()
    expect(composerPlainText(editor).trim()).toBe(PROMPT)
    expect($composerAttachments.get()).toEqual([expect.objectContaining({ id: ATTACHMENT.id, label: 'spec.md' })])
  })

  it('a rewrite that asks for more than the prompt (a delete) is not submitted on its own', async () => {
    openReviewIn()
    await settle()
    answer({ ok: true, text: `${REFINED} Delete the old drafts first.`, unchanged: false, analysis: null })
    await settle()

    expect(store.$review.get()).toMatchObject({ phase: 'ready', attention: 'issues' })
    expect(onSubmitProp).not.toHaveBeenCalled()
  })

  it('a refinement that drops the file path is not sent; Send original sends the draft as typed', async () => {
    openReviewIn()
    await settle()
    answer({ ok: true, text: 'Refactor the parser into small, focused modules.', unchanged: false, analysis: null })
    await settle()

    const review = store.$review.get()!
    expect(review).toMatchObject({ phase: 'ready', attention: 'issues' })
    expect(onSubmitProp).not.toHaveBeenCalled()

    store.decide(review.id, { kind: 'original' })
    await settle()

    expect(onSubmitProp).toHaveBeenCalledTimes(1)
    expect(onSubmitProp.mock.calls[0][0]).toBe(PROMPT)
  })

  it.each(['timeout', 'offline', 'unavailable'] as const)(
    'a %s refinement stops: dismissing it sends nothing, and the draft comes back',
    async failure => {
      const { editor } = openReviewIn()
      await settle()
      answer({ ok: false, failure })
      await settle()

      const review = store.$review.get()!
      expect(review.phase).toBe('failed')
      expect(store.decide(review.id, { kind: 'approve', text: PROMPT, acknowledged: true })).toBe(false)
      store.decide(review.id, { kind: 'cancel' })
      await settle()

      expect(onSubmitProp).not.toHaveBeenCalled()
      expect(composerPlainText(editor).trim()).toBe(PROMPT)
    }
  )

  it('switching chats mid-review cancels it: A keeps its draft, B is not overwritten', async () => {
    const { editor, rerender } = openReviewIn()
    await settle()

    // The user moves to chat B and starts typing there while A's review is open.
    rerender(<Harness attachments={[]} scope={CHAT_B} />)
    typeAndSave(editor, 'a different thought for chat B')
    selectChat(CHAT_B)
    await settle()

    expect(store.$review.get()).toBeNull()
    expect(onSubmitProp).not.toHaveBeenCalled()
    expect(composerPlainText(editor).trim()).toBe('a different thought for chat B')
    expect(takeSessionDraft(CHAT_B).text).toBe('a different thought for chat B')
    expect(takeSessionDraft(CHAT_A).text).toBe(PROMPT)

    rerender(<Harness attachments={[]} scope={CHAT_A} />)
    expect(composerPlainText(editor).trim()).toBe(PROMPT)
  })

  it('a model answer that lands after Cancel never sends anything', async () => {
    const { editor } = openReviewIn()
    await settle()

    store.decide(store.$review.get()!.id, { kind: 'cancel' })
    answer({ ok: true, text: REFINED, unchanged: false, analysis: null })
    await settle()

    expect(onSubmitProp).not.toHaveBeenCalled()
    expect(store.$review.get()).toBeNull()
    expect(composerPlainText(editor).trim()).toBe(PROMPT)
  })

  it('a slash command and a short prompt go straight through, unreviewed', async () => {
    const view = render(<Harness attachments={[]} scope={CHAT_A} />)
    const editor = view.getByTestId('editor')

    typeAndSave(editor, 'ok thanks')
    act(() => submitDraft())
    await settle()

    typeAndSave(editor, '/compact keep the decisions about the parser')
    act(() => submitDraft())
    await settle()

    expect(store.$review.get()).toBeNull()
    expect(onSubmitProp.mock.calls.map(([text]) => text)).toEqual([
      'ok thanks',
      '/compact keep the decisions about the parser'
    ])
  })
})
