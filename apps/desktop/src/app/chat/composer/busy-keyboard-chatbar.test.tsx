// @vitest-environment jsdom
import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, createEvent, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { comboTokens, formatCombo } from '@/lib/keybinds/combo'
import { $busyInputConfig, type BusyInputMode, busyInputOwnerKey } from '@/store/busy-input-mode'
import { $clarifyRequests } from '@/store/clarify'
import { $compactingSessions, setSessionCompacting } from '@/store/compaction'
import { clearSessionDraft, type ComposerAttachment, createComposerAttachmentScope } from '@/store/composer'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  parkQueuedPrompts
} from '@/store/composer-queue'
import { $gateway } from '@/store/gateway'
import { $onboardingPhase } from '@/store/onboarding-phase'
import { $onboardingSession } from '@/store/onboarding-session'
import { $activeGatewayProfile } from '@/store/profile'
import { clearAllPrompts, hasBlockingPromptRequest, setApprovalRequest } from '@/store/prompts'
import { hasOpenServerRequest, rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $connection, $gatewayState, setSessionsLoading } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from './scope'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

// Keep the real editor, AUI runtime, busy-policy hook, draft/submit/queue hooks,
// queue stores, parked-card routing, controls, help and tooltips. Only peripheral
// media, layout and status work is inert; no handler is reproduced by the test.
vi.mock('./hooks/use-composer-voice', () => ({
  useComposerVoice: () => ({
    conversation: { level: 0, muted: false, status: 'idle', stopTurn() {}, toggleMute() {} },
    dictate() {},
    endConversation() {},
    handleToggleAutoSpeak() {},
    startConversation() {},
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceConversationActive: false,
    voiceStatus: 'idle'
  })
}))
vi.mock('./hooks/use-composer-screenshot', () => ({ useComposerScreenshot() {} }))
vi.mock('./hooks/use-composer-branch', () => ({
  useComposerBranch: () => ({
    handleBranchOff() {}, handleConvertBranch() {}, handleListBranches() {}, handleSwitchBranch() {}, openInWorktree() {}
  })
}))
vi.mock('./hooks/use-composer-metrics', () => ({
  useComposerMetrics: () => ({ compactPill: true, foldVoice: true, minimal: true, stacked: false })
}))
vi.mock('./hooks/use-status-presence', () => ({ useSessionStatusPresence: () => false }))
vi.mock('./hooks/use-micro-actions', () => ({ useComposerMicroActions() {} }))
vi.mock('./status-stack', () => ({ ComposerStatusStack: ({ queue }: { queue: ReactNode }) => queue }))
vi.mock('./status-stack/coding-row', () => ({ CodingStatusRow: () => null }))
vi.mock('./prepared-image-recovery', () => ({ PreparedImageRecovery: () => null }))

const SESSION = 'busy-keyboard-stored'
const RUNTIME = 'busy-keyboard-runtime'
const OTHER = 'busy-keyboard-other'
const PAYLOAD = 'please check the failing import'
const EXISTING = { id: 'already-queued', text: 'keep this queued', attachments: [] }
const FILE: ComposerAttachment = { id: 'notes-file', kind: 'file', label: 'notes.txt' }

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

const MODES: BusyInputMode[] = ['queue', 'steer', 'interrupt']

const CHORDS: { name: string; init: KeyboardEventInit }[] = [
  { name: 'Enter', init: {} },
  { name: 'Cmd+Enter', init: { metaKey: true } },
  { name: 'Ctrl+Enter', init: { ctrlKey: true } }
]

type Action = 'queue' | 'redirect' | 'steer'

// An independent behavior table: do not derive expectations from the production
// alternateBusyInputMode helper (that would bless a broken mapping).
const MATRIX: { mode: BusyInputMode; name: string; init: KeyboardEventInit; action: Action }[] = [
  { mode: 'queue', ...CHORDS[0], action: 'queue' },
  { mode: 'queue', ...CHORDS[1], action: 'redirect' },
  { mode: 'queue', ...CHORDS[2], action: 'redirect' },
  { mode: 'steer', ...CHORDS[0], action: 'steer' },
  { mode: 'steer', ...CHORDS[1], action: 'queue' },
  { mode: 'steer', ...CHORDS[2], action: 'queue' },
  { mode: 'interrupt', ...CHORDS[0], action: 'redirect' },
  { mode: 'interrupt', ...CHORDS[1], action: 'queue' },
  { mode: 'interrupt', ...CHORDS[2], action: 'queue' }
]

function setMode(mode: BusyInputMode) {
  $busyInputConfig.set({
    connection: $connection.get(),
    owner: busyInputOwnerKey($connection.get()?.connectionId, $activeGatewayProfile.get()),
    mode
  })
}

function mountChatBar(mode: BusyInputMode, overrides: Partial<ChatBarProps> = {}) {
  setMode(mode)
  const attachments = createComposerAttachmentScope()
  const scope = { ...MAIN_COMPOSER_SCOPE, attachments, $awaitingInput: atom(false) }
  const onCancel = vi.fn<ChatBarProps['onCancel']>()
  const onSteer = vi.fn<NonNullable<ChatBarProps['onSteer']>>().mockResolvedValue(true)
  const onSteerHidden = vi.fn<NonNullable<ChatBarProps['onSteerHidden']>>().mockResolvedValue(true)
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>().mockResolvedValue(true)

  const props: ChatBarProps = {
    busy: true, disabled: false, gateway: null, queueSessionKey: SESSION, sessionId: RUNTIME,
    state, onCancel, onSteer, onSteerHidden, onSubmit, ...overrides
  }

  let composer!: ReturnType<typeof useExternalStoreRuntime>['thread']['composer']

  function Harness({ chatProps }: { chatProps: ChatBarProps }) {
    const runtime = useExternalStoreRuntime({
      convertMessage: (message: ThreadMessageLike) => message,
      messages: [] as ThreadMessageLike[],
      isRunning: chatProps.busy,
      onNew: async () => {}
    })

    composer = runtime.thread.composer

    return (
      <AssistantRuntimeProvider runtime={runtime}>
        <MemoryRouter>
          <I18nProvider configClient={null} initialLocale="en">
            <ComposerScopeProvider value={scope}>
              <ChatBar {...chatProps} />
            </ComposerScopeProvider>
          </I18nProvider>
        </MemoryRouter>
      </AssistantRuntimeProvider>
    )
  }

  const view = render(<Harness chatProps={props} />)
  const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  expect(editor).toBeTruthy()
  // jsdom does not make a contentEditable focusable/isContentEditable itself.
  editor.tabIndex = 0
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  act(() => editor.focus())

  return {
    ...view, editor, attachments, onCancel, onSteer, onSteerHidden, onSubmit,
    composer: () => composer,
    update(next: Partial<ChatBarProps>) {
      Object.assign(props, next)
      view.rerender(<Harness chatProps={{ ...props }} />)
    }
  }
}

type Mounted = ReturnType<typeof mountChatBar>

async function writeDraft(view: Mounted, text = PAYLOAD, syncInput = true) {
  act(() => {
    view.editor.textContent = text

    if (syncInput) {
      fireEvent.input(view.editor, { inputType: 'insertText' })
    }
  })

  if (syncInput) {
    await waitFor(() => expect(view.composer().getState().text).toBe(text))
  }
}

async function press(view: Mounted, init: KeyboardEventInit = {}) {
  const event = createEvent.keyDown(view.editor, {
    key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true, cancelable: true, ...init
  })

  await act(async () => { fireEvent(view.editor, event) })

  return event
}

function expectNoDispatch(view: Mounted) {
  expect(view.onCancel).not.toHaveBeenCalled()
  expect(view.onSubmit).not.toHaveBeenCalled()
  expect(view.onSteer).not.toHaveBeenCalled()
  expect(view.onSteerHidden).not.toHaveBeenCalled()
}

function expectOtherQueueUntouched() {
  expect(getQueuedPrompts(OTHER).map(entry => [entry.id, entry.text])).toEqual([['other-entry', 'other conversation']])
}

function expectOriginalQueue() {
  expect(getQueuedPrompts(SESSION).map(entry => [entry.id, entry.text])).toEqual([[EXISTING.id, EXISTING.text]])
  expectOtherQueueUntouched()
}

function expectAction(view: Mounted, action: Action, text = PAYLOAD, files: ComposerAttachment[] = []) {
  expect(view.onCancel).not.toHaveBeenCalled()
  expect(view.onSubmit).not.toHaveBeenCalled()

  if (action === 'queue') {
    expectNoDispatch(view)
    const entries = getQueuedPrompts(SESSION)
    expect(entries).toHaveLength(2)
    expect(entries[0]).toMatchObject(EXISTING)
    expect(entries[1]).toMatchObject({ text, attachments: files })
  } else {
    expectOriginalQueue()

    if (action === 'redirect') {
      expect(view.onSteer).toHaveBeenCalledExactlyOnceWith(text, 'interrupt')
      expect(view.onSteerHidden).not.toHaveBeenCalled()
    } else {
      expect(view.onSteerHidden).toHaveBeenCalledExactlyOnceWith(text)
      expect(view.onSteer).not.toHaveBeenCalled()
    }
  }

  expect(composerPlainText(view.editor)).toBe('')
  expect(view.attachments.$attachments.get()).toEqual([])
  expectOtherQueueUntouched()
}

beforeEach(() => {
  localStorage.clear()
  $connection.set(null)
  $gateway.set(null)
  $gatewayState.set('open')
  $activeGatewayProfile.set('default')
  $sessionStates.set({})
  $sessionTiles.set([])
  $compactingSessions.set({})
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  clearAllPrompts()
  setSessionsLoading(false)
  enqueueQueuedPrompt(SESSION, EXISTING)
  enqueueQueuedPrompt(OTHER, { id: 'other-entry', text: 'other conversation', attachments: [] })
  // Keep idle tests' pre-existing queue from auto-draining on mount. A new
  // payload must still send normally; busy tests must never deliver this row.
  parkQueuedPrompts(SESSION)
})

afterEach(() => {
  cleanup()
  clearSessionDraft(SESSION)
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  $compactingSessions.set({})
  $sessionStates.set({})
  $sessionTiles.set([])
  $busyInputConfig.set(null)
  $clarifyRequests.set({})
  $onboardingSession.set(null)
  $onboardingPhase.set('idle')
  resetServerRequestsForTests()
  clearAllPrompts()
  $gateway.set(null)
  $connection.set(null)
  $gatewayState.set('idle')
  localStorage.clear()
  vi.restoreAllMocks()
})

describe('real ChatBar busy Enter routing (#121383)', () => {
  describe.each([
    { source: 'input-synchronized draft', syncInput: true },
    { source: 'live DOM with no input event', syncInput: false }
  ])('$source', ({ syncInput }) => {
    it.each(MATRIX)('$mode / $name => $action, without another carrier or queue drain', async ({ mode, init, action }) => {
      const view = mountChatBar(mode)
      const config = $busyInputConfig.get()
      await writeDraft(view, PAYLOAD, syncInput)

      if (!syncInput) {
        expect(view.composer().getState().text).toBe('')
      }

      expect((await press(view, init)).defaultPrevented).toBe(true)
      expectAction(view, action)
      // A modifier overrides this submission, never the session preference.
      expect($busyInputConfig.get()).toBe(config)
    })
  })

  it.each(MATRIX)('$mode / Shift+$name leaves native newline editing enabled', async ({ mode, init }) => {
    const view = mountChatBar(mode)
    await writeDraft(view)
    expect((await press(view, { ...init, shiftKey: true })).defaultPrevented).toBe(false)
    // jsdom does not execute native contentEditable editing. An unprevented
    // event, with no submit side effect, is the browser newline contract.
    expectNoDispatch(view)
    expectOriginalQueue()
    expect(composerPlainText(view.editor)).toBe(PAYLOAD)
  })

  it.each(CHORDS)('idle Shift+$name also leaves native newline editing enabled', async ({ init }) => {
    const view = mountChatBar('queue', { busy: false })
    await writeDraft(view)
    expect((await press(view, { ...init, shiftKey: true })).defaultPrevented).toBe(false)
    expectNoDispatch(view)
    expectOriginalQueue()
  })

  describe.each(['keyCode 229', 'native composition'] as const)('%s', guard => {
    it.each(CHORDS)('$name confirms IME text without sending; the next real Enter still works', async ({ init }) => {
      const view = mountChatBar('queue')
      await writeDraft(view, '确认输入', false)

      if (guard === 'native composition') {
        act(() => fireEvent.compositionStart(view.editor))
      }

      const event = await press(view, {
        ...init,
        ...(guard === 'keyCode 229' ? { keyCode: 229 } : { isComposing: true })
      })

      expect(event.defaultPrevented).toBe(false)
      expectNoDispatch(view)
      expectOriginalQueue()
      expect(composerPlainText(view.editor)).toBe('确认输入')

      if (guard === 'native composition') {
        act(() => fireEvent.compositionEnd(view.editor))
      }

      await press(view, init)
      expectAction(view, init.metaKey || init.ctrlKey ? 'redirect' : 'queue', '确认输入')
    })
  })
})

describe('real ChatBar priority and availability boundaries', () => {
  describe.each([
    { name: 'disabled while open', gatewayState: 'open' as const, editable: 'false' },
    { name: 'reconnecting', gatewayState: 'closed' as const, editable: 'true' }
  ])('$name', ({ gatewayState, editable }) => {
    it.each(MATRIX)('$mode / $name preserves the draft and dispatches nothing', async ({ mode, init, action }) => {
      $gatewayState.set(gatewayState)
      const view = mountChatBar(mode, { disabled: true })
      await writeDraft(view)
      expect(view.editor.getAttribute('contenteditable')).toBe(editable)
      await press(view, init)
      expectNoDispatch(view)
      expectOriginalQueue()
      expect(composerPlainText(view.editor)).toBe(PAYLOAD)

      // The same mounted editor must become usable after the connection returns.
      act(() => $gatewayState.set('open'))
      view.update({ disabled: false })
      await press(view, init)
      expectAction(view, action)
    })
  })

  it.each(MODES)('idle Enter sends live DOM text in %s mode without draining a parked queue', async mode => {
    const view = mountChatBar(mode, { busy: false })
    await writeDraft(view, PAYLOAD, false)
    await press(view)
    expect(view.onSubmit).toHaveBeenCalledExactlyOnceWith(PAYLOAD, expect.objectContaining({
      composerScope: SESSION, attachments: [], submission_id: expect.any(String)
    }))
    expect(view.onSubmit.mock.calls[0][1]?.fromQueue).not.toBe(true)
    expect(view.onCancel).not.toHaveBeenCalled()
    expect(view.onSteer).not.toHaveBeenCalled()
    expect(view.onSteerHidden).not.toHaveBeenCalled()
    expectOriginalQueue()
    expect(composerPlainText(view.editor)).toBe('')
  })

  it.each(CHORDS)('idle $name keeps the existing modifier behavior', async ({ init }) => {
    const view = mountChatBar('queue', { busy: false })
    await writeDraft(view)
    await press(view, init)

    if (init.metaKey || init.ctrlKey) {
      // #121383 overrides busy submission only; it does not adopt #117752.
      expectNoDispatch(view)
      expect(composerPlainText(view.editor)).toBe(PAYLOAD)
    } else {
      expect(view.onSubmit).toHaveBeenCalledTimes(1)
      expect(view.onSubmit.mock.calls[0][0]).toBe(PAYLOAD)
      expect(view.onCancel).not.toHaveBeenCalled()
      expect(view.onSteer).not.toHaveBeenCalled()
      expect(view.onSteerHidden).not.toHaveBeenCalled()
    }

    expectOriginalQueue()
  })

  it.each(MATRIX)('$mode / $name answers a real parked clarify card before busy routing', async ({ mode, init }) => {
    const respond = vi.fn()
    const request = vi.fn(async () => ({ status: 'ok', remaining: [] }))
    rememberServerRequest({ id: 'keyboard-clarify', method: 'clarify', params: {}, fail: vi.fn(), respond })
    $clarifyRequests.set({ [RUNTIME]: {
      requestId: 'keyboard-clarify', sessionId: RUNTIME,
      questions: [{ qid: 'destination', question: 'Which environment?', choices: ['Staging', 'Prod'], multiSelect: false }]
    } })
    $gateway.set({ request } as never)
    const view = mountChatBar(mode)
    await writeDraft(view, '  prod  ', false)
    await press(view, init)
    expect(request).toHaveBeenCalledExactlyOnceWith('clarify.lock', {
      answer: 'Prod', question_id: 'destination', request_id: 'keyboard-clarify'
    })
    expect($clarifyRequests.get()[RUNTIME]).toBeUndefined()
    expect(respond).not.toHaveBeenCalled()
    expectNoDispatch(view)
    expectOriginalQueue()
    expect(composerPlainText(view.editor)).toBe('')
  })

  it.each(MATRIX)('$mode / $name executes a slash command immediately', async ({ mode, init }) => {
    const view = mountChatBar(mode)
    // No input event: exercise submission, without accepting a completion first.
    await writeDraft(view, '/status', false)
    await press(view, init)
    expect(view.onSubmit).toHaveBeenCalledExactlyOnceWith('/status', expect.objectContaining({ composerScope: SESSION }))
    expect(view.onSubmit.mock.calls[0][1]?.fromQueue).not.toBe(true)
    expect(view.onCancel).not.toHaveBeenCalled()
    expect(view.onSteer).not.toHaveBeenCalled()
    expect(view.onSteerHidden).not.toHaveBeenCalled()
    expectOriginalQueue()
    expect(composerPlainText(view.editor)).toBe('')
  })

  it.each(MATRIX)('$mode / $name saves a queued edit in place', async ({ mode, init }) => {
    const before = getQueuedPrompts(SESSION)[0]
    const view = mountChatBar(mode)
    await press(view, { key: 'ArrowUp', code: 'ArrowUp' })
    expect(composerPlainText(view.editor)).toBe(EXISTING.text)
    await writeDraft(view, 'revised queued message', false)
    await press(view, init)
    expect(getQueuedPrompts(SESSION)).toEqual([{ ...before, text: 'revised queued message' }])
    expectNoDispatch(view)
    expectOtherQueueUntouched()
    expect(composerPlainText(view.editor)).toBe('')
  })
})

describe('real ChatBar correction fallbacks', () => {
  const corrections = MATRIX.filter(row => row.action !== 'queue')

  it.each(corrections)('$mode / $name queues when its carrier is absent, even if the other exists', async ({ mode, init, action }) => {
    const view = mountChatBar(mode, action === 'steer' ? { onSteerHidden: undefined } : { onSteer: undefined })
    await writeDraft(view)
    await press(view, init)
    expectAction(view, 'queue')
  })

  it.each(corrections)('$mode / $name queues during compaction', async ({ mode, init }) => {
    setSessionCompacting(RUNTIME, true)
    const view = mountChatBar(mode)
    await writeDraft(view)
    await press(view, init)
    expectAction(view, 'queue')
    expect($compactingSessions.get()[RUNTIME]).toBe(true)
  })

  it.each(corrections)('$mode / $name queues attachments with the text', async ({ mode, init }) => {
    const view = mountChatBar(mode)
    act(() => view.attachments.$attachments.set([FILE]))
    await writeDraft(view)
    await press(view, init)
    expectAction(view, 'queue', PAYLOAD, [FILE])
  })

  it.each(corrections)('$mode / $name queues behind an approval without answering or dismissing it', async ({ mode, init }) => {
    const respond = vi.fn()
    rememberServerRequest({ id: 'keyboard-approval', method: 'approval', params: {}, fail: vi.fn(), respond })
    setApprovalRequest({ requestId: 'keyboard-approval', sessionId: RUNTIME, command: 'test operation', description: 'Allow?' })
    const view = mountChatBar(mode)
    await writeDraft(view)
    await press(view, init)
    expectAction(view, 'queue')
    expect(hasBlockingPromptRequest(RUNTIME)).toBe(true)
    expect(hasOpenServerRequest('keyboard-approval')).toBe(true)
    expect(respond).not.toHaveBeenCalled()
  })

  describe.each(['refused', 'rejected', 'synchronous throw'] as const)('%s correction', failure => {
    it.each(corrections)('$mode / $name retains exactly one queued copy and never tries the other carrier', async ({ mode, init, action }) => {
      const view = mountChatBar(mode)
      const carrier = action === 'steer' ? view.onSteerHidden : view.onSteer

      if (failure === 'refused') {
        carrier.mockResolvedValue(false)
      } else if (failure === 'rejected') {
        carrier.mockRejectedValue(new Error('isolated test rejection'))
      } else {
        carrier.mockImplementation(() => { throw new Error('isolated synchronous rejection') })
      }

      await writeDraft(view, PAYLOAD, false)
      await press(view, init)
      expect(carrier).toHaveBeenCalledTimes(1)

      if (action === 'redirect') {
        expect(view.onSteer).toHaveBeenCalledWith(PAYLOAD, 'interrupt')
        expect(view.onSteerHidden).not.toHaveBeenCalled()
      } else {
        expect(view.onSteerHidden).toHaveBeenCalledWith(PAYLOAD)
        expect(view.onSteer).not.toHaveBeenCalled()
      }

      expect(view.onSubmit).not.toHaveBeenCalled()
      expect(view.onCancel).not.toHaveBeenCalled()
      expect(getQueuedPrompts(SESSION).map(entry => entry.text)).toEqual([EXISTING.text, PAYLOAD])
      expectOtherQueueUntouched()
      expect(composerPlainText(view.editor)).toBe('')
    })
  })
})

describe('ChatBar help and primary tooltip describe the actual busy action', () => {
  it.each([
    { mode: 'queue' as const, primary: en.composer.queueMessage, enter: 'composer.queue', alternate: 'composer.redirect' },
    { mode: 'steer' as const, primary: en.composer.steer, enter: 'composer.steer', alternate: 'composer.queue' },
    { mode: 'interrupt' as const, primary: en.composer.redirect, enter: 'composer.redirect', alternate: 'composer.queue' }
  ])('$mode labels Enter and mod+Enter correctly', async ({ mode, primary, enter, alternate }) => {
    const view = mountChatBar(mode)
    await writeDraft(view, '?')
    const dialog = screen.getByRole('dialog')
    const descriptions: Record<string, string> = en.composer.hotkeyDescs
    const actions: Record<string, string> = en.keybinds.actions
    const label = (id: string) => descriptions[id] ?? actions[id]
    const enterRow = within(dialog).getByText(label(enter)).parentElement!
    const alternateRow = within(dialog).getByText(label(alternate)).parentElement!
    expect(Array.from(enterRow.querySelectorAll('kbd'), key => key.textContent)).toEqual(comboTokens('enter'))
    expect(Array.from(alternateRow.querySelectorAll('kbd'), key => key.textContent)).toEqual(comboTokens('mod+enter'))

    // Read the real controls/Tip, so a correct helper with a stale ChatBar prop
    // or a misleading primary-button tooltip still fails this integration.
    await writeDraft(view)
    const button = screen.getByRole('button', { name: primary })
    expect(button.getAttribute('type')).toBe('submit')
    fireEvent.pointerMove(button, { pointerType: 'mouse' })
    const tooltip = await screen.findByRole('tooltip')
    expect(tooltip.textContent).toContain(primary)
    expect(tooltip.textContent).toContain(formatCombo('enter'))
    expectNoDispatch(view)
    expectOriginalQueue()
  })

  it('advertises Queue for both keys when queue-mode redirect has no carrier', async () => {
    const view = mountChatBar('queue', { onSteer: undefined })
    await writeDraft(view, '?')
    const dialog = screen.getByRole('dialog')
    const rows = within(dialog).getAllByText(en.keybinds.actions['composer.queue']).map(node => node.parentElement!)
    expect(rows).toHaveLength(2)
    expect(rows.map(row => Array.from(row.querySelectorAll('kbd'), key => key.textContent))).toEqual([
      comboTokens('enter'), comboTokens('mod+enter')
    ])
    expect(within(dialog).queryByText(en.keybinds.actions['composer.redirect'])).toBeNull()
    expectNoDispatch(view)
    expectOriginalQueue()
  })
})
