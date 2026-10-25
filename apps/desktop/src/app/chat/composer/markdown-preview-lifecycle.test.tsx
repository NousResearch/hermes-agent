// @vitest-environment jsdom
import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $busyInputConfig, busyInputOwnerKey } from '@/store/busy-input-mode'
import { clearSessionDraft, mainComposerScope } from '@/store/composer'
import { clearQueuedPrompts, getQueuedPrompts } from '@/store/composer-queue'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

const QUEUE_KEY = 'markdown-preview-lifecycle'
const DRAFT = '# Latest draft\n\nKeep **these words**.'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness(props: Partial<ChatBarProps>) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: props.busy ?? false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            queueSessionKey={QUEUE_KEY}
            state={state}
            {...props}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

const previousGatewayState = $gatewayState.get()
const previousConnection = $connection.get()
const previousBusyConfig = $busyInputConfig.get()

beforeEach(() => {
  mainComposerScope.clear()
  clearSessionDraft(QUEUE_KEY)
  clearSessionDraft(null)
  clearQueuedPrompts(QUEUE_KEY)
  $connection.set(null)
  $gatewayState.set('open')
  $busyInputConfig.set({
    owner: busyInputOwnerKey(null, $activeGatewayProfile.get()),
    connection: null,
    mode: 'interrupt'
  })
})

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft(QUEUE_KEY)
  clearSessionDraft(null)
  clearQueuedPrompts(QUEUE_KEY)
  $busyInputConfig.set(previousBusyConfig)
  $connection.set(previousConnection)
  $gatewayState.set(previousGatewayState)
})

function editorIn(container: HTMLElement) {
  return container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
}

async function enterPreview(container: HTMLElement, text = DRAFT) {
  const editor = editorIn(container)

  // No input event: the final keystroke has reached the DOM but the coalesced
  // draft flush has not run. Preview must synchronously capture that text.
  await act(async () => {
    editor.textContent = text
    fireEvent.click(screen.getByRole('button', { name: 'Preview Markdown' }))
  })

  expect(container.querySelector('[data-slot="composer-markdown-preview"]')).toBeTruthy()
  expect(editor.classList.contains('hidden')).toBe(true)

  return editor
}

function expectEditing(container: HTMLElement) {
  expect(container.querySelector('[data-slot="composer-markdown-preview"]')).toBeNull()
  expect(editorIn(container).classList.contains('hidden')).toBe(false)
  expect(screen.getByRole('button', { name: 'Preview Markdown' }).getAttribute('aria-pressed')).toBe('false')
}

describe('ChatBar Markdown preview lifecycle', () => {
  it('captures the latest editor text and returns to editing without submitting', async () => {
    const onSubmit = vi.fn(async () => true)
    const onSteer = vi.fn(async () => true)
    const { container } = render(<Harness onSteer={onSteer} onSubmit={onSubmit} />)
    const editor = await enterPreview(container)

    expect(await screen.findByRole('heading', { name: 'Latest draft' })).toBeTruthy()
    expect(screen.getByText('these words').getAttribute('data-streamdown')).toBe('strong')
    fireEvent.click(screen.getByRole('button', { name: 'Edit Markdown' }))

    expectEditing(container)
    expect(composerPlainText(editor)).toBe(DRAFT)
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onSteer).not.toHaveBeenCalled()
    expect(getQueuedPrompts(QUEUE_KEY)).toEqual([])
  })

  it('exits preview and queues the synchronized draft through Queue', async () => {
    const onSubmit = vi.fn(async () => true)
    const onSteer = vi.fn(async () => true)
    const { container } = render(<Harness busy onSteer={onSteer} onSubmit={onSubmit} />)
    const editor = await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Queue message' }))

    expectEditing(container)
    expect(getQueuedPrompts(QUEUE_KEY)).toEqual([expect.objectContaining({ text: DRAFT, attachments: [] })])
    expect(composerPlainText(editor).trim()).toBe('')
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('exits preview and sends the synchronized draft through Send', async () => {
    const onSubmit = vi.fn(async () => true)
    const { container } = render(<Harness onSubmit={onSubmit} />)
    const editor = await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Send' }))

    expectEditing(container)
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(DRAFT, expect.objectContaining({ attachments: [] })))
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(composerPlainText(editor).trim()).toBe('')
  })

  it('exits preview before independently steering with the steer mode', async () => {
    const onSubmit = vi.fn(async () => true)
    const onSteer = vi.fn(async () => true)
    const { container } = render(<Harness busy onSteer={onSteer} onSubmit={onSubmit} />)
    const editor = await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Steer the current run' }))

    expectEditing(container)
    await waitFor(() => expect(onSteer).toHaveBeenCalledWith(DRAFT, 'steer'))
    expect(onSteer).toHaveBeenCalledTimes(1)
    expect(composerPlainText(editor).trim()).toBe('')
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(QUEUE_KEY)).toEqual([])
  })

  it('exits preview and retains the configured interrupt behavior for busy Send', async () => {
    const onSteer = vi.fn(async () => true)
    const onSubmit = vi.fn(async () => true)
    const { container } = render(<Harness busy onSteer={onSteer} onSubmit={onSubmit} />)
    await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Redirect the current run' }))

    expectEditing(container)
    await waitFor(() => expect(onSteer).toHaveBeenCalledWith(DRAFT, 'interrupt'))
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('restores the editable draft when Send is rejected', async () => {
    const onSubmit = vi.fn(async () => false)
    const { container } = render(<Harness onSubmit={onSubmit} />)
    const editor = await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Send' }))

    expectEditing(container)
    await waitFor(() => expect(composerPlainText(editor)).toBe(DRAFT))
    expect(onSubmit).toHaveBeenCalledWith(DRAFT, expect.objectContaining({ attachments: [] }))
    expect(getQueuedPrompts(QUEUE_KEY)).toEqual([])
  })

  it.each(['refused', 'failed'] as const)('preserves the draft in the queue when steering is %s', async outcome => {
    const onSteer = vi.fn(async () => {
      if (outcome === 'failed') {
        throw new Error('Steering unavailable')
      }

      return false
    })

    const onSubmit = vi.fn(async () => true)
    const { container } = render(<Harness busy onSteer={onSteer} onSubmit={onSubmit} />)
    await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Steer the current run' }))

    expectEditing(container)
    await waitFor(() =>
      expect(getQueuedPrompts(QUEUE_KEY)).toEqual([expect.objectContaining({ text: DRAFT, attachments: [] })])
    )
    expect(onSteer).toHaveBeenCalledWith(DRAFT, 'steer')
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('restores the editable draft when steering is refused before a queue exists', async () => {
    const onSteer = vi.fn(async () => false)
    const { container } = render(<Harness busy onSteer={onSteer} queueSessionKey={null} />)
    const editor = await enterPreview(container)

    fireEvent.click(screen.getByRole('button', { name: 'Steer the current run' }))

    expectEditing(container)
    await waitFor(() => expect(composerPlainText(editor)).toBe(DRAFT))
    expect(onSteer).toHaveBeenCalledWith(DRAFT, 'steer')
    expect(getQueuedPrompts(QUEUE_KEY)).toEqual([])
  })

  it('keeps the independent Steer unavailable for slash commands in preview', async () => {
    const onSteer = vi.fn(async () => true)
    const { container } = render(<Harness busy onSteer={onSteer} />)
    await enterPreview(container, '/help')

    expect(screen.queryByRole('button', { name: 'Steer the current run' })).toBeNull()
    expect(onSteer).not.toHaveBeenCalled()
  })
})
