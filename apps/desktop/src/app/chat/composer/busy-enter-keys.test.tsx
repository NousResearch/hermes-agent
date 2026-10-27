import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { clearSessionDraft, mainComposerScope } from '@/store/composer'
import { clearQueuedPrompts, getQueuedPrompts } from '@/store/composer-queue'

import { markActiveComposer } from './focus'
import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

vi.mock('@/app/session/hooks/use-busy-input-mode', () => ({ useBusyInputMode: () => 'interrupt' }))

const SESSION = 'test-busy-enter-keys'

let removeBehavior = () => {}
beforeEach(() => {
  removeBehavior = registry.register({
    id: 'busy-key-fixture',
    source: 'plugin:neo-next-prompt',
    area: 'composer.behavior.v2',
    data: {
      protocol: 2,
      create: () => ({}),
      keyAction: (e: {
        key: string
        busy: boolean
        ctrlKey?: boolean
        metaKey?: boolean
        altKey?: boolean
        shiftKey?: boolean
      }) =>
        e.key === 'Enter' && e.busy && !e.shiftKey && !e.altKey
          ? e.ctrlKey || e.metaKey
            ? 'steer'
            : 'queue'
          : undefined
    }
  })
})
afterEach(() => removeBehavior())

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
            busy={true}
            disabled={false}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            sessionId={SESSION}
            state={state}
            {...props}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

function openComposer(props: Partial<ChatBarProps> = {}) {
  const { container } = render(<Harness {...props} />)
  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  editor.tabIndex = 0

  return editor
}

async function pressEnter(editor: HTMLElement, text: string, modifiers: KeyboardEventInit & { keyCode?: number } = {}) {
  await act(async () => {
    editor.focus()
    // Deliberately omit input: the real handler must use the latest DOM text,
    // even when a fast keystroke precedes the composer state's next render.
    editor.textContent = text
    fireEvent.keyDown(editor, { key: 'Enter', ...modifiers })
  })
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft(SESSION)
  clearQueuedPrompts(SESSION)
})

describe('busy composer Enter preference', () => {
  it('restores native Enter steering when the plugin is disabled', async () => {
    removeBehavior()
    const onSteer = vi.fn(async () => true)
    const editor = openComposer({ onSteer })
    await pressEnter(editor, 'native steering')
    expect(onSteer).toHaveBeenCalledExactlyOnceWith('native steering', 'interrupt')
    expect(getQueuedPrompts(SESSION)).toEqual([])
  })
  it('queues plain Enter without steering or stopping the running turn', async () => {
    const onSteer = vi.fn(async () => true)
    const onCancel = vi.fn()
    const editor = openComposer({ onSteer, onCancel })

    await pressEnter(editor, 'follow up later')

    expect(getQueuedPrompts(SESSION).map(entry => entry.text)).toEqual(['follow up later'])
    expect(onSteer).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('uses Ctrl+Enter to steer the running turn instead of queueing', async () => {
    const onSteer = vi.fn(async () => true)
    const editor = openComposer({ onSteer })

    await pressEnter(editor, 'correct this now', { ctrlKey: true })

    expect(onSteer).toHaveBeenCalledExactlyOnceWith('correct this now', 'steer')
    expect(getQueuedPrompts(SESSION)).toEqual([])
  })

  it('queues a text-and-image draft when the plugin steer chord cannot carry attachments', async () => {
    const onSteer = vi.fn(async () => true)
    const onSubmit = vi.fn(async () => true)
    const editor = openComposer({ onSteer, onSubmit })
    const image = { id: 'image-fixture', kind: 'image' as const, label: 'shot.png', path: 'fixture-shot.png' }
    await act(async () => mainComposerScope.$attachments.set([image]))

    await pressEnter(editor, 'describe this image', { ctrlKey: true })

    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION)).toEqual([
      expect.objectContaining({ text: 'describe this image', attachments: [image] })
    ])
    expect(mainComposerScope.$attachments.get()).toEqual([])
  })

  it('shows the Ctrl+Enter steer hint while steerable with a payload', async () => {
    const onSteer = vi.fn(async () => true)
    const view = render(<Harness onSteer={onSteer} />)
    const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
    editor.tabIndex = 0

    await act(async () => {
      editor.focus()
      editor.textContent = 'draft text'
      fireEvent.input(editor)
      // Native input mirrors into composer state on the next animation frame.
      await new Promise<void>(resolve => window.requestAnimationFrame(() => resolve()))
    })

    expect(view.container.querySelector('[data-slot="composer-steer-hint"]')).toBeTruthy()
  })

  it('hides the steer hint when idle', async () => {
    const view = render(<Harness busy={false} />)
    const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
    editor.tabIndex = 0

    await act(async () => {
      editor.focus()
      editor.textContent = 'draft text'
      fireEvent.input(editor)
      // Native input mirrors into composer state on the next animation frame.
      await new Promise<void>(resolve => window.requestAnimationFrame(() => resolve()))
    })

    expect(view.container.querySelector('[data-slot="composer-steer-hint"]')).toBeNull()
  })

  it('still sends plain Enter immediately while idle', async () => {
    const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
    const editor = openComposer({ busy: false, onSubmit })

    await pressEnter(editor, 'send while idle')

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('send while idle')
    expect(getQueuedPrompts(SESSION)).toEqual([])
  })

  it('does not deliver the queue or stop on a second empty Enter', async () => {
    const onSteer = vi.fn(async () => true)
    const onCancel = vi.fn()
    const editor = openComposer({ onSteer, onCancel })

    await pressEnter(editor, 'wait for the current turn')
    await pressEnter(editor, '')

    expect(getQueuedPrompts(SESSION).map(entry => entry.text)).toEqual(['wait for the current turn'])
    expect(onSteer).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it.each([
    { busy: false, shiftKey: true },
    { busy: true, shiftKey: true },
    { busy: true, isComposing: true },
    { busy: true, keyCode: 229 },
    { busy: true, disabled: true },
    { busy: true, disabled: true, ctrlKey: true },
    { busy: false, ctrlKey: true }
  ])('does not send or queue for protected input %j', async ({ busy, disabled, ...modifiers }) => {
    const onSteer = vi.fn(async () => true)
    const onSubmit = vi.fn(async () => true)
    const editor = openComposer({ busy, disabled, onSteer, onSubmit })

    await pressEnter(editor, 'keep this draft', modifiers)

    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION)).toEqual([])
    expect(editor.textContent).toBe('keep this draft')
  })

  it('keeps slash commands on their normal immediate path', async () => {
    const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
    const editor = openComposer({ onSubmit })

    await pressEnter(editor, '/status')

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('/status')
    expect(getQueuedPrompts(SESSION)).toEqual([])
  })
})

it('focus alone activates an inactive composer suggestion', async () => {
  removeBehavior()
  markActiveComposer('tile:other')
  removeBehavior = registry.register({
    id: 'focus-fixture',
    source: 'plugin:neo-next-prompt',
    area: 'composer.behavior.v2',
    data: {
      protocol: 2,
      create: (env: { publish: (text: string | null) => void }) => ({
        update: (state: { enabled: boolean; empty: boolean }) =>
          env.publish(state.enabled && state.empty ? 'focus candidate' : null),
        refresh: () => {},
        decide: () => null,
        dispose: () => {}
      })
    }
  })
  const gateway = { onEvent: () => () => {} } as never
  const view = render(<Harness busy={false} gateway={gateway} />)
  const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  await act(async () => {
    markActiveComposer('tile:other')
    view.rerender(<Harness busy={false} gateway={gateway} state={{ ...state }} />)
  })
  expect(editor.getAttribute('data-next-prompt')).toBeNull()
  await act(async () => {
    fireEvent.focus(editor)
  })
  expect(editor.getAttribute('data-next-prompt')).toBe('ready')
  markActiveComposer('main')
})
