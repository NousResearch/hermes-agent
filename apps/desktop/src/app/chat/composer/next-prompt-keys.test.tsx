import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearSessionDraft, mainComposerScope } from '@/store/composer'
import { clearQueuedPrompts } from '@/store/composer-queue'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

const suggestion = vi.hoisted(() => ({
  text: '运行解析器回归测试',
  decide: vi.fn(() => '运行解析器回归测试'),
  refresh: vi.fn()
}))

vi.mock('./hooks/use-composer-behavior', () => ({ useComposerBehavior: () => suggestion }))
import { ChatBar } from './index'
const SESSION = 'next-prompt-key-test'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ onSubmit }: { onSubmit: ChatBarProps['onSubmit'] }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (m: ThreadMessageLike) => m,
    isRunning: false,
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
            onSubmit={onSubmit}
            sessionId={SESSION}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

const open = () => {
  const submit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const view = render(<Harness onSubmit={submit} />)
  const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  editor.tabIndex = 0
  editor.focus()

  return { editor, submit }
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft(SESSION)
  clearQueuedPrompts(SESSION)
  vi.clearAllMocks()
})

describe('native next-prompt key ownership', () => {
  it('keeps ghost text outside draft; Tab accepts, then Enter sends', async () => {
    const { editor, submit } = open()
    expect(editor.textContent).toBe('')
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter' })
    })
    expect(submit).not.toHaveBeenCalled()
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Tab' })
    })
    expect(editor.textContent).toBe(suggestion.text)
    expect(submit).not.toHaveBeenCalled()
    expect(suggestion.decide).toHaveBeenCalledWith('consumed')
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Enter' })
    })
    expect(submit).toHaveBeenCalledWith(suggestion.text, expect.anything())
  })
  it('does not overwrite a live keystroke before React has synchronized', async () => {
    const { editor } = open()
    editor.textContent = '我自己的输入'
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Tab' })
    })
    expect(editor.textContent).toBe('我自己的输入')
    expect(suggestion.decide).not.toHaveBeenCalled()
  })
  it('yields to composition and Shift+Tab; Esc dismisses without stopping', async () => {
    const { editor, submit } = open()
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Tab', isComposing: true })
      fireEvent.keyDown(editor, { key: 'Tab', shiftKey: true })
    })
    expect(suggestion.decide).not.toHaveBeenCalled()
    await act(async () => {
      const escape = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })
      fireEvent(editor, escape)
      expect(escape.defaultPrevented).toBe(true)
    })
    expect(suggestion.decide).toHaveBeenCalledWith('dismissed')
    expect(submit).not.toHaveBeenCalled()
  })
})
