import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearQueuedPrompts, getQueuedPrompts } from '@/store/composer-queue'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import { MAIN_COMPOSER_SCOPE } from './scope'
import type { ChatBarProps, ChatBarState } from './types'
import { ComposerVisibleContext } from './visibility'

import { ChatBar } from './index'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ busy = false, onSubmit, onSteer }: Pick<ChatBarProps, 'onSubmit' | 'onSteer'> & { busy?: boolean }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: busy,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ChatBar
            busy={busy}
            disabled={false}
            gateway={null}
            onCancel={() => {}}
            onSteer={onSteer}
            onSubmit={onSubmit}
            queueSessionKey="quote-flow-stored"
            sessionId="quote-flow-session"
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

async function addQuote(text = '波动率也不能预测。', id = 'quote-a') {
  await act(async () => {
    window.dispatchEvent(
      new CustomEvent('hermes:composer-quote-add', {
        detail: { target: 'main', quote: { id, role: 'assistant', text, timestamp: '12:45' } }
      })
    )
  })
}

function editorOf(container: HTMLElement) {
  return container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
}

function submitEditor(container: HTMLElement) {
  const editor = editorOf(container)
  fireEvent.submit(editor.closest('form')!)
}

afterEach(() => {
  cleanup()
  clearQueuedPrompts('quote-flow-stored')
  MAIN_COMPOSER_SCOPE.attachments.clear()

  // On the base version the entire quote scope is absent.
  if ('quotes' in MAIN_COMPOSER_SCOPE) {
    ;(MAIN_COMPOSER_SCOPE.quotes as { forSession(key: string): { clear(): void } })
      .forSession('quote-flow-stored')
      .clear()
  }
})

describe('quoting a selected message without copying', () => {
  it('only lets the visible composer claim a quote when a docked composer remains mounted', async () => {
    const { container } = render(
      <>
        <ComposerVisibleContext.Provider value={false}>
          <Harness onSubmit={async () => true} />
        </ComposerVisibleContext.Provider>
        <ComposerVisibleContext.Provider value={true}>
          <Harness onSubmit={async () => true} />
        </ComposerVisibleContext.Provider>
      </>
    )

    await addQuote()

    expect(MAIN_COMPOSER_SCOPE.quotes.forSession('quote-flow-stored').$quotes.get()).toHaveLength(1)
    expect(container.querySelector('[data-slot="composer-quote-card"]')).toBeTruthy()
  })
  it('parks the exact selection as a removable card and sends it without inserting it into the editor', async () => {
    const submitted: string[] = []

    const { container } = render(
      <Harness
        onSubmit={async text => {
          submitted.push(text)

          return true
        }}
      />
    )

    const text = '前者讨论“方向有没有可预测的平均偏差”，\n后者讨论“明天的风险有多大”。'

    await addQuote(text)

    expect(container.querySelector('[data-slot="composer-quote-card"]')?.getAttribute('title')).toBe(text)
    expect(composerPlainText(editorOf(container))).toBe('')
    await act(async () => submitEditor(container))

    await waitFor(() => expect(submitted).toEqual([`> [assistant 12:45]\n> ${text.replaceAll('\n', '\n> ')}\n\n`]))
    expect(container.querySelector('[data-slot="composer-quote-card"]')).toBeNull()
  })

  it('keeps a rejected quote send available to retry', async () => {
    const { container } = render(<Harness onSubmit={async () => false} />)
    await addQuote()
    await act(async () => submitEditor(container))

    expect(container.querySelector('[data-slot="composer-quote-card"]')?.textContent).toContain('波动率也不能预测。')
    expect(composerPlainText(editorOf(container))).toBe('')
  })

  it('queues quote-only payloads while the assistant is busy instead of treating them as Stop', async () => {
    const onSteer = vi.fn(async () => true)
    const { container } = render(<Harness busy onSteer={onSteer} onSubmit={async () => true} />)
    await addQuote()
    await act(async () => submitEditor(container))

    expect(getQueuedPrompts('quote-flow-stored').map(entry => entry.text)).toEqual([
      '> [assistant 12:45]\n> 波动率也不能预测。\n\n'
    ])
    expect(onSteer).not.toHaveBeenCalled()
    expect(container.querySelector('[data-slot="composer-quote-card"]')).toBeNull()
  })

  it('removes a parked quote with its close button', async () => {
    const { container } = render(<Harness onSubmit={async () => true} />)
    await addQuote()
    const button = container.querySelector<HTMLElement>('[data-slot="composer-quote-remove"]')

    expect(button).toBeTruthy()
    fireEvent.click(button!)
    expect(container.querySelector('[data-slot="composer-quote-card"]')).toBeNull()
  })
})
