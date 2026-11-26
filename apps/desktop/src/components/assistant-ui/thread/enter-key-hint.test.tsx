// A coarse-pointer keyboard takes its action key from the field: Chromium's
// default there reads as "send", which fights this composer's contract (Enter
// opens a line; only the button sends). `enterKeyHint="enter"` relabels that key
// as Return on touch hosts and must stay absent everywhere else, so a fine
// pointer keeps the browser's own choice.
import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ChatBar } from '@/app/chat/composer'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import { $touchPointer } from '@/lib/touch-interaction'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

stubThreadEnvironment()
stubThreadViewportSize()

const chatBarState: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function ComposerHarness() {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
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
            onSubmit={vi.fn(async () => true)}
            state={chatBarState}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

function editor(container: HTMLElement) {
  const node = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)

  expect(node).not.toBeNull()

  return node!
}

afterEach(() => {
  cleanup()
  $touchPointer.set(false)
})

describe('the composer Enter key on a phone keyboard', () => {
  it('is labelled Return, not Send, on a coarse pointer', () => {
    $touchPointer.set(true)

    expect(editor(render(<ComposerHarness />).container).getAttribute('enterkeyhint')).toBe('enter')
  })

  it('keeps the browser default on a fine pointer', () => {
    $touchPointer.set(false)

    expect(editor(render(<ComposerHarness />).container).hasAttribute('enterkeyhint')).toBe(false)
  })
})
