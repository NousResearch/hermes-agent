// @vitest-environment jsdom
import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness() {
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
            onCancel={() => {}}
            onSubmit={async () => true}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  vi.restoreAllMocks()
})

describe('native composer text replacements', () => {
  it.each([
    ['MacIntel', 'on'],
    ['Win32', 'off'],
    ['Linux x86_64', 'off']
  ])('admits macOS substitutions without enabling spellcheck on %s', (platform, autocorrect) => {
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue(platform)
    const { container } = render(<Harness />)
    const editor = container.querySelector(`[data-slot="${RICH_INPUT_SLOT}"]`)!

    expect(editor.getAttribute('autocorrect')).toBe(autocorrect)
    expect(editor.getAttribute('spellcheck')).toBe('false')
    expect(editor.getAttribute('autocapitalize')).toBe('off')
    expect(container.querySelector('textarea[aria-hidden]')?.getAttribute('autocorrect')).toBe('off')
  })
})
