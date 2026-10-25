// @vitest-environment jsdom
// The plugin seam's headline guarantee is a NEGATIVE one: with no plugin asking,
// the composer input keeps `spellcheck="false"` (#44415). That has to be
// asserted on the RENDERED composer, not on a bare div, or flipping the app prop
// (`spellCheck={false}` -> `{true}`) goes unnoticed. The second test pins the
// seam to the real surface: the hook must actually be mounted on ChatBar, or the
// SDK verb can never reach the input a user types in.
import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { composerHost } from '@/sdk/composer'
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
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** Render the main composer and return the editable the user types in. */
function renderComposer(): HTMLElement {
  const { container } = render(<Harness />)

  return container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
})

describe('composer spell check', () => {
  it('renders the input with spell check off by default', () => {
    const editor = renderComposer()

    expect(editor.getAttribute('spellcheck')).toBe('false')
  })

  it('lets the plugin seam turn spell check on for the rendered input', async () => {
    const editor = renderComposer()

    // `null` = the active composer, which is this ChatBar's 'main' surface.
    await act(async () => {
      await expect(composerHost.setSpellcheck(null, true)).resolves.toBe(true)
    })

    expect(editor.getAttribute('spellcheck')).toBe('true')

    // ...and back off again, so the seam is a toggle rather than a one-way door.
    await act(async () => {
      await expect(composerHost.setSpellcheck(null, false)).resolves.toBe(true)
    })

    expect(editor.getAttribute('spellcheck')).toBe('false')
  })
})
