import { act } from '@testing-library/react'
import { type RefObject, useRef } from 'react'
import { flushSync } from 'react-dom'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerAttachments, clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'

import type { QueueEditState } from '../composer-utils'

import { useComposerDraft } from './use-composer-draft'

const composer = vi.hoisted(() => {
  let text = ''
  const listeners = new Set<() => void>()

  return {
    getText: () => text,
    reset: () => {
      text = ''
      listeners.clear()
    },
    runtime: {
      getState: () => ({ text }),
      subscribe: (listener: () => void) => {
        listeners.add(listener)

        return () => listeners.delete(listener)
      }
    },
    setText: (next: string) => {
      text = next
      listeners.forEach(listener => listener())
    }
  }
})

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => ({ setText: composer.setText }) }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: { text: composer.getText() } }),
  useComposerRuntime: () => composer.runtime
}))

function Harness({ scope }: { scope: string | null }) {
  const queueEditRef = useRef<QueueEditState | null>(null)
  const draft = useComposerDraft({
    activeQueueSessionKey: scope,
    focusKey: scope,
    inputDisabled: false,
    queueEditRef: queueEditRef as RefObject<QueueEditState | null>,
    sessionId: scope
  })

  return <div data-testid="editor" ref={draft.editorRef} />
}

describe('useComposerDraft session swap', () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    composer.reset()
    $composerAttachments.set([])
    clearSessionDraft('session-a')
    clearSessionDraft(null)

    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
  })

  afterEach(() => {
    act(() => root.unmount())
    container.remove()

    composer.reset()
    $composerAttachments.set([])
    clearSessionDraft('session-a')
    clearSessionDraft(null)
    vi.restoreAllMocks()
  })

  it('loads the new-session draft before paint when leaving an existing session', () => {
    stashSessionDraft('session-a', 'previous draft', [])

    act(() => root.render(<Harness scope="session-a" />))

    const editor = container.querySelector('[data-testid="editor"]')
    expect(editor?.textContent).toBe('previous draft')

    flushSync(() => root.render(<Harness scope={null} />))

    expect(editor?.textContent).toBe('')
    expect(takeSessionDraft('session-a').text.trim()).toBe('previous draft')
    expect(takeSessionDraft(null).text).toBe('')
  })
})
