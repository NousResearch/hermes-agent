import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from '@/app/chat/composer/floating-target'
import { $reactionsEnabled } from '@/store/reactions-enabled'

import { assistantMessage, stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()

/** The transcript inside a chat surface that owns a floating composer — the
 * shape of every chat pane, and the one the window-level focus-follow acts on. */
function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div data-chat-surface="" data-composer-surface-id="surface-1">
        <Thread />
        <div data-composer-owner="surface-1">
          <div contentEditable data-slot="composer-rich-input" suppressContentEditableWarning tabIndex={-1} />
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}

let unregister: (() => void) | undefined

beforeEach(() => {
  $reactionsEnabled.set(true)
  unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })
})

afterEach(() => {
  unregister?.()
  unregister = undefined
  cleanup()
  $reactionsEnabled.set(false)
})

describe('assistant reaction picker', () => {
  it('stays open while the pointer moves over the message toward it', async () => {
    render(<Harness />)

    const message = (await screen.findByText('done')).closest<HTMLElement>('[data-slot="aui_assistant-message-root"]')
    const trigger = message?.querySelector<HTMLButtonElement>('[data-slot="aui_msg-reactions"]')

    expect(trigger).toBeTruthy()
    fireEvent.click(trigger!)
    expect(await screen.findByRole('button', { name: '👍' })).toBeTruthy()

    fireEvent.pointerMove(message!, { buttons: 0, clientX: 40, clientY: 40 })

    expect(screen.queryByRole('button', { name: '👍' })).not.toBeNull()
    expect(trigger?.getAttribute('data-state')).toBe('open')
  })
})
