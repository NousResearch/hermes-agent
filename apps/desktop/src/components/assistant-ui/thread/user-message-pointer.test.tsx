import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $touchPointer } from '@/lib/touch-interaction'
import { $reactionsEnabled } from '@/store/reactions-enabled'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

/** A contextmenu event as Chromium sends it: a PointerEvent that says which pointer raised it. */
function contextMenu(target: Element, pointerType: 'mouse' | 'pen' | 'touch'): MouseEvent {
  const event = new MouseEvent('contextmenu', { bubbles: true, button: 2, cancelable: true })
  Object.defineProperty(event, 'pointerType', { value: pointerType })
  fireEvent(target, event)

  return event
}

beforeEach(() => {
  $reactionsEnabled.set(true)
})

afterEach(() => {
  cleanup()
  $reactionsEnabled.set(false)
  $touchPointer.set(false)
  delete window.document.documentElement.dataset.hermesDesktopHost
})

describe('user message reactions by pointer', () => {
  // A browser tab on a touch surface leaves a long press to the platform (selection handles and
  // Copy), but a mouse plugged into that device right-clicks the way it does anywhere else: the
  // reaction picker opens.
  it('opens the reaction picker on a mouse right-click in a touch-layout browser tab', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    $touchPointer.set(true)
    render(<Harness />)

    const event = contextMenu(await screen.findByText('edit me please'), 'mouse')

    expect(event.defaultPrevented).toBe(true)
    expect(await screen.findByRole('button', { name: '👍' })).toBeTruthy()
  })

  it('leaves a touch long-press in a touch-layout browser tab to the platform', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    $touchPointer.set(true)
    render(<Harness />)

    const event = contextMenu(await screen.findByText('edit me please'), 'touch')

    expect(event.defaultPrevented).toBe(false)
    expect(screen.queryByRole('button', { name: '👍' })).toBeNull()
  })

  // Native Desktop has no touch handling: a long press on a touchscreen raises the picker as a
  // right-click does, as it always has.
  it.each(['mouse', 'touch'] as const)(
    'opens the picker for a %s context menu in native Desktop',
    async pointerType => {
      render(<Harness />)

      const event = contextMenu(await screen.findByText('edit me please'), pointerType)

      expect(event.defaultPrevented).toBe(true)
      expect(await screen.findByRole('button', { name: '👍' })).toBeTruthy()
    }
  )
})
