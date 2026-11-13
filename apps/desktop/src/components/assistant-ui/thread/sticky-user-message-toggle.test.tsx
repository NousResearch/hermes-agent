import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { setStickyUserMessage } from '@/store/sticky-user-message'

import {
  assistantMessage,
  stubThreadEnvironment,
  stubThreadViewportSize,
  ThreadRuntime,
  userMessage
} from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const PROMPT = 'keep me pinned'

const userRoot = (container: HTMLElement) => {
  const root = container.querySelector<HTMLElement>('[data-slot="aui_user-message-root"]')

  expect(root).toBeTruthy()

  return root!
}

function renderThread() {
  return render(
    <ThreadRuntime messages={[userMessage('user-1', PROMPT), assistantMessage()]}>
      <Thread />
    </ThreadRuntime>
  )
}

afterEach(() => {
  cleanup()
  act(() => setStickyUserMessage(true))
})

describe('sticky user message toggle', () => {
  // The shipped behavior: every user bubble carries position: sticky so the
  // prompt that started the turn stays on screen while its reply streams.
  it('pins the user bubble by default', async () => {
    const { container } = renderThread()
    await screen.findByText(PROMPT)

    expect(userRoot(container).className).toContain('sticky')
    expect(userRoot(container).className).toContain('z-40')
  })

  // Opt-out (#38372/#80146): the gate is the container itself, so the bubble
  // scrolls like any other message and the switch is the one lever every
  // pinnable surface reads. The clamp (#39721) is independent and stays.
  it('drops the pin while the switch is off and restores it on', async () => {
    act(() => setStickyUserMessage(false))
    const { container } = renderThread()
    await screen.findByText(PROMPT)

    expect(userRoot(container).className).not.toContain('sticky')
    expect(userRoot(container).className).not.toContain('z-40')

    act(() => setStickyUserMessage(true))

    expect(userRoot(container).className).toContain('sticky')
  })

  it('unpins live without a remount', async () => {
    const { container } = renderThread()
    await screen.findByText(PROMPT)

    const pinned = userRoot(container)

    expect(pinned.className).toContain('sticky')

    act(() => setStickyUserMessage(false))

    expect(userRoot(container)).toBe(pinned)
    expect(pinned.className).not.toContain('sticky')

    act(() => setStickyUserMessage(true))

    expect(pinned.className).toContain('sticky')
  })
})
