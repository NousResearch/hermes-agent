import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import { completeDesktopBoot, failDesktopBoot } from '@/store/boot'

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

afterEach(() => {
  cleanup()
  completeDesktopBoot()
})

describe('transcript while a Webapp tab is signed out', () => {
  // After sign-out the sign-in view covers the tab. The conversation must not stay on the page
  // underneath it, where removing the overlay would show it; it comes back once the tab is signed
  // in again.
  it('leaves the page while the sign-in view is up and returns after sign-in', async () => {
    render(<Harness />)
    expect(await screen.findByText('edit me please')).toBeTruthy()
    expect(screen.getByText('done')).toBeTruthy()

    act(() => failDesktopBoot(WEBAPP_LAUNCH_REQUIRED))

    expect(screen.queryByText('edit me please')).toBeNull()
    expect(screen.queryByText('done')).toBeNull()

    act(() => completeDesktopBoot())

    expect(await screen.findByText('edit me please')).toBeTruthy()
    expect(screen.getByText('done')).toBeTruthy()
  })

  it('keeps the transcript under any other startup failure', async () => {
    render(<Harness />)
    expect(await screen.findByText('edit me please')).toBeTruthy()

    act(() => failDesktopBoot('Gateway exited during startup'))

    expect(screen.getByText('edit me please')).toBeTruthy()
  })
})
