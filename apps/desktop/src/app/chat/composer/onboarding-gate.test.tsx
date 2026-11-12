// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $onboardingPhase } from '@/store/onboarding-phase'
import { $onboardingSession } from '@/store/onboarding-session'
import { $portalShell, $shellChoice } from '@/store/shell-variant'
import type * as Windows from '@/store/windows'

import type { ChatBarState } from './types'

import { ChatBar } from './index'

// The HUD keeps the classic look in every mode; that must not hand it the classic first run's
// composer rules while the redesign (and so the setup guide) is the user's preference.
vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<typeof Windows>()),
  isHudWindow: () => true
}))

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
            sessionId="rt-1"
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

const skip = () => screen.queryByRole('button', { name: 'Skip setup' })

beforeEach(() => {
  // The composer shows the classic setup chat: a shown run whose chat is this session.
  $onboardingPhase.set('shown')
  $onboardingSession.set({ connectionId: null, runId: 'run-1', runtimeId: 'rt-1', storedId: 'st-1' })
})

afterEach(() => {
  cleanup()
  $onboardingSession.set(null)
  $onboardingPhase.set('idle')
  $shellChoice.set(null)
})

describe('composer onboarding rules in the HUD', () => {
  it('follow the redesign preference, not the classic look', () => {
    $shellChoice.set(null)
    render(<Harness />)

    expect($portalShell.get()).toBe(false)
    expect(skip()).toBeNull()

    act(() => $shellChoice.set(false))

    expect(skip()).toBeTruthy()
  })
})
