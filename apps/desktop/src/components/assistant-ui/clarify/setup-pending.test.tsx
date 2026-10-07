import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { accentsFor, NOUS_ACCENT } from '@/components/onboarding-chat/options'
import { I18nProvider } from '@/i18n'
import { answerSetupCard, type ClarifyRequest, clearClarifyRequest, setClarifyRequest, type SetupChooseSpec } from '@/store/clarify'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $accentOverride } from '@/themes/accent-override'

import { SetupChoosePending } from './setup-pending'

const SESSION = 'setup-session'

const view: SessionView = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom(SESSION), $storedId: atom('stored-setup') }

function renderCard(requestId: string, question: string, kind: SetupChooseSpec['kind']) {
  const request: ClarifyRequest = {
    questions: [{ choices: null, multiSelect: false, qid: 'q0', question }],
    requestId,
    sessionId: SESSION,
    setup: { kind, multiSelect: false, options: null, preselected: [] }
  }

  rememberServerRequest({ fail: vi.fn(), id: requestId, method: 'setup_choose', params: {}, respond: vi.fn() })
  setClarifyRequest(request)

  return render(
    <QueryClientProvider client={new QueryClient()}>
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={view}>
          <SetupChoosePending fromArgs={null} onAnswered={vi.fn()} request={request} undelivered={false} />
        </SessionViewProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

let consoleError: MockInstance<typeof console.error>

beforeEach(() => {
  resetServerRequestsForTests()
  consoleError = vi.spyOn(console, 'error')
})

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $accentOverride.set(null)
  consoleError.mockRestore()
})

const loopErrors = () => consoleError.mock.calls.filter(([first]) => /Maximum update depth/.test(String(first)))

describe('setup cards render without redrawing themselves', () => {
  it('shows the name card and waits for an answer', async () => {
    renderCard('name-1', 'What should I call you?', 'question')

    expect(await screen.findByText('What should I call you?')).toBeTruthy()
    expect(loopErrors()).toEqual([])
  })

  it('applies a look typed in the composer the same way a click does', async () => {
    renderCard('accent-1', 'Pick an accent', 'accent')
    await screen.findByText('Pick an accent')

    const swatch = accentsFor(false).find(({ hex }) => hex !== NOUS_ACCENT)

    expect(swatch).toBeTruthy()
    expect(answerSetupCard(SESSION, swatch!.name)).toBe(true)
    expect($accentOverride.get()).toBe(swatch!.hex)
    expect(loopErrors()).toEqual([])
  })
})
