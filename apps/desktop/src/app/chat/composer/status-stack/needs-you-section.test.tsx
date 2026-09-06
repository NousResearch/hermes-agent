import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $clarifyDockedSessions, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-needs-you'

function renderStack(sessionId: null | string = SID) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={sessionId} />
      </I18nProvider>
    </MemoryRouter>
  )
}

function parkClarify(sessionId = SID) {
  const request = vi.fn().mockResolvedValue({ ok: true })

  $activeSessionId.set(sessionId)
  $gateway.set({ request } as never)
  setClarifyRequest({
    choices: ['Go ahead with the redesign', 'Pause here for review'],
    multiSelect: false,
    question: 'Should I continue into the redesign, or pause for review?',
    requestId: 'req-1',
    sessionId
  })

  return request
}

describe('ComposerStatusStack Needs-you section', () => {
  afterEach(() => {
    cleanup()
    clearClarifyRequest()
    $activeSessionId.set(null)
    $gateway.set(null)
    vi.clearAllMocks()
  })

  it('renders nothing when the session has no pending question', () => {
    const view = renderStack()

    expect(view.container.firstChild).toBeNull()
    expect($clarifyDockedSessions.get()[SID]).toBeUndefined()
  })

  it('docks the pending question above the composer with its choices and a Needs-you headline', () => {
    parkClarify()

    const view = renderStack()

    expect(screen.getByText('Needs you')).toBeTruthy()
    expect(screen.getByText('Should I continue into the redesign, or pause for review?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Go ahead with the redesign/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Pause here for review/ })).toBeTruthy()
    // The docked form is flat — the transcript widget shell is not reused.
    expect(view.container.querySelector('[data-slot="composer-needs-you"] [data-slot="clarify-docked"]')).toBeTruthy()
    expect(view.container.querySelector('[data-slot="composer-needs-you"] [data-slot="clarify-inline"]')).toBeNull()
  })

  it('registers itself as the session dock while mounted and releases it on unmount', () => {
    parkClarify()

    const view = renderStack()

    expect($clarifyDockedSessions.get()[SID]).toBe(1)

    view.unmount()

    expect($clarifyDockedSessions.get()[SID]).toBeUndefined()
  })

  it('does not dock another session’s question', () => {
    parkClarify('sess-other')

    const view = renderStack()

    expect(view.container.firstChild).toBeNull()
  })

  it('stays fully opaque while the thread is scrolled up (a ghosted question is a missed question)', () => {
    parkClarify()
    $threadScrolledUp.set(true)

    const view = renderStack()
    const card = view.container.querySelector('[data-slot="composer-status-stack"] > div') as HTMLElement

    expect(card.classList.contains('opacity-30')).toBe(false)
    expect(card.classList.contains('opacity-100')).toBe(true)

    $threadScrolledUp.set(false)
  })

  it('answers the same request the inline card would and clears the panel', async () => {
    const request = parkClarify()

    renderStack()

    fireEvent.click(screen.getByRole('button', { name: /Pause here for review/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await vi.waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', { answer: 'Pause here for review', request_id: 'req-1' })
    })

    await vi.waitFor(() => {
      expect(screen.queryByText('Needs you')).toBeNull()
    })
  })
})
