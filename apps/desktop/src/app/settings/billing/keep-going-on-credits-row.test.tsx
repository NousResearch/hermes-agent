import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createElement } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const apiMocks = vi.hoisted(() => ({
  stepUp: vi.fn(),
  updateKeepGoingOnCredits: vi.fn()
}))

const gatewayMock = vi.hoisted(() => {
  const handlers = new Map<string, Set<(event: unknown) => void>>()

  return {
    emit: (eventName: string, event: unknown) => handlers.get(eventName)?.forEach(handler => handler(event)),
    gateway: {
      on: (eventName: string, handler: (event: unknown) => void) => {
        const set = handlers.get(eventName) ?? new Set()
        handlers.set(eventName, set)
        set.add(handler)

        return () => set.delete(handler)
      }
    },
    reset: () => handlers.clear()
  }
})

vi.mock('@/store/gateway', async () => {
  const { atom } = (await vi.importActual('nanostores')) as { atom: (value: unknown) => unknown }

  return { $gateway: atom(gatewayMock.gateway) }
})

vi.mock('./api', () => ({
  useBillingApi: () => apiMocks
}))

import { KeepGoingOnCreditsRow } from './keep-going-on-credits-row'
import { $stepUpInFlight } from './use-step-up'

const NEEDS_APPROVAL = { ok: false, refusal: { kind: 'insufficient_scope', message: '' } }

function renderRow(enabled = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(createElement(QueryClientProvider, { client }, createElement(KeepGoingOnCreditsRow, { enabled })))
}

beforeEach(() => {
  apiMocks.stepUp.mockReset()
  apiMocks.updateKeepGoingOnCredits.mockReset()
  gatewayMock.reset()
  $stepUpInFlight.set(false)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { openExternal: vi.fn() } })
})

describe('KeepGoingOnCreditsRow', () => {
  it('without Remote Spending: one click opens the approval, holds the switch, then applies the change', async () => {
    let resolveStepUp: (value: unknown) => void = () => {}
    apiMocks.updateKeepGoingOnCredits
      .mockResolvedValueOnce(NEEDS_APPROVAL)
      .mockResolvedValueOnce({ data: {}, ok: true })
    apiMocks.stepUp.mockReturnValue(new Promise(resolve => (resolveStepUp = resolve)))

    renderRow(false)
    const toggle = screen.getByRole('switch')
    fireEvent.click(toggle)

    // The approval is in flight: one line under the switch, the switch stays on, billing refetches pause.
    expect(await screen.findByText(/Approve Remote Spending in your browser/)).toBeTruthy()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true')
    expect($stepUpInFlight.get()).toBe(true)

    act(() => {
      gatewayMock.emit('billing.step_up.verification', {
        payload: { user_code: 'ABCD-1234', verification_url: 'https://portal.nousresearch.com/device' }
      })
    })

    // The browser opens on its own (no second "Verify" click).
    expect(window.hermesDesktop?.openExternal).toHaveBeenCalledWith('https://portal.nousresearch.com/device')
    expect(screen.getByText('ABCD-1234')).toBeTruthy()

    await act(async () => resolveStepUp({ data: { granted: true, ok: true }, ok: true }))

    // Approved: the SAME change is re-applied automatically, and the prompt goes away.
    await waitFor(() => expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenCalledTimes(2))
    expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenLastCalledWith(true)
    await waitFor(() => expect(screen.queryByText(/Approve Remote Spending/)).toBeNull())
    expect($stepUpInFlight.get()).toBe(false)
  })

  it('a declined approval puts the switch back and says why, once', async () => {
    apiMocks.updateKeepGoingOnCredits.mockResolvedValue(NEEDS_APPROVAL)
    apiMocks.stepUp.mockResolvedValue({ data: { granted: false, ok: true }, ok: true })

    renderRow(false)
    fireEvent.click(screen.getByRole('switch'))

    expect(await screen.findByText(/Verification was not approved/)).toBeTruthy()
    await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false'))
    expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/Remote Spending needs approval/)).toBeNull()
  })

  it('with Remote Spending already allowed: a plain toggle, no approval', async () => {
    apiMocks.updateKeepGoingOnCredits.mockResolvedValue({ data: {}, ok: true })

    renderRow(true)
    fireEvent.click(screen.getByRole('switch'))

    await waitFor(() => expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenCalledWith(false))
    expect(apiMocks.stepUp).not.toHaveBeenCalled()
    expect(screen.queryByText(/Approve Remote Spending/)).toBeNull()
  })
})

describe('KeepGoingOnCreditsRow after an approval', () => {
  it('a refusal right after approving offers Try again, never the generic approval/top-up copy', async () => {
    apiMocks.updateKeepGoingOnCredits
      .mockResolvedValueOnce(NEEDS_APPROVAL)
      .mockResolvedValueOnce(NEEDS_APPROVAL)
      .mockResolvedValueOnce({ data: {}, ok: true })
    apiMocks.stepUp.mockResolvedValue({ data: { granted: true, ok: true }, ok: true })

    renderRow(true)
    fireEvent.click(screen.getByRole('switch'))

    expect(await screen.findByText(/approved, but the change didn't save/)).toBeTruthy()
    expect(screen.queryByText(/Start a top-up/)).toBeNull()
    expect(screen.queryByText('Verify to continue')).toBeNull()

    fireEvent.click(screen.getByText('Try again'))
    await waitFor(() => expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenCalledTimes(3))
    expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenLastCalledWith(false)
    expect(apiMocks.stepUp).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(screen.queryByText(/didn't save/)).toBeNull())
  })
})

describe('KeepGoingOnCreditsRow with overlapping attempts', () => {
  it('a cancelled approval that resolves late never snaps back the newer attempt', async () => {
    const resolvers: ((value: unknown) => void)[] = []
    apiMocks.updateKeepGoingOnCredits.mockResolvedValue(NEEDS_APPROVAL)
    apiMocks.stepUp.mockImplementation(() => new Promise(resolve => resolvers.push(resolve)))

    renderRow(false)
    fireEvent.click(screen.getByRole('switch'))
    expect(await screen.findByText(/Approve Remote Spending in your browser/)).toBeTruthy()

    fireEvent.click(screen.getByText('Cancel'))
    await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false'))

    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(apiMocks.updateKeepGoingOnCredits).toHaveBeenCalledTimes(2))
    expect(await screen.findByText(/Approve Remote Spending in your browser/)).toBeTruthy()

    // The first (cancelled) approval comes back now: the second attempt keeps its switch and its prompt.
    await act(async () => resolvers[0]?.({ data: { granted: false, ok: true }, ok: true }))
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true')
    expect(screen.getByText(/Approve Remote Spending in your browser/)).toBeTruthy()
  })
})
