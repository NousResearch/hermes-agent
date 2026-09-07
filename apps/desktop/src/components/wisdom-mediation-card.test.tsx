// @vitest-environment jsdom
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { WisdomMediationCard } from './wisdom-mediation-card'

const { read, resolve } = vi.hoisted(() => ({ read: vi.fn(), resolve: vi.fn() }))
vi.mock('@/hermes', () => ({ getWisdomMediation: read, resolveWisdomConsent: resolve }))

const interaction = {
  id: 'consent', assessment_id: 'event', state: 'pending', operation: 'update',
  expires_at: Date.now() / 1000 + 86400, actions: ['defer', 'inspect', 'confirm'],
  facts: { slug: 'Team Runbook', version: 2, compatibility: { outcome: 'compatible' } }
}
const activity = {
  mode: 'agent', assessments: [{ id: 'event', owner_session: 'session', state: 'delivered',
    advice: { title: 'An updated runbook', explanation: 'It may help your current task.', relevance: 'recommend' } }],
  interactions: [interaction]
}

describe('WisdomMediationCard', () => {
  beforeEach(() => { vi.clearAllMocks(); read.mockResolvedValue(activity) })
  afterEach(() => vi.useRealTimers())

  it('shows advice and native controls without applying anything automatically', async () => {
    render(<WisdomMediationCard sessionId="session" />)
    await screen.findByText('An updated runbook')
    expect(screen.getAllByRole('button').map(button => button.textContent)).toEqual(['Not Now', 'Review first', 'Update'])
    expect(resolve).not.toHaveBeenCalled()
    resolve.mockResolvedValue({ ...interaction, state: 'completed', actions: ['inspect'] })
    fireEvent.click(screen.getByRole('button', { name: 'Update' }))
    await waitFor(() => expect(resolve).toHaveBeenCalledWith('consent', 'session', 'confirm', undefined))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Update' })).toBeNull())
  })

  it('keeps other surfaces passive', async () => {
    render(<WisdomMediationCard sessionId="other" passive />)
    await screen.findByText('An updated runbook')
    expect(screen.queryByRole('button', { name: 'Update' })).toBeNull()
  })

  it('retains advice through a polling failure', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    render(<WisdomMediationCard sessionId="session" />)
    await screen.findByText('An updated runbook')
    read.mockRejectedValue(new Error('offline'))
    await act(async () => { vi.advanceTimersByTime(10_000) })
    expect(screen.getByText('An updated runbook')).toBeTruthy()
  })

  it('honors durable surface-local defer but retains passive access', async () => {
    read.mockResolvedValue({ ...activity, interactions: [{ ...interaction, deferred_surfaces: ['local'] }] })
    const { rerender } = render(<WisdomMediationCard sessionId="session" />)
    await waitFor(() => expect(read).toHaveBeenCalled())
    expect(screen.queryByText('An updated runbook')).toBeNull()
    rerender(<WisdomMediationCard sessionId="session" passive />)
    await screen.findByText('An updated runbook')
  })
})
