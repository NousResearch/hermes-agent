import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AgentExchangeCard, agentExchangePreview } from './agent-exchange-card'

afterEach(cleanup)

describe('agentExchangePreview', () => {
  it('uses the first sentence and normalizes transport whitespace', () => {
    expect(agentExchangePreview('Own the task.\nDo not publish.')).toBe('Own the task.')
  })

  it('truncates an oversized first sentence without changing the stored body', () => {
    expect(agentExchangePreview('abcdefghij', 6)).toBe('abcde…')
  })
})

describe('AgentExchangeCard', () => {
  it('renders a compact collapsed handoff with the exact body behind the same header', () => {
    const { container } = render(
      <AgentExchangeCard
        agent="Hermes"
        avatar={<span>H</span>}
        body={<p>Own the task. Do not publish.</p>}
        bodyText="Own the task. Do not publish."
        kind="handoff"
        slot="test-agent-handoff"
      />
    )

    expect(screen.getByText('Handoff')).toBeTruthy()
    expect(screen.getByText('Hermes')).toBeTruthy()
    expect(screen.getByText('Own the task.')).toBeTruthy()

    const details = container.querySelector('details')
    const summary = container.querySelector('summary')

    expect(details?.open).toBe(false)
    expect(summary?.textContent).not.toContain('show message')
    expect(container.querySelector('[data-slot="test-agent-handoff"]')?.className).toContain('w-full')

    fireEvent.click(summary as HTMLElement)

    expect(details?.open).toBe(true)
    expect(screen.getByText('Own the task. Do not publish.')).toBeTruthy()
  })

  it('renders sent state as one compact non-disclosure row', () => {
    const { container } = render(
      <AgentExchangeCard agent="Atlas" avatar={<span>S</span>} kind="sent" slot="test-agent-sent" />
    )

    expect(screen.getByText('Sent to')).toBeTruthy()
    expect(screen.getByText('Atlas')).toBeTruthy()
    expect(container.querySelector('details')).toBeNull()
  })

  it('opens the addressed agent DM from an exchange card', () => {
    const submit = vi.fn()
    const previous = window.hermesDesktop
    window.hermesDesktop = { quickEntry: { submit } } as unknown as Window['hermesDesktop']

    render(
      <AgentExchangeCard
        agent="Jarvis"
        avatar={<span>J</span>}
        body={<p>Review complete.</p>}
        bodyText="Review complete."
        kind="handoff"
        replyProfile="jarvis"
        slot="test-agent-reply"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Reply to Jarvis in their direct message' }))

    expect(submit).toHaveBeenCalledWith({
      action: 'open-agent',
      profile: 'jarvis',
      requestId: expect.any(String)
    })

    window.hermesDesktop = previous
  })
})
