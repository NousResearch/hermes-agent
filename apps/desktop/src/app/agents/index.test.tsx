import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

import { AgentsView } from './index'

const getSubagentContext = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', () => ({
  getSubagentContext
}))

vi.mock('@/lib/use-enter-animation', () => ({
  useEnterAnimation: () => () => {}
}))

const subagent = (over: Partial<SubagentProgress> = {}): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: 'Inspect raw prompt',
  id: 'agent-1',
  parentId: null,
  startedAt: 1,
  status: 'running',
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 1,
  ...over
})

describe('AgentsView context inspector', () => {
  beforeEach(() => {
    getSubagentContext.mockReset()
    $subagentsBySession.set({})
  })

  afterEach(() => cleanup())

  it('shows the inspector affordance only when context metadata is available', () => {
    $subagentsBySession.set({
      s1: [subagent({ contextAvailable: true, contextSessionId: 'child-1' })],
      s2: [subagent({ goal: 'No context', id: 'agent-2' })]
    })

    render(<AgentsView onClose={vi.fn()} />)

    expect(screen.getByRole('button', { name: /inspect context/i })).toBeTruthy()
    expect(screen.getByText('No context')).toBeTruthy()
  })

  it('lazy-fetches and renders raw context as escaped JSON text', async () => {
    const rawHtml = '<img src=x onerror=alert(1)>'

    getSubagentContext.mockResolvedValue({
      artifact: {
        canonical_messages: [{ content: rawHtml, role: 'system' }],
        child_session_id: 'child-1',
        provider_request: {
          messages: [{ content: rawHtml, role: 'system' }],
          temperature: 0.1
        }
      },
      child_session_id: 'child-1',
      ok: true
    })

    $subagentsBySession.set({ s1: [subagent({ contextAvailable: true, contextSessionId: 'child-1' })] })

    render(<AgentsView onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /inspect context/i }))

    await waitFor(() => expect(getSubagentContext).toHaveBeenCalledWith('child-1'))
    expect(await screen.findByText('Raw context')).toBeTruthy()

    const rawContext = screen.getByText(
      (content: string, element: Element | null) => element?.tagName === 'PRE' && content.includes(rawHtml)
    )

    expect(rawContext.textContent).toContain('provider_request')
    expect(document.querySelector('img')).toBeNull()
  })

  it('renders an inline failure state when the endpoint fails', async () => {
    getSubagentContext.mockRejectedValue(new Error('missing artifact'))
    $subagentsBySession.set({ s1: [subagent({ contextAvailable: true, contextSessionId: 'child-1' })] })

    render(<AgentsView onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: /inspect context/i }))

    expect(await screen.findByText(/Context failed to load: missing artifact/)).toBeTruthy()
  })
})
