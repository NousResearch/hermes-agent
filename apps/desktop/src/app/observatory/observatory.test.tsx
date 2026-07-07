import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { buildRunTimeline } from '@/store/run-timeline'
import { $activeSessionId, $messages, $selectedStoredSessionId, $sessions } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'
import type { SessionInfo } from '@/types/hermes'

import { ObservatoryView, parentToolInputs, timelineSummary } from './index'

const CREATED_S = 1_700_000_000

const { getSessionChildren } = vi.hoisted(() => ({ getSessionChildren: vi.fn() }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSessionChildren
}))

// ObservatoryView now calls useQuery, so renders need a QueryClientProvider.
function renderWithQuery(node: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

const session = (id: string, title = id, overrides: Partial<SessionInfo> = {}): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: CREATED_S,
  title,
  tool_call_count: 0,
  ...overrides
})

function toolMessage(id: string, calls: { id: string; name: string; startedAt?: number; durationS?: number }[]): ChatMessage {
  return {
    id,
    role: 'assistant',
    parts: calls.map(c => ({
      type: 'tool-call',
      toolCallId: c.id,
      toolName: c.name,
      args: c.startedAt !== undefined ? { started_at: c.startedAt } : {},
      ...(c.durationS !== undefined ? { result: { duration_s: c.durationS } } : {})
    })) as never
  }
}

beforeEach(() => {
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $messages.set([])
  $sessions.set([])
  $subagentsBySession.set({})
  getSessionChildren.mockReset()
  getSessionChildren.mockResolvedValue({ session_id: '', children: [] })
})

afterEach(cleanup)

describe('parentToolInputs', () => {
  it('extracts tool calls with started_at from args and duration_s from result', () => {
    const msgs = [
      toolMessage('m1', [
        { id: 't1', name: 'terminal', startedAt: CREATED_S, durationS: 2 },
        { id: 't2', name: 'read_file', startedAt: CREATED_S + 5, durationS: 1 }
      ])
    ]

    const inputs = parentToolInputs(msgs)
    expect(inputs).toHaveLength(2)
    expect(inputs[0]).toEqual({ toolCallId: 't1', name: 'terminal', startedAt: CREATED_S, durationS: 2 })
    expect(inputs[1].name).toBe('read_file')
  })

  it('ignores non-tool-call parts', () => {
    const msg: ChatMessage = { id: 'm', role: 'assistant', parts: [{ type: 'text', text: 'hi' }] as never }
    expect(parentToolInputs([msg])).toEqual([])
  })
})

describe('timelineSummary', () => {
  it('counts blocks, families, and outliers', () => {
    const tl = buildRunTimeline(
      [],
      [
        { toolCallId: 'a', name: 'terminal', startedAt: CREATED_S, durationS: 1 },
        { toolCallId: 'b', name: 'browser_navigate', startedAt: CREATED_S + 2, durationS: 30 }
      ],
      CREATED_S * 1000,
      'p',
      'Parent'
    )

    const s = timelineSummary(tl)
    expect(s.blocks).toBe(2)
    expect(s.outliers).toBe(1)
    expect(s.byFamily.terminal).toBe(1)
    expect(s.byFamily.browser).toBe(1)
  })
})

describe('ObservatoryView', () => {
  it('shows the empty state with no run activity', () => {
    renderWithQuery(<ObservatoryView onClose={vi.fn()} />)
    expect(screen.getByText('Observatory')).toBeTruthy()
    expect(screen.getByText('No run activity yet')).toBeTruthy()
  })

  it('renders the canvas when the active session has tool calls', () => {
    $activeSessionId.set('s1')
    $sessions.set([session('s1', 'My run')])
    $messages.set([toolMessage('m1', [{ id: 't1', name: 'terminal', startedAt: CREATED_S, durationS: 2 }])])

    renderWithQuery(<ObservatoryView onClose={vi.fn()} />)
    expect(screen.getByTestId('timeline-canvas')).toBeTruthy()
    expect(screen.getByTestId('timeline-minimap')).toBeTruthy()
    expect(screen.getByTestId('block-t1')).toBeTruthy()
  })

  it('switches to the Charts tab', () => {
    $activeSessionId.set('s1')
    $sessions.set([session('s1')])
    $messages.set([toolMessage('m1', [{ id: 't1', name: 'terminal', startedAt: CREATED_S, durationS: 2 }])])

    renderWithQuery(<ObservatoryView onClose={vi.fn()} />)
    fireEvent.click(screen.getByText('Charts'))
    expect(screen.getByText(/Total tool blocks/)).toBeTruthy()
  })

  it('calls onClose from the close button', () => {
    const onClose = vi.fn()
    renderWithQuery(<ObservatoryView onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalled()
  })

  it('renders persisted child lanes for a finished session with no live subagents', async () => {
    // Finished session (ended_at set), parent has one tool block, no live subs.
    $activeSessionId.set('done-1')
    $sessions.set([session('done-1', 'Finished run', { ended_at: CREATED_S + 120 })])
    $messages.set([toolMessage('m1', [{ id: 't1', name: 'terminal', startedAt: CREATED_S, durationS: 2 }])])
    getSessionChildren.mockResolvedValue({
      session_id: 'done-1',
      children: [
        {
          id: 'child-a',
          title: 'scout the repo',
          started_at: CREATED_S + 1,
          ended_at: CREATED_S + 40,
          tool_call_count: 22,
          message_count: 32,
          model: 'gpt-5.5',
          status: 'completed'
        }
      ]
    })

    renderWithQuery(<ObservatoryView onClose={vi.fn()} />)

    // The persisted child renders as a bar lane once the query resolves.
    await waitFor(() => expect(screen.getByTestId('bar-child-a')).toBeTruthy())
    expect(getSessionChildren).toHaveBeenCalledWith('done-1', undefined)
    // Header flags the historical source.
    expect(screen.getAllByText(/historical/).length).toBeGreaterThan(0)
  })

  it('renders persisted child lanes for a selected stored session even when the active runtime id differs', async () => {
    // Regression for real Desktop QA: the visible/loaded transcript is keyed by
    // the stored session id, but activeSessionId may point at a live runtime/new
    // chat id. Observatory must query historical children for the selected
    // stored session, otherwise the backend 404s and the UI falls back to a
    // parent-only lane.
    $activeSessionId.set('runtime-current')
    $selectedStoredSessionId.set('stored-historical')
    $sessions.set([session('stored-historical', 'Stored historical run', { ended_at: CREATED_S + 120 })])
    $messages.set([toolMessage('m1', [{ id: 't1', name: 'delegate_task', startedAt: CREATED_S, durationS: 1 }])])
    getSessionChildren.mockResolvedValue({
      session_id: 'stored-historical',
      children: [
        {
          id: 'stored-child',
          title: null,
          started_at: CREATED_S + 1,
          ended_at: CREATED_S + 60,
          tool_call_count: 6,
          message_count: 13,
          model: 'gpt-5.5',
          status: 'completed'
        }
      ]
    })

    renderWithQuery(<ObservatoryView onClose={vi.fn()} />)

    await waitFor(() => expect(screen.getByTestId('bar-stored-child')).toBeTruthy())
    expect(getSessionChildren).toHaveBeenCalledWith('stored-historical', undefined)
    expect(getSessionChildren).not.toHaveBeenCalledWith('runtime-current', undefined)
    expect(screen.getAllByText(/historical/).length).toBeGreaterThan(0)
  })

  it('does NOT fetch persisted children when the live store already has subagents', () => {
    $activeSessionId.set('live-1')
    $sessions.set([session('live-1', 'Live run')])
    $messages.set([toolMessage('m1', [{ id: 't1', name: 'terminal', startedAt: CREATED_S, durationS: 2 }])])
    $subagentsBySession.set({
      'live-1': [
        {
          id: 'sa-live',
          parentId: null,
          goal: 'live child',
          status: 'running',
          taskCount: 1,
          taskIndex: 0,
          startedAt: 0,
          updatedAt: 0,
          toolCount: 3,
          filesRead: [],
          filesWritten: [],
          stream: []
        }
      ]
    })

    renderWithQuery(<ObservatoryView onClose={vi.fn()} />)

    // Live child lane renders; the persisted endpoint is never queried.
    expect(screen.getByTestId('bar-sa-live')).toBeTruthy()
    expect(getSessionChildren).not.toHaveBeenCalled()
  })
})
