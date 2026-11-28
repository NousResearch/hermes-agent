/**
 * Execution tree rendering — session/subagent/tool rows, selection callback
 * and filter behavior, driven by store fixtures.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { AGENT_TRACE_LOCALES } from './i18n'
import { TraceTree } from './tree'
import type { SelectedStep, SessionTrace } from './types'

const T0 = 1_700_000_000

function sessionFixture(): SessionTrace {
  return {
    sessionId: 's1',
    profile: 'default',
    title: 'Refactor the gateway',
    firstSeenAt: T0,
    lastActivityAt: T0 + 5,
    turnActive: true,
    model: 'nous/qwen4-max',
    usage: { input: 1000, output: 200, context_used: 1200, context_max: 4000, context_percent: 30 },
    tools: [
      {
        id: 't1',
        name: 'read_file',
        status: 'done',
        origin: 'live',
        preview: '/tmp/x',
        startedAt: T0,
        endedAt: T0 + 1,
        durationS: 1
      },
      { id: 't2', name: 'terminal', status: 'running', origin: 'live', preview: 'pnpm test', startedAt: T0 + 2 }
    ],
    subagents: [
      {
        id: 'a1',
        depth: 0,
        goal: 'research competitors',
        taskIndex: 0,
        taskCount: 2,
        status: 'running',
        model: 'nous/qwen4-max',
        startedAt: T0 + 1,
        toolCount: 3,
        lastTool: 'web_search',
        inputTokens: 500,
        outputTokens: 100,
        reasoningTokens: 0,
        apiCalls: 2,
        outputTail: [],
        tools: [
          {
            id: 'a1:t0',
            name: 'web_search',
            status: 'done',
            origin: 'live',
            preview: 'q…',
            subagentId: 'a1',
            endedAt: T0 + 2
          }
        ]
      }
    ]
  }
}

let disposeLocales: () => void

beforeEach(() => {
  disposeLocales = registerPluginLocales('agent-trace', AGENT_TRACE_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
})

describe('TraceTree', () => {
  it('renders session, subagent and tool rows', () => {
    render(<TraceTree filter="" onSelect={() => {}} selected={null} sessions={[sessionFixture()]} />)
    expect(screen.getByText(/Refactor the gateway/)).toBeTruthy()
    expect(screen.getByText('research competitors')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('terminal')).toBeTruthy()
    expect(screen.getByText('web_search')).toBeTruthy()
    // "Running" paints on both the child badge and the running tool row.
    expect(screen.getAllByText('Running').length).toBeGreaterThanOrEqual(2)
  })

  it('selects a tool row with its session scope', () => {
    const onSelect = vi.fn<(sel: SelectedStep) => void>()
    render(<TraceTree filter="" onSelect={onSelect} selected={null} sessions={[sessionFixture()]} />)
    fireEvent.click(screen.getByText('read_file'))
    expect(onSelect).toHaveBeenCalledWith({ kind: 'tool', sessionId: 's1', toolId: 't1', subagentId: null })
  })

  it('selects a subagent node', () => {
    const onSelect = vi.fn<(sel: SelectedStep) => void>()
    render(<TraceTree filter="" onSelect={onSelect} selected={null} sessions={[sessionFixture()]} />)
    fireEvent.click(screen.getByText('research competitors'))
    expect(onSelect).toHaveBeenCalledWith({ kind: 'subagent', sessionId: 's1', subagentId: 'a1' })
  })

  it('filter hides non-matching rows and keeps matches', () => {
    render(<TraceTree filter="terminal" onSelect={() => {}} selected={null} sessions={[sessionFixture()]} />)
    expect(screen.getByText('terminal')).toBeTruthy()
    expect(screen.queryByText('read_file')).toBeNull()
  })

  it('filter matches subagent goals', () => {
    render(<TraceTree filter="competitors" onSelect={() => {}} selected={null} sessions={[sessionFixture()]} />)
    expect(screen.getByText('research competitors')).toBeTruthy()
  })
})
