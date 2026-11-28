/**
 * Inspection drawer rendering — args, latency, output/error and the history
 * note, driven by store fixtures (no gateway). Locale registration uses the
 * same harness the kanban tests use.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { InspectionDrawer } from './drawer'
import { AGENT_TRACE_LOCALES } from './i18n'
import { $selectedStep, $trace, EMPTY_TRACE } from './trace-store'
import type { SessionTrace, ToolCall } from './types'

const T0 = 1_700_000_000

function toolFixture(overrides: Partial<ToolCall> = {}): ToolCall {
  return {
    id: 't1',
    name: 'read_file',
    status: 'done',
    origin: 'live',
    args: { path: '/tmp/notes.txt' },
    startedAt: T0,
    endedAt: T0 + 1.4,
    durationS: 1.4,
    resultText: 'file body here',
    resultChars: 14,
    raw: '{"tool_id":"t1"}',
    ...overrides
  }
}

function sessionFixture(tools: ToolCall[]): SessionTrace {
  return {
    sessionId: 's1',
    profile: 'default',
    firstSeenAt: T0,
    lastActivityAt: T0 + 2,
    turnActive: false,
    model: 'nous/qwen4-max',
    usage: { context_used: 1200, context_max: 4000, context_percent: 30, input: 900, output: 300 },
    tools,
    subagents: []
  }
}

let client: QueryClient
let disposeLocales: () => void

function renderDrawer() {
  return render(
    <QueryClientProvider client={client}>
      <InspectionDrawer />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('agent-trace', AGENT_TRACE_LOCALES)
  $trace.set(EMPTY_TRACE)
  $selectedStep.set(null)
})

afterEach(() => {
  cleanup()
  disposeLocales()
})

describe('InspectionDrawer', () => {
  it('renders nothing without a selection', () => {
    const { container } = renderDrawer()
    expect(container.querySelector('[role="dialog"]')).toBeNull()
  })

  it('shows exact arguments, latency and context for a live tool call', () => {
    $trace.set({ sessions: [sessionFixture([toolFixture()])] })
    $selectedStep.set({ kind: 'tool', sessionId: 's1', toolId: 't1' })
    const { container } = renderDrawer()
    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    // Arguments tab is the default: pretty-printed JSON in the code block.
    expect(container.querySelector('pre')?.textContent).toContain('"/tmp/notes.txt"')
    expect(screen.getByText('1.4 s')).toBeTruthy()
    expect(screen.getByText('30%')).toBeTruthy()
    expect(screen.getByText('Live')).toBeTruthy()
  })

  it('switches to the output tab and shows the result text', () => {
    $trace.set({ sessions: [sessionFixture([toolFixture()])] })
    $selectedStep.set({ kind: 'tool', sessionId: 's1', toolId: 't1' })
    const { container } = renderDrawer()
    fireEvent.click(screen.getByText('Output'))
    expect(container.querySelector('pre')?.textContent).toContain('file body here')
  })

  it('surfaces the error trace for a failed call', () => {
    $trace.set({
      sessions: [
        sessionFixture([
          toolFixture({ status: 'error', error: 'Command exited with code 2', resultText: '{"exit_code":2}' })
        ])
      ]
    })
    $selectedStep.set({ kind: 'tool', sessionId: 's1', toolId: 't1' })
    renderDrawer()
    fireEvent.click(screen.getByText('Output'))
    // "Error" paints on both the status badge and the output-tab label.
    expect(screen.getAllByText('Error').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText(/Command exited with code 2/)).toBeTruthy()
  })

  it('explains missing output on a hydrated history row', () => {
    $trace.set({
      sessions: [
        sessionFixture([
          toolFixture({ origin: 'history', resultText: null, resultChars: null, durationS: null, raw: undefined })
        ])
      ]
    })
    $selectedStep.set({ kind: 'tool', sessionId: 's1', toolId: 't1' })
    renderDrawer()
    fireEvent.click(screen.getByText('Output'))
    expect(screen.getByText('History')).toBeTruthy()
    expect(screen.getByText(/Persisted row/)).toBeTruthy()
  })

  it('closes on Escape', () => {
    $trace.set({ sessions: [sessionFixture([toolFixture()])] })
    $selectedStep.set({ kind: 'tool', sessionId: 's1', toolId: 't1' })
    renderDrawer()
    expect($selectedStep.get()).not.toBeNull()
    // The shell listens on document; the event bubbles up from the dialog.
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
    expect($selectedStep.get()).toBeNull()
  })
})
