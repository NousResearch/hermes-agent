import { type ThreadMessage } from '@assistant-ui/react'
import {
  TOOL_ACTIVITY_HEADER_AREA,
  TOOL_RUN_PRESENTATION_API_VERSION,
  TOOL_RUN_PRESENTATION_AREA,
  type ToolActivityHeaderContext,
  type ToolRunPresentationContribution
} from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { setShowToolActivityFromConfig } from '@/store/tool-activity'
import { clearDismissedToolRows } from '@/store/tool-dismiss'
import { $toolDisclosureStates } from '@/store/tool-view'

import { createdAt, stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()
const disposers: (() => void)[] = []
const ctx = createPluginContext('presentation-fixture', dispose => disposers.push(dispose))
type Part = {
  toolName: string
  result?: unknown
  args?: Record<string, unknown>
  isError?: boolean
  interrupted?: boolean
  timestamp?: number
  completedAt?: number
}

const read: Part = {
  toolName: 'read_file',
  args: { path: '/repo/a.txt' },
  result: { content: 'native retained output' }
}

const terminal: Part = {
  toolName: 'terminal',
  args: { command: 'fixture-command' },
  result: { stdout: 'fixture stdout', exit_code: 0 }
}

let withdraw: (() => void) | undefined
let instances = 0

function Stamp(c: ToolActivityHeaderContext) {
  const [instance] = useState(() => ++instances)

  return (
    <span data-instance={instance} data-name={c.toolName} data-stamp="" data-status={c.status}>
      {c.toolName}:{c.status}
    </span>
  )
}

function mode(value: 'grouped' | 'individual') {
  act(() => {
    withdraw?.()
    withdraw = ctx.register({
      id: 'preference',
      area: TOOL_RUN_PRESENTATION_AREA,
      data: { apiVersion: TOOL_RUN_PRESENTATION_API_VERSION, mode: value } satisfies ToolRunPresentationContribution
    })
  })
}

function Transcript({ parts, running = false }: { parts: Part[]; running?: boolean }) {
  const message = {
    id: 'presentation-message',
    role: 'assistant',
    content: parts.map((p, i) => ({ type: 'tool-call', toolCallId: `call-${i}`, args: {}, argsText: '{}', ...p })),
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as unknown as ThreadMessage

  return (
    <ThreadRuntime messages={[message]}>
      <Thread />
    </ThreadRuntime>
  )
}

const rows = (container: HTMLElement) => container.querySelectorAll('[data-tool-row]')
const stamps = (container: HTMLElement) => [...container.querySelectorAll('[data-stamp]')]
beforeEach(() => {
  withdraw = undefined
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
  setShowToolActivityFromConfig(undefined)
})
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(d => d())
  withdraw = undefined
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
  setShowToolActivityFromConfig(undefined)
})
describe('native tool run presentation integration', () => {
  it('preserves default grouped DOM including the native summary after withdrawal', () => {
    const { container } = render(<Transcript parts={[read, terminal]} />)
    const native = container.querySelector('[data-tool-group]')!.outerHTML
    expect(container.querySelectorAll('[data-tool-summary]')).toHaveLength(1)
    expect(rows(container)).toHaveLength(0)
    mode('individual')
    expect(rows(container)).toHaveLength(2)
    expect(container.querySelector('[data-tool-group]')).toBeNull()
    expect(container.querySelector('[data-tool-summary]')).toBeNull()
    act(() => withdraw?.())
    expect(container.querySelector('[data-tool-group]')!.outerHTML).toBe(native)
  })
  it.each([
    { name: 'idle/no tools', parts: [] as Part[], running: false },
    { name: 'one running', parts: [{ toolName: 'terminal' }] as Part[], running: true },
    { name: 'multiple running', parts: [{ toolName: 'terminal' }, { toolName: 'read_file' }] as Part[], running: true },
    {
      name: 'failure',
      parts: [read, { ...terminal, result: { stderr: 'fixture failure', exit_code: 1 }, isError: true }],
      running: false
    },
    { name: 'cancelled', parts: [read, { toolName: 'terminal', interrupted: true }] as Part[], running: false },
    { name: 'historical', parts: [read, terminal], running: false }
  ])('switches both directions with $name, without changing transcript input', async ({ parts, running }) => {
    const before = JSON.stringify(parts)
    const { container } = render(<Transcript parts={parts} running={running} />)

    for (let i = 0; i < 2; i++) {
      mode('individual')
      await waitFor(() => expect(rows(container)).toHaveLength(parts.length))
      expect(container.querySelector('[data-tool-group]')).toBeNull()
      expect(container.querySelector('[data-tool-summary]')).toBeNull()
      mode('grouped')
      expect(container.querySelectorAll('[data-tool-group]')).toHaveLength(parts.length ? 1 : 0)
    }

    expect(JSON.stringify(parts)).toBe(before)
  })
  it('keeps row identities, ordering and header instances stable during streaming updates', async () => {
    mode('individual')
    ctx.register({ id: 'stamp', area: TOOL_ACTIVITY_HEADER_AREA, data: { apiVersion: 1, render: Stamp } })

    const { container, rerender } = render(
      <Transcript parts={[{ toolName: 'terminal' }, { toolName: 'read_file' }]} running />
    )

    const initial = stamps(container)
    expect(initial.map(s => s.getAttribute('data-status'))).toEqual(['running', 'running'])
    const identities = initial.map(s => s.getAttribute('data-instance'))
    rerender(<Transcript parts={[terminal, { toolName: 'read_file' }]} running />)
    await waitFor(() =>
      expect(stamps(container).map(s => s.getAttribute('data-status'))).toEqual(['success', 'running'])
    )
    rerender(<Transcript parts={[terminal, read]} />)
    await waitFor(() =>
      expect(stamps(container).map(s => s.getAttribute('data-status'))).toEqual(['success', 'success'])
    )
    expect(stamps(container)).toEqual(initial)
    expect(stamps(container).map(s => s.getAttribute('data-instance'))).toEqual(identities)
    expect(stamps(container).map(s => s.getAttribute('data-name'))).toEqual(['terminal', 'read_file'])
    expect(rows(container)).toHaveLength(2)
    // An unrelated registry update or same-mode replacement must not remount rows.
    mode('individual')
    ctx.register({ id: 'unrelated', area: 'fixture.other', data: {} })
    expect(stamps(container)).toEqual(initial)
  })
  it('retains an expanded group and each native output disclosure across scoped mode remounts', () => {
    const { container } = render(<Transcript parts={[read, terminal]} />)
    fireEvent.click(container.querySelector('[data-tool-summary] button')!)
    const button = rows(container)[0].querySelector('button[aria-expanded]')!
    fireEvent.click(button)
    expect(screen.getByText('native retained output')).toBeTruthy()
    const intent = { ...$toolDisclosureStates.get() }
    mode('individual')
    expect(rows(container)[0].querySelector('button[aria-expanded]')!.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('native retained output')).toBeTruthy()
    mode('grouped')
    expect(rows(container)).toHaveLength(2)
    expect(screen.getByText('native retained output')).toBeTruthy()
    expect($toolDisclosureStates.get()).toEqual(intent)
  })
  it('preserves card siblings and individual native keyboard/disclosure controls', () => {
    const patch: Part = {
      toolName: 'patch',
      args: { path: '/repo/a.txt' },
      result: { inline_diff: '--- a\n+++ b\n-old\n+new' }
    }

    const { container } = render(<Transcript parts={[read, patch, terminal]} />)
    const card = container.querySelector('[data-file-edit]')!
    expect(card).not.toBeNull()
    mode('individual')
    expect(container.querySelector('[data-file-edit]')).toBe(card)
    expect(rows(container)).toHaveLength(3)
    const button = rows(container)[0].querySelector<HTMLButtonElement>('button[aria-expanded]')!
    expect(button.tagName).toBe('BUTTON')
    expect(button.textContent).toContain('Read')
    button.focus()
    expect(container.ownerDocument.activeElement).toBe(button)
    // Native HTML button activation; real Enter/Space/Tab is browser-qualified separately.
    fireEvent.click(button)
    expect(button.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('native retained output')).toBeTruthy()
    mode('grouped')
    expect(container.querySelector('[data-file-edit]')).toBe(card)
  })
  it('preserves native status semantics for failed, cancelled and missing historical metadata in individual mode', () => {
    mode('individual')
    ctx.register({ id: 'status-stamp', area: TOOL_ACTIVITY_HEADER_AREA, data: { apiVersion: 1, render: Stamp } })

    const { container } = render(
      <Transcript
        parts={[
          read,
          { ...terminal, result: { stderr: 'failure fixture', exit_code: 1 }, isError: true },
          { toolName: 'unknown_tool', interrupted: true },
          { toolName: 'unknown_tool' }
        ]}
      />
    )

    expect(stamps(container).map(s => s.getAttribute('data-status'))).toEqual(['success', 'error', 'notice', 'warning'])
    expect(rows(container)).toHaveLength(4)
    expect(container.querySelector('[data-tool-group]')).toBeNull()
  })

  it('limits mode-change remounts to ordinary descendants, retaining native disclosure identity', () => {
    ctx.register({ id: 'mount-stamp', area: TOOL_ACTIVITY_HEADER_AREA, data: { apiVersion: 1, render: Stamp } })
    const { container } = render(<Transcript parts={[read, terminal]} />)
    fireEvent.click(container.querySelector('[data-tool-summary] button')!)
    const initial = stamps(container).map(s => s.getAttribute('data-instance'))
    mode('individual')
    expect(stamps(container).map(s => s.getAttribute('data-instance'))).not.toEqual(initial)
    const next = stamps(container)
    mode('individual')
    expect(stamps(container)).toEqual(next)
    mode('grouped')
    expect(rows(container)).toHaveLength(2)
    expect(stamps(container).map(s => s.getAttribute('data-instance'))).not.toEqual(
      next.map(s => s.getAttribute('data-instance'))
    )
  })

  it('does not override the independent native activity visibility preference', () => {
    mode('individual')
    setShowToolActivityFromConfig('off')
    const { container } = render(<Transcript parts={[read, terminal]} />)
    expect(rows(container)).toHaveLength(0)
    expect(container.querySelector('[data-tool-group]')).toBeNull()
  })
})
