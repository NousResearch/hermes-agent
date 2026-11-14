import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanTask } from './types'
import { TaskAttention } from './ui'

let disposeLocales: () => void
beforeEach(() => {
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})
afterEach(() => {
  cleanup()
  disposeLocales()
})

const task: KanbanTask = { id: 't_question', title: 'Original title', status: 'review' }

describe('current task attention', () => {
  it.each(['awaiting-answer', 'awaiting-test', 'run-failed'] as const)('shows a distinct %s badge', kind => {
    render(<TaskAttention task={{ ...task, current_attention: { kind, run_id: 9 } }} />)
    expect(screen.getByText(en.attention[kind])).toBeTruthy()
  })

  it('does not guess a question from legacy prose', () => {
    const { container } = render(<TaskAttention task={{ ...task, latest_summary: 'Please answer my question' }} />)
    expect(container.textContent).toBe('')
  })

  it('clears a question when the authoritative field disappears', () => {
    const { rerender } = render(
      <TaskAttention task={{ ...task, current_attention: { kind: 'awaiting-answer', run_id: 9 } }} />
    )

    expect(screen.getByText(en.attention['awaiting-answer'])).toBeTruthy()
    rerender(<TaskAttention task={task} />)
    expect(screen.queryByText(en.attention['awaiting-answer'])).toBeNull()
  })

  it('clears stale attention during an optimistic status transition', () => {
    const { container } = render(
      <TaskAttention task={{ ...task, status: 'running', current_attention: { kind: 'awaiting-answer', run_id: 9 } }} />
    )

    expect(container.textContent).toBe('')
  })

  it('shows the concrete detail as escaped text', () => {
    const summary = '<img src=x onerror="alert(1)"> Which option should I use?'

    const { container } = render(
      <TaskAttention detail task={{ ...task, current_attention: { kind: 'awaiting-answer', run_id: 9, summary } }} />
    )

    expect(screen.getByText(summary)).toBeTruthy()
    expect(container.querySelector('img')).toBeNull()
  })
})

describe('producer review badge', () => {
  it('replaces the testing label with one green badge in card and drawer', () => {
    const selected = { ...task, current_attention: {
      kind: 'awaiting-test' as const, run_id: 9, badge: { label: 'Final PR', tone: 'success' as const }
    } }

    const { rerender } = render(<TaskAttention task={selected} />)
    expect(screen.queryByText(en.attention['awaiting-test'])).toBeNull()
    expect(screen.getAllByText('Final PR')).toHaveLength(1)
    expect(screen.getByText('Final PR').className).toContain('text-emerald-600')
    rerender(<TaskAttention detail task={selected} />)
    expect(screen.getAllByText('Final PR')).toHaveLength(1)
    rerender(<TaskAttention task={{ ...selected, status: 'ready' }} />)
    expect(screen.queryByText('Final PR')).toBeNull()
  })

  it.each(['awaiting-answer', 'run-failed'] as const)('does not replace %s with a stale badge', kind => {
    render(<TaskAttention task={{ ...task, current_attention: {
      kind, run_id: 9, badge: { label: 'Old approval', tone: 'success' }
    } }} />)
    expect(screen.queryByText('Old approval')).toBeNull()
    expect(screen.getByText(en.attention[kind])).toBeTruthy()
  })

  it.each([
    ['awaiting-test', 'text-blue-600'], ['awaiting-answer', 'text-amber-600'], ['run-failed', 'text-destructive']
  ] as const)('uses the default color for %s', (kind, color) => {
    render(<TaskAttention task={{ ...task, current_attention: { kind, run_id: 9 } }} />)
    expect(screen.getByText(en.attention[kind]).className).toContain(color)
  })

  it('renders arbitrary producer wording as plain text', () => {
    const label = '<img src=x onerror=alert(1)>'

    const { container } = render(<TaskAttention task={{ ...task, current_attention: {
      kind: 'awaiting-test', run_id: 9, badge: { label, tone: 'info' }
    } }} />)

    expect(screen.getByText(label)).toBeTruthy()
    expect(container.querySelector('img')).toBeNull()
  })
})
