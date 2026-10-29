import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId } from '@/store/session'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

import { AgentsView } from './index'

const NOW = 1_000_000

Element.prototype.animate = vi.fn(() => ({ cancel() {} }) as Animation)

function subagent(overrides: Partial<SubagentProgress> = {}): SubagentProgress {
  return {
    id: 'worker-1',
    parentId: null,
    goal: 'Audit the release pipeline',
    model: 'gpt-6.1-sol',
    status: 'running',
    taskCount: 1,
    taskIndex: 0,
    startedAt: NOW - 42_000,
    updatedAt: NOW,
    durationSeconds: 42,
    costUsd: 0.08,
    inputTokens: 1200,
    outputTokens: 300,
    toolCount: 4,
    filesRead: ['scripts/release.ts'],
    filesWritten: ['apps/desktop/package.json'],
    stream: [
      { at: NOW - 4_000, kind: 'thinking', text: 'Checking the release manifest' },
      { at: NOW - 2_000, kind: 'tool', text: 'Read scripts/release.ts' },
      { at: NOW, kind: 'progress', text: 'Comparing signing and upload stages' }
    ],
    currentTool: 'read_file',
    ...overrides
  }
}

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $subagentsBySession.set({})
  vi.restoreAllMocks()
})

describe('AgentsView cards', () => {
  it('puts each worker status, activity, runtime, usage, files, cost, and recency on its card', () => {
    vi.spyOn(Date, 'now').mockReturnValue(NOW)
    $activeSessionId.set('owner')
    $subagentsBySession.set({ owner: [subagent()] })

    const { container } = render(<AgentsView onClose={vi.fn()} />)
    const card = container.querySelector<HTMLElement>('[data-slot="agent-card"]')

    expect(card).toBeTruthy()
    expect(within(card!).getByText('Audit the release pipeline')).toBeTruthy()
    expect(within(card!).getByText('Running')).toBeTruthy()
    expect(within(card!).getByText('Comparing signing and upload stages')).toBeTruthy()
    expect(within(card!).getByText('gpt-6.1-sol')).toBeTruthy()
    expect(within(card!).getByText('42.0s')).toBeTruthy()
    expect(within(card!).getByText('4 tools')).toBeTruthy()
    expect(within(card!).getByText('1.5k tok')).toBeTruthy()
    expect(within(card!).getByText('2 files')).toBeTruthy()
    expect(within(card!).getByText('$0.08')).toBeTruthy()
    expect(within(card!).getByText('updated now')).toBeTruthy()
  })

  it('keeps file paths and the longer activity trail in an accessible disclosure inside the card', () => {
    vi.spyOn(Date, 'now').mockReturnValue(NOW)
    $activeSessionId.set('owner')
    $subagentsBySession.set({ owner: [subagent()] })

    const { container } = render(<AgentsView onClose={vi.fn()} />)
    const card = container.querySelector<HTMLElement>('[data-slot="agent-card"]')!
    const disclosure = within(card).getByRole('button', { name: /Audit the release pipeline/ })

    expect(disclosure.getAttribute('aria-expanded')).toBe('false')
    expect(within(card).queryByText('+ apps/desktop/package.json')).toBeNull()
    expect(within(card).queryByText('Checking the release manifest')).toBeNull()
    expect(within(card).getByText('Comparing signing and upload stages')).toBeTruthy()

    fireEvent.click(disclosure)

    expect(disclosure.getAttribute('aria-expanded')).toBe('true')
    expect(within(card).getByText('+ apps/desktop/package.json')).toBeTruthy()
    expect(within(card).getByText('· scripts/release.ts')).toBeTruthy()
    expect(within(card).getByText('Checking the release manifest')).toBeTruthy()
  })

  it('renders queued and terminal workers as separate cards with visible state labels', () => {
    vi.spyOn(Date, 'now').mockReturnValue(NOW)
    $activeSessionId.set('owner')
    $subagentsBySession.set({
      owner: [
        subagent({ id: 'failed', goal: 'Check notarization', status: 'failed' }),
        subagent({ id: 'interrupted', goal: 'Inspect cancelled upload', status: 'interrupted' }),
        subagent({ id: 'queued', goal: 'Review accessibility', status: 'queued' }),
        subagent({ id: 'done', goal: 'Verify checksums', status: 'completed' })
      ]
    })

    const { container } = render(<AgentsView onClose={vi.fn()} />)
    const cards = Array.from(container.querySelectorAll<HTMLElement>('[data-slot="agent-card"]'))

    expect(cards).toHaveLength(4)
    expect(within(cards[0]!).getByText('Failed')).toBeTruthy()
    expect(within(cards[1]!).getByText('Failed')).toBeTruthy()
    expect(within(cards[2]!).getByText('Queued')).toBeTruthy()
    expect(within(cards[3]!).getByText('Done')).toBeTruthy()
    expect(screen.getByText('Check notarization')).toBeTruthy()
    expect(screen.getByText('Inspect cancelled upload')).toBeTruthy()
    expect(screen.getByText('Review accessibility')).toBeTruthy()
    expect(screen.getByText('Verify checksums')).toBeTruthy()
  })
})
