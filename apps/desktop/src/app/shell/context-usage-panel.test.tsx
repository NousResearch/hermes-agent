import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { renderHook } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { ContextUsagePanel } from './context-usage-panel'
import { useContextBreakdown } from './hooks/use-context-breakdown'

const usage: UsageStats = {
  calls: 1,
  context_max: 272_000,
  context_percent: 47,
  context_used: 128_200,
  input: 0,
  output: 0,
  total: 0
}

const breakdown: ContextBreakdown = {
  categories: [{ color: 'teal', id: 'conversation', label: 'Conversation', tokens: 241_400 }],
  context_max: 272_000,
  context_percent: 89,
  context_used: 241_400,
  estimated_total: 286_600,
  model: 'test-model'
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useContextBreakdown', () => {
  it('fetches for a session that has not run a turn yet', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await waitFor(() => expect(result.current.breakdown).toEqual(breakdown))
    expect(requestGateway).toHaveBeenCalledWith('session.context_breakdown', { session_id: 'runtime-1' })
  })

  it('does not fetch while the gauge is hidden, and fetches once it is shown', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { rerender } = renderHook(
      ({ enabled }) => useContextBreakdown({ busy: false, enabled, requestGateway, sessionId: 'runtime-1' }),
      { initialProps: { enabled: false } }
    )

    expect(requestGateway).not.toHaveBeenCalled()

    rerender({ enabled: true })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))
  })

  it('skips the estimate mid-turn — the gateway streams measured usage then', () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    renderHook(() => useContextBreakdown({ busy: true, enabled: true, requestGateway, sessionId: 'runtime-1' }))

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('refetches on a session switch and never reports the previous session numbers', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { rerender, result } = renderHook(
      ({ sessionId }) => useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId }),
      { initialProps: { sessionId: 'runtime-1' } }
    )

    await waitFor(() => expect(result.current.breakdown).toEqual(breakdown))

    // Switching sessions must drop the numbers immediately — painting them
    // under the new session's name would be a lie until its own fetch lands.
    requestGateway.mockImplementation(() => new Promise(() => undefined))
    rerender({ sessionId: 'runtime-2' })

    expect(result.current.breakdown).toBeNull()
    expect(requestGateway).toHaveBeenLastCalledWith('session.context_breakdown', { session_id: 'runtime-2' })
  })
})

describe('ContextUsagePanel', () => {
  it('marks estimates but preserves the provider-usage header', () => {
    for (const estimated of [true, false]) {
      const { container, unmount } = render(
        <ContextUsagePanel breakdown={breakdown} loading={false} usage={{ ...usage, context_estimated: estimated }} />
      )

      const header = container.querySelector('[data-slot="context-usage-panel"] > div')?.textContent ?? ''

      expect(header.includes('~')).toBe(estimated)
      expect(container.querySelector('li')?.textContent).toContain('~')
      unmount()
    }
  })

  it('renders the usage it is handed, so the popover matches the bar', () => {
    render(<ContextUsagePanel breakdown={breakdown} loading={false} usage={usage} />)

    expect(screen.getByText('47% Full')).toBeTruthy()
    expect(screen.getByText('Conversation')).toBeTruthy()
  })

  it('explains which context files were loaded or skipped', () => {
    render(
      <DropdownMenu open>
        <DropdownMenuTrigger>Context</DropdownMenuTrigger>
        <DropdownMenuContent>
          <ContextUsagePanel
            breakdown={{
              ...breakdown,
              context_files: [
                {
                  chars: 4_000,
                  est_tokens: 1_000,
                  label: 'AGENTS.md',
                  loaded: true,
                  path: '/repo/AGENTS.md',
                  status: 'loaded'
                },
                {
                  chars: 800,
                  est_tokens: 200,
                  label: 'CLAUDE.md',
                  loaded: false,
                  path: '/repo/CLAUDE.md',
                  status: 'shadowed'
                }
              ]
            }}
            loading={false}
            usage={usage}
          />
        </DropdownMenuContent>
      </DropdownMenu>
    )

    const trigger = screen.getByRole('menuitem', { name: 'Context files (2)' })

    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('/repo/AGENTS.md')).toBeNull()

    fireEvent.click(trigger)

    expect(trigger.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('AGENTS.md')).toBeTruthy()
    expect(screen.getByText(/~1k/i)).toBeTruthy()
    expect(screen.getByText('/repo/AGENTS.md')).toBeTruthy()
    expect(screen.getByText('Loaded')).toBeTruthy()
    expect(screen.getByText('Not loaded — a higher-priority context file won')).toBeTruthy()
  })

  it('is reachable and expandable from the statusbar menu with the keyboard', async () => {
    render(
      <MemoryRouter>
        <StatusbarControls
          items={[
            {
              id: 'context-manifest-test',
              label: '47%',
              menuContent: (
                <ContextUsagePanel
                  breakdown={{
                    ...breakdown,
                    context_files: [
                      {
                        chars: 4_000,
                        est_tokens: 1_000,
                        label: 'AGENTS.md',
                        loaded: true,
                        path: '/repo/AGENTS.md',
                        status: 'loaded'
                      }
                    ]
                  }}
                  loading={false}
                  usage={usage}
                />
              ),
              variant: 'menu'
            }
          ]}
        />
      </MemoryRouter>
    )

    const trigger = screen.getByRole('button', { name: '47%' })

    trigger.focus()
    fireEvent.keyDown(trigger, { key: 'ArrowDown' })

    const disclosure = await screen.findByRole('menuitem', { name: 'Context files (1)' })

    await waitFor(() => expect(globalThis.document.activeElement).toBe(disclosure))
    fireEvent.keyDown(disclosure, { key: 'Enter' })

    expect(disclosure.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('/repo/AGENTS.md')).toBeTruthy()
    expect(screen.getByRole('menu')).toBeTruthy()
  })
})
