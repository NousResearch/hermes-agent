import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { renderHook } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { I18nProvider } from '@/i18n'
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
              categories: [
                { color: 'teal', id: 'conversation', label: 'Conversation', tokens: 241_400 },
                { color: 'amber', id: 'rules', label: 'Rules', tokens: 800 }
              ],
              context_files: [
                {
                  chars: 48_000,
                  est_tokens: 12_000,
                  label: 'AGENTS.md',
                  loaded: true,
                  path: '/repo/AGENTS.md',
                  status: 'truncated'
                },
                {
                  chars: 16_000,
                  est_tokens: 4_000,
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
    expect(screen.getByText('~12k full file')).toBeTruthy()
    expect(screen.getByText('~4k full file')).toBeTruthy()
    expect(screen.getByText('/repo/AGENTS.md')).toBeTruthy()
    expect(screen.getByText('Loaded — truncated at the context-file limit')).toBeTruthy()
    expect(screen.getByText('Not loaded — a higher-priority context file won')).toBeTruthy()
    expect(
      screen.getByText(
        'These are full-file estimates before truncation, not tokens used in the context total above.'
      )
    ).toBeTruthy()

    const categoryList = globalThis.document.querySelector('[data-slot="context-usage-panel"] > ul')
    const rules = [...(categoryList?.querySelectorAll('li') ?? [])].find(item => item.textContent?.includes('Rules'))

    expect(rules?.textContent).toContain('~800')
    expect(rules?.textContent).not.toContain('full file')
    expect(categoryList?.textContent).not.toContain('full file')

    const region = globalThis.document.getElementById(trigger.getAttribute('aria-controls') ?? '')
    const fileList = region?.querySelector('ul')

    expect(region?.tagName).toBe('DIV')
    expect(fileList?.querySelectorAll('li')).toHaveLength(2)
    expect(fileList?.querySelector('[data-status="truncated"]')?.textContent).toContain('~12k full file')
    expect(fileList?.querySelector('[data-status="shadowed"]')?.textContent).toContain('~4k full file')
    expect(region?.textContent).not.toMatch(/not included in totals/i)
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
    expect(globalThis.document.activeElement).toBe(disclosure)
    expect(screen.getByText('/repo/AGENTS.md')).toBeTruthy()
    expect(screen.getByText('~1k full file')).toBeTruthy()
    expect(screen.getByRole('menu')).toBeTruthy()

    const region = globalThis.document.getElementById(disclosure.getAttribute('aria-controls') ?? '')

    expect(region?.querySelector('ul > li')).toBeTruthy()
    expect(region?.contains(disclosure)).toBe(false)
  })

  it('uses the active locale for the full-file estimate note', () => {
    render(
      <I18nProvider configClient={null} initialLocale="de">
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
                    status: 'truncated'
                  }
                ]
              }}
              loading={false}
              usage={usage}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('menuitem', { name: 'Kontextdateien (1)' }))

    expect(screen.getByText('~1k ganze Datei')).toBeTruthy()
    expect(
      screen.getByText('Schätzungen der ganzen Datei vor dem Kürzen, nicht die Tokens der Kontext-Summe oben.')
    ).toBeTruthy()
    expect(screen.queryByText(/full file/i)).toBeNull()
    expect(screen.queryByText(/not included in totals/i)).toBeNull()
  })
})
