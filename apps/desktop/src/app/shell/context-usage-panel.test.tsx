import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CompressSessionOutcome } from '@/app/session/compress-session'
import { clearNotifications } from '@/store/notifications'
import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { ContextMeterDetail, ContextUsagePanel } from './context-usage-panel'
import { _resetManualCompactingForTests, useCompactConversation } from './hooks/use-compact-conversation'
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

const compressed = (): CompressSessionOutcome => ({
  aborted: false,
  kind: 'done',
  message: 'compressed 12 → 2 messages',
  result: {},
  sessionId: 'runtime-1'
})

/** The popover as the statusbar mounts it: the real hook drives the real panel,
 *  so these tests click the buttons a user clicks. */
function CompactHarness({
  busy = false,
  compress,
  sessionId = 'runtime-1',
  storedSessionId
}: {
  busy?: boolean
  compress: (sessionId: string) => Promise<CompressSessionOutcome>
  sessionId?: string
  storedSessionId?: null | string
}) {
  const compact = useCompactConversation({ busy, compress, sessionId, storedSessionId })

  return <ContextUsagePanel breakdown={breakdown} compact={compact} loading={false} usage={usage} />
}

beforeEach(() => {
  _resetManualCompactingForTests()
  clearNotifications()
})

afterEach(() => {
  _resetManualCompactingForTests()
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

  it('reports the live compression count, zero included, and never invents one', () => {
    const { rerender } = render(<ContextUsagePanel breakdown={breakdown} loading={false} usage={usage} />)

    expect(screen.queryByTestId('context-panel-compressions')).toBeNull()

    for (const compressions of [0, 3]) {
      rerender(<ContextUsagePanel breakdown={breakdown} loading={false} usage={{ ...usage, compressions }} />)
      expect(screen.getByTestId('context-panel-compressions').textContent).toBe(`Compressions: ${compressions}`)
    }
  })
})

describe('ContextUsagePanel compact action', () => {
  it('opens a focused confirmation dialog and Escape cancels without dispatch', async () => {
    const compress = vi.fn().mockResolvedValue(compressed())
    render(<CompactHarness compress={compress} />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    const dialog = screen.getByRole('dialog')
    const confirm = screen.getByRole('button', { name: 'Compact' })
    await waitFor(() => expect(confirm.ownerDocument.activeElement).toBe(confirm))
    fireEvent.keyDown(dialog, { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(compress).not.toHaveBeenCalled()
  })
  it('confirms by Enter exactly once for the owning session', async () => {
    const compress = vi.fn().mockResolvedValue(compressed())
    render(<CompactHarness compress={compress} />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Enter' })
    await waitFor(() => expect(compress).toHaveBeenCalledTimes(1))
    expect(compress).toHaveBeenCalledWith('runtime-1')
  })

  it('permanently cancels confirmation when a reply starts', () => {
    const compress = vi.fn().mockResolvedValue(compressed())
    const { rerender } = render(<CompactHarness compress={compress} />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    rerender(<CompactHarness busy compress={compress} />)
    rerender(<CompactHarness compress={compress} />)
    expect(screen.queryByText('Compress this chat into a summary?')).toBeNull()
    expect(compress).not.toHaveBeenCalled()
  })

  it('dismisses confirmation on session switch without dispatch', () => {
    const compress = vi.fn().mockResolvedValue(compressed())
    const { rerender } = render(<CompactHarness compress={compress} sessionId="runtime-1" />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    rerender(<CompactHarness compress={compress} sessionId="runtime-2" />)
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(compress).not.toHaveBeenCalled()
  })

  it('asks before it compacts, and cancelling really cancels', () => {
    const compress = vi.fn().mockResolvedValue(compressed())

    render(<CompactHarness compress={compress} />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))

    expect(screen.getByText('Compress this chat into a summary?')).toBeTruthy()
    expect(compress).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(screen.queryByText('Compress this chat into a summary?')).toBeNull()
    expect(screen.getByRole('button', { name: 'Compact conversation' })).toBeTruthy()
    expect(compress).not.toHaveBeenCalled()
  })

  it('compacts the captured session once, then shows the running state instead of the button', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    render(<CompactHarness compress={compress} />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    fireEvent.click(screen.getByRole('button', { name: 'Compact' }))

    expect(compress).toHaveBeenCalledTimes(1)
    expect(compress).toHaveBeenCalledWith('runtime-1')
    expect(screen.getByTestId('context-compact-pending')).toBeTruthy()
    // No second affordance to click while the summarise call is out.
    expect(screen.queryByRole('button', { name: 'Compact' })).toBeNull()

    await act(async () => settle(compressed()))

    expect(screen.queryByTestId('context-compact-pending')).toBeNull()
    expect(screen.getByRole('button', { name: 'Compact conversation' })).toBeTruthy()
  })

  it('drops a late refusal for the chat that is no longer on screen', async () => {
    let fail: (err: unknown) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>((_resolve, reject) => {
          fail = reject
        })
    )

    const { rerender } = render(<CompactHarness compress={compress} sessionId="runtime-1" />)

    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    fireEvent.click(screen.getByRole('button', { name: 'Compact' }))

    rerender(<CompactHarness compress={compress} sessionId="runtime-2" />)
    await act(async () => fail(new Error('session busy: compression')))

    // The refused chat is gone from the popover; the failure must not print
    // under the chat now in view.
    expect(screen.queryByTestId('context-compact-error')).toBeNull()

    rerender(<CompactHarness compress={compress} sessionId="runtime-1" />)
    expect(screen.getByTestId('context-compact-error').textContent).toContain('session busy: compression')
  })

  it('stays in the running state when recovery re-mints the runtime id mid-flight', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    const { rerender } = render(<CompactHarness compress={compress} sessionId="runtime-1" storedSessionId="stored-1" />)

    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    fireEvent.click(screen.getByRole('button', { name: 'Compact' }))
    expect(screen.getByTestId('context-compact-pending')).toBeTruthy()

    // Same chat, fresh runtime id (the dead one 404'd during the summarise).
    rerender(<CompactHarness compress={compress} sessionId="runtime-1-live" storedSessionId="stored-1" />)

    expect(screen.getByTestId('context-compact-pending')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Compact conversation' })).toBeNull()
    expect(compress).toHaveBeenCalledTimes(1)

    await act(async () => settle(compressed()))
    expect(screen.getByRole('button', { name: 'Compact conversation' })).toBeTruthy()
  })

  it('prints a backend refusal where the action was opened', async () => {
    const compress = vi.fn().mockRejectedValue(new Error('session busy: compression'))

    render(<CompactHarness compress={compress} />)
    fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Compact' })))

    expect(screen.getByTestId('context-compact-error').textContent).toContain('session busy: compression')
    expect(screen.getByRole('button', { name: 'Compact conversation' })).toBeTruthy()
  })

  it('cannot compact mid-turn, and says why', () => {
    const compress = vi.fn().mockResolvedValue(compressed())

    render(<CompactHarness busy compress={compress} />)

    const button = screen.getByRole('button', { name: 'Compact conversation' }) as HTMLButtonElement

    expect(button.disabled).toBe(true)
    expect(screen.getByText('Finish or stop the current reply first')).toBeTruthy()
    fireEvent.click(button)
    expect(compress).not.toHaveBeenCalled()
  })
})

describe('ContextMeterDetail', () => {
  it('adds the count to the meter only once the session has compacted', () => {
    for (const compressions of [undefined, 0]) {
      const { container, unmount } = render(<ContextMeterDetail bar="[██░░] 47%" compressions={compressions} />)

      expect(container.textContent).toBe('[██░░] 47%')
      unmount()
    }

    render(<ContextMeterDetail bar="[██░░] 47%" compressions={2} />)

    expect(screen.getByTestId('context-meter-compressions').textContent).toBe('2')
  })
})
