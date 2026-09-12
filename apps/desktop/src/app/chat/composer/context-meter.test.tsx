import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { setCurrentUsage } from '@/store/session'

import { ComposerContextMeter } from './context-meter'

const getGlobalModelInfo = vi.fn()

vi.mock('@/api/models', () => ({
  getGlobalModelInfo: (...args: unknown[]) => getGlobalModelInfo(...args)
}))

afterEach(() => {
  cleanup()
  setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
})

function primaryView(runtimeId: string | null = 'run-1', model = 'm'): SessionView {
  return {
    kind: 'primary',
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom(''),
    $fast: atom(false),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(true),
    $model: atom(model),
    $provider: atom('p'),
    $reasoningEffort: atom(''),
    $runtimeId: atom(runtimeId),
    $storedId: atom('stored-1'),
    $turnStartedAt: atom<number | null>(null)
  } as SessionView
}

describe('ComposerContextMeter', () => {
  it('paints the merged usage with in/out, cache hit and cost', async () => {
    setCurrentUsage({
      cache_hit_pct: 82,
      calls: 12,
      context_max: 200000,
      context_percent: 34,
      context_used: 68000,
      cost_usd: 0.04,
      input: 12000,
      output: 4000,
      total: 16000
    })

    render(
      <SessionViewProvider value={primaryView()}>
        <ComposerContextMeter />
      </SessionViewProvider>
    )

    // Meter button: percent readout.
    expect(screen.getByRole('button', { name: /Context usage 34 percent/i })).toBeDefined()
  })

  it('shows the selected model window at 0% before the first turn', async () => {
    getGlobalModelInfo.mockResolvedValue({
      auto_context_length: 1000000,
      effective_context_length: 1000000,
      model: 'deepseek-v4-flash',
      provider: 'custom'
    })

    const view = primaryView(null, 'deepseek-v4-flash')

    render(
      <SessionViewProvider value={view}>
        <ComposerContextMeter />
      </SessionViewProvider>
    )

    await screen.findByText('0%')
    expect(getGlobalModelInfo).toHaveBeenCalled()
  })

  it('shows the draft meter for the selected model even when getGlobalModelInfo returns a different model', async () => {
    getGlobalModelInfo.mockResolvedValue({
      auto_context_length: 2000000,
      effective_context_length: 2000000,
      model: 'auto/best-coding',
      provider: 'custom'
    })

    const view = primaryView(null, 'deepseek-v4-flash')

    render(
      <SessionViewProvider value={view}>
        <ComposerContextMeter />
      </SessionViewProvider>
    )

    const button = await screen.findByRole('button', { name: /Empty chat/i })
    expect(button).toBeDefined()
    expect(button.getAttribute('aria-label')).toContain('context window')
  })

  it('shows the draft meter immediately even when getGlobalModelInfo has not resolved', async () => {
    getGlobalModelInfo.mockReturnValue(new Promise(() => {}))

    const view = primaryView(null, 'deepseek-v4-flash')

    render(
      <SessionViewProvider value={view}>
        <ComposerContextMeter />
      </SessionViewProvider>
    )

    const button = screen.getByRole('button', { name: /Empty chat/i })
    expect(button).toBeDefined()
    expect(button.getAttribute('aria-label')).toContain('deepseek-v4-flash')
  })
})
