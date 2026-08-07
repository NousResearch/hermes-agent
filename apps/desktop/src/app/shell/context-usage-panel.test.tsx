/**
 * Regression: the context usage bar must not normalize used tokens to 100%.
 *
 * The bar scaled every segment against the sum of used categories, so the
 * widths always summed to 100% and the track was drawn edge to edge no matter
 * how little of the window was consumed — a session at 26% full rendered a
 * completely full bar, and free space was never visible. The header ("N% Full")
 * and the bar disagreed.
 *
 * Default scaling is now against the whole context window, so the filled
 * portion matches the header. Clicking the bar switches to used-token scaling,
 * which fills the bar on purpose to compare small categories.
 */
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ContextUsagePanel } from './context-usage-panel'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      shell: {
        statusbar: {
          contextUsagePanel: {
            categories: {},
            empty: 'No context data yet',
            loading: 'Loading…',
            percentFull: (p: number) => `${p}% Full`,
            scaleToUsed: 'Scale bar to used tokens',
            scaleToWindow: 'Scale bar to full context window',
            showPercentages: 'Show %',
            showTokens: 'Show tokens',
            title: 'Context Usage',
            tokenSummary: (used: string, max: string) => `${used} / ${max} Tokens`
          }
        }
      }
    }
  })
}))

const CATEGORIES = [
  { color: '#aaa', id: 'system_prompt', label: 'System prompt', tokens: 5_000 },
  { color: '#bbb', id: 'tool_definitions', label: 'Tool definitions', tokens: 20_000 },
  { color: '#ccc', id: 'conversation', label: 'Conversation', tokens: 225_000 }
]

// 250k of a 1M window = 25% full.
const BREAKDOWN = {
  categories: CATEGORIES,
  context_max: 1_000_000,
  context_percent: 25,
  context_used: 250_000
}

const USAGE = { context_max: 1_000_000, context_percent: 25, context_used: 250_000 } as never

function renderPanel() {
  const requestGateway = vi.fn(async () => BREAKDOWN) as never

  return render(<ContextUsagePanel currentUsage={USAGE} requestGateway={requestGateway} sessionId="s1" />)
}

function segmentWidths(container: HTMLElement): number[] {
  const bar = container.querySelector('[data-slot="context-usage-bar"]')

  return Array.from(bar?.children ?? []).map(el =>
    Number.parseFloat((el as HTMLElement).style.width.replace('%', '')) || 0
  )
}

afterEach(cleanup)

describe('ContextUsagePanel bar scaling', () => {
  it('does not fill the bar when the window is mostly free', async () => {
    const { container } = renderPanel()
    await screen.findByText('25% Full')

    const total = segmentWidths(container).reduce((a, b) => a + b, 0)

    // The whole point: used tokens are 25% of the window, so the drawn portion
    // must be about 25%, leaving free space visible — never normalized to 100%.
    expect(total).toBeGreaterThan(20)
    expect(total).toBeLessThan(30)
  })

  it('keeps segment widths proportional to the window', async () => {
    const { container } = renderPanel()
    await screen.findByText('25% Full')

    const [systemPrompt, toolDefs, conversation] = segmentWidths(container)

    expect(systemPrompt).toBeCloseTo(0.5, 1) // 5k / 1M
    expect(toolDefs).toBeCloseTo(2, 1) // 20k / 1M
    expect(conversation).toBeCloseTo(22.5, 1) // 225k / 1M
  })

  it('fills the bar after switching to used-token scaling', async () => {
    const { container } = renderPanel()
    await screen.findByText('25% Full')

    fireEvent.click(screen.getByLabelText('Scale bar to used tokens'))

    const total = segmentWidths(container).reduce((a, b) => a + b, 0)

    expect(total).toBeCloseTo(100, 0)
  })

  it('toggles the list between tokens and percentages', async () => {
    renderPanel()
    await screen.findByText('25% Full')

    // Tokens by default (225k conversation).
    expect(screen.getByText('225k')).toBeTruthy()

    fireEvent.click(screen.getByText('Show %'))

    // 225k of a 1M window.
    expect(screen.getByText('22.5%')).toBeTruthy()
  })
})
