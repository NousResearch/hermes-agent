import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { enModelMenu } from '@/i18n/en_model_menu'

import { displayPrice, ModelMetrics, modelPriceTitle, perThousand } from './model-metrics'

const compact = (tokens: number) =>
  new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(tokens)

afterEach(cleanup)

describe('ModelMetrics', () => {
  it('shows context, longest reply, image input and tool calling for a fully known model', () => {
    render(
      <ModelMetrics
        caps={{
          context_window: 1_048_576,
          fast: false,
          max_output: 32_768,
          reasoning: true,
          supports_tools: true,
          supports_vision: true
        }}
      />
    )

    expect(screen.getByText(compact(1_048_576)).getAttribute('title')).toBe(
      enModelMenu.contextTitle(compact(1_048_576))
    )
    expect(screen.getByText(enModelMenu.maxOutputLabel(compact(32_768))).getAttribute('title')).toBe(
      enModelMenu.maxOutputTitle(compact(32_768))
    )
    expect(screen.getByRole('img', { name: enModelMenu.vision })).toBeTruthy()
    expect(screen.getByRole('img', { name: enModelMenu.tools })).toBeTruthy()
  })

  it('never marks a capability the catalog says the model lacks', () => {
    render(
      <ModelMetrics
        caps={{ context_window: 128_000, fast: false, reasoning: true, supports_tools: false, supports_vision: false }}
      />
    )

    expect(screen.getByText(compact(128_000))).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('renders nothing for a model the catalog does not know (#112649)', () => {
    const { container } = render(<ModelMetrics caps={{ context_window: 0, fast: false, reasoning: true }} />)

    expect(container.innerHTML).toBe('')
  })
})

describe('displayPrice', () => {
  it('shows the backend figure per 1M and converts it for per 1K', () => {
    expect(displayPrice('$0.15', 'mtok')).toBe('$0.15')
    expect(displayPrice('$0.15', '1k')).toBe('$0.00015')
    expect(displayPrice('free', '1k')).toBe('free')
    expect(displayPrice('', 'mtok')).toBeNull()
  })
})

describe('price tooltip', () => {
  it('derives the per-1K price from the per-1M figure the chip shows', () => {
    expect(perThousand('$0.15')).toBe('$0.00015')
    expect(perThousand('$15.00')).toBe('$0.015')
    expect(perThousand('free')).toBeNull()
    expect(perThousand('')).toBeNull()
  })

  it('adds the per-1K line, and names a catalog list price only when the catalog supplied it', () => {
    const live = modelPriceTitle('base', { input: '$0.15', output: '$0.50' }, enModelMenu)
    const catalog = modelPriceTitle('base', { input: '$0.15', output: '$0.50', source: 'catalog' }, enModelMenu)

    expect(live).toBe(`base\n${enModelMenu.perThousandTitle('$0.00015', '$0.0005')}`)
    expect(catalog).toBe(`${live}\n${enModelMenu.catalogPrice}`)
    expect(modelPriceTitle('base', { input: 'free', output: 'free' }, enModelMenu)).toBe('base')
  })
})
