import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'

import { ReasoningRenderBoundary } from './reasoning-render-boundary'

afterEach(cleanup)

function BrokenReasoningRenderer(): never {
  throw new RangeError('Maximum call stack size exceeded')
}

describe('ReasoningRenderBoundary', () => {
  it('renders reasoning normally while its rich renderer is healthy', () => {
    const { container } = render(
      <ReasoningRenderBoundary text="raw reasoning">
        <div>rich reasoning</div>
      </ReasoningRenderBoundary>
    )

    expect(screen.getByText('rich reasoning')).toBeTruthy()
    expect(container.querySelector('[data-render-fallback="reasoning"]')).toBeNull()
  })

  it('keeps reasoning readable when the rich renderer overflows', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    const { container } = render(
      <ReasoningRenderBoundary text={'first line\nsecond line'}>
        <BrokenReasoningRenderer />
      </ReasoningRenderBoundary>
    )

    const fallback = container.querySelector('[data-render-fallback="reasoning"]')

    expect(fallback?.textContent).toBe('first line\nsecond line')
    expect(fallback?.classList.contains('whitespace-pre-wrap')).toBe(true)
    spy.mockRestore()
  })

  it('contains deeply nested reasoning markdown inside the reasoning part', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const text = `${'> '.repeat(10_000)}still readable`

    const { container } = render(
      <ReasoningRenderBoundary text={text}>
        <MarkdownTextContent disableArtifacts isRunning={false} text={text} />
      </ReasoningRenderBoundary>
    )

    expect(container.querySelector('[data-render-fallback="reasoning"]')?.textContent).toBe(text)
    spy.mockRestore()
  })
})
