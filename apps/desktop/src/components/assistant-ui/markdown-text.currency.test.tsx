import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// A Brazilian price (`R$ 361,67`) must reach the DOM as plain prose. Before the
// preprocessor learned the `R$` form, the `$` opened a math span that swallowed
// the sentence up to the next price, so KaTeX painted it (spaces gone, italic)
// and a copy of the selection carried the MathML/HTML duplicate.
afterEach(cleanup)

const SENTENCE = 'O investimento fica em 12x de R$ 361,67 no cartão ou R$ 3.497 à vista.'

describe('real-currency prose in the rendered DOM', () => {
  it.each([
    ['finished', false],
    ['streaming', true]
  ])('renders R$ prices literally with no KaTeX/MathML (%s)', (_label, isRunning) => {
    const { container } = render(<MarkdownTextContent isRunning={isRunning} text={SENTENCE} />)

    expect(container.querySelector('.katex, math, .katex-mathml')).toBeNull()
    expect(container.textContent).toBe(SENTENCE)
  })

  it('still renders legitimate math beside a price', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'Custa R$ 3.497 e a fórmula é $x^2 + y^2$ e $$a^2$$.'} />
    )

    expect(container.querySelectorAll('.katex').length).toBeGreaterThanOrEqual(2)
    expect(container.textContent).toContain('Custa R$ 3.497 e a fórmula é ')
    expect(container.textContent).not.toContain('\\$')
  })
})
