import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// The real surface re-renders preprocessMarkdown → Streamdown on the accumulated
// text every flush, so a settled message never shows this bug: only frames do.
const norm = (value: string) => value.replace(/\s+/g, ' ').trim()

afterEach(cleanup)

const REASONED = '<thinking>let me think about the render pipeline in detail</thinking>Resposta final com acentuação.'

it('never paints the chain of thought, and never un-paints an answer, in any 3-char frame', async () => {
  const { container, rerender } = render(<MarkdownTextContent isRunning text="" />)
  let previous = ''

  for (let end = 3; end <= REASONED.length + 2; end += 3) {
    const frame = REASONED.slice(0, end)

    await act(async () => {
      rerender(<MarkdownTextContent isRunning text={frame} />)
      await Promise.resolve()
    })

    const visible = container.textContent ?? ''
    expect(visible).not.toContain('let me think')
    expect(visible).not.toMatch(/<\/?thinking/)

    // A half-arrived tag (`<thin`) is not a tag yet, so those frames sit outside the curve.
    if (!/<[a-z]*$/i.test(frame)) {
      expect(norm(visible).startsWith(previous)).toBe(true)
      previous = norm(visible)
    }
  }

  expect(container.textContent).toBe('Resposta final com acentuação.')
})
