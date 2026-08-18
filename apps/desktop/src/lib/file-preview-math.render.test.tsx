import { render, waitFor } from '@testing-library/react'
import { Streamdown } from 'streamdown'
import { describe, expect, it } from 'vitest'

import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { normalizeFilePreviewMath } from '@/lib/markdown-preprocess'

// This is the exact pipeline the file preview runs: normalizeFilePreviewMath
// → Streamdown with the memoized math plugin. Rendering here (jsdom) proves the
// wiring produces KaTeX output, not raw `$..$` source text.
describe('file-preview math rendering', () => {
  const mathPlugin = createMemoizedMathPlugin({ singleDollarTextMath: true })

  it('renders inline $..$ math as KaTeX', async () => {
    const { container } = render(
      <Streamdown mode="static" plugins={{ math: mathPlugin }}>
        {normalizeFilePreviewMath('The formula $x^2 + y^2$ here.')}
      </Streamdown>
    )

    await waitFor(() => {
      expect(container.querySelector('.katex')).not.toBeNull()
    })
    expect(container.textContent).not.toContain('$x^2')
  })

  it('renders $$..$$ display math as KaTeX', async () => {
    const { container } = render(
      <Streamdown mode="static" plugins={{ math: mathPlugin }}>
        {normalizeFilePreviewMath('Block:\n\n$$E = mc^2$$')}
      </Streamdown>
    )

    await waitFor(() => {
      expect(container.querySelector('.katex-display') || container.querySelector('.katex')).not.toBeNull()
    })
  })

  it('renders delimited math as KaTeX', async () => {
    const { container } = render(
      <Streamdown mode="static" plugins={{ math: mathPlugin }}>
        {normalizeFilePreviewMath('A fraction \\(\\frac{a}{b}\\) inline.')}
      </Streamdown>
    )

    await waitFor(() => {
      expect(container.querySelector('.katex')).not.toBeNull()
    })
    expect(container.querySelector('.katex-mathml, .katex-html')).not.toBeNull()
  })

  it('leaves a literal code-fence $ alone as code, not math', async () => {
    const { container } = render(
      <Streamdown mode="static" plugins={{ math: mathPlugin }}>
        {normalizeFilePreviewMath('```bash\necho $HOME\n```')}
      </Streamdown>
    )

    await waitFor(() => {
      expect(container.querySelector('.katex')).toBeNull()
    })
    expect(container.querySelector('code')?.textContent).toContain('$HOME')
  })
})