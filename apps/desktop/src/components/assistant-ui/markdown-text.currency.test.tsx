import { cleanup, render } from '@testing-library/react'
import { Streamdown } from 'streamdown'
import { afterEach, describe, expect, it } from 'vitest'

import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { normalizeFilePreviewMath, preprocessMarkdown } from '@/lib/markdown-preprocess'

import { MarkdownTextContent } from './markdown-text'

// Regression for #133839: two postfix prices must not delimit a math span.
afterEach(cleanup)

type Surface = 'chat' | 'file preview'
const mathPlugin = createMemoizedMathPlugin({ singleDollarTextMath: true })

function renderMarkdown(source: string, surface: Surface) {
  return render(
    surface === 'chat' ? (
      <MarkdownTextContent isRunning={false} text={source} />
    ) : (
      <Streamdown mode="static" plugins={{ math: mathPlugin }}>
        {normalizeFilePreviewMath(source)}
      </Streamdown>
    )
  )
}

describe.each<Surface>(['chat', 'file preview'])('postfix currency in %s Markdown', surface => {
  it('preserves a lone postfix amount without a second dollar to pair with', () => {
    const source = 'il costo è 12,87 $'
    const { container } = renderMarkdown(source, surface)

    expect(container.textContent).toBe(source)
    expect(container.querySelectorAll('.katex')).toHaveLength(0)
    expect(preprocessMarkdown(source)).toBe('il costo è 12,87 \\$')
    expect(normalizeFilePreviewMath(source)).toBe('il costo è 12,87 \\$')
  })

  it.each(['inline code', 'fenced code'])('preserves shell dollars inside %s', route => {
    const command = "ip route get $(getent hosts example.com | awk '{print $1;exit}')"
    const source = route === 'inline code' ? `\`${command}\`` : `\`\`\`bash\n${command}\n\`\`\``
    const { container } = renderMarkdown(source, surface)

    expect(container.querySelector('code')?.textContent?.trimEnd()).toBe(command)
    expect(container.querySelectorAll('.katex')).toHaveLength(0)
    expect(preprocessMarkdown(source)).toBe(source)
    expect(normalizeFilePreviewMath(source)).toBe(source)
  })

  it.each<[string, string, string, string[]]>([
    [
      'an invoice with emphasis',
      "Le sous-total est 1 000,00 $, la remise **10 %** s'applique, et le total à payer est **1 500,00 $** - livraison incluse.",
      "Le sous-total est 1 000,00 $, la remise 10 % s'applique, et le total à payer est 1 500,00 $ - livraison incluse.",
      ['10 %', '1 500,00 $']
    ],
    [
      'nonbreaking spaces',
      'Budget 1 000,00\u00a0$ ; total 1 500,00\u202f$.',
      'Budget 1 000,00\u00a0$ ; total 1 500,00\u202f$.',
      []
    ],
    ['compact amounts', 'Budget 1000$ ; total 1500$.', 'Budget 1000$ ; total 1500$.', []],
    ['compact arithmetic', '1000$+200$=1200$', '1000$+200$=1200$', []],
    ['an attached annotation', '1000$(tax included), total 1500$', '1000$(tax included), total 1500$', []],
    [
      'prose before emphasis',
      'Budget 1000 $ and **discount** applied; total 1500 $.',
      'Budget 1000 $ and discount applied; total 1500 $.',
      ['discount']
    ],
    [
      'an article before emphasis',
      'Budget 1000 $ with a **discount** applied; total 1500 $.',
      'Budget 1000 $ with a discount applied; total 1500 $.',
      ['discount']
    ],
    [
      'an arithmetic price list',
      '1 000,00 $ + 200,00 $ = 1 200,00 $ ; frais 50,00 $.',
      '1 000,00 $ + 200,00 $ = 1 200,00 $ ; frais 50,00 $.',
      []
    ]
  ])('renders %s with literal dollars and normal prose', (_label, source, expected, emphasis) => {
    const { container } = renderMarkdown(source, surface)

    expect(container.querySelectorAll('.katex')).toHaveLength(0)
    expect(container.textContent).toBe(expected)
    expect(
      Array.from(container.querySelectorAll('strong, [data-streamdown="strong"]'), node => node.textContent)
    ).toEqual(emphasis)
  })

  it.each([' ', '\u00a0', '\u202f', ''])('preserves a postfix dollar inside inline code (%j)', space => {
    const code = `1000${space}$`
    const source = `Use \`${code}\` literally; total 1 500,00 $.`
    const expected = `Use \`${code}\` literally; total 1 500,00 \\$.`

    const { container } = renderMarkdown(source, surface)

    expect(container.querySelector('code')?.textContent).toBe(code)
    expect(container.querySelectorAll('.katex')).toHaveLength(0)
    expect(preprocessMarkdown(source)).toBe(expected)
    expect(normalizeFilePreviewMath(source)).toBe(expected)
  })

  it.each([
    [
      'numeric',
      'Le total est $1000 / 1{,}05 = 952{,}38$ et la remise est 1 500,00 $.',
      'Le total est $1000 / 1{,}05 = 952{,}38$ et la remise est 1 500,00 \\$.'
    ],
    ['symbolic', 'Calcul $x^2$ ; total 1 500,00 $.', 'Calcul $x^2$ ; total 1 500,00 \\$.'],
    ['symbolic after a number', 'At n=2 $x^2$; total 1 500,00 $.', 'At n=2 $x^2$; total 1 500,00 \\$.'],
    ['symbolic after a NBSP', 'At n=2\u00a0$x^2$; total 1 500,00 $.', 'At n=2\u00a0$x^2$; total 1 500,00 \\$.'],
    ['symbolic touching a number', 'At n=2$x^2$; total 1 500,00 $.', 'At n=2$x^2$; total 1 500,00 \\$.'],
    ['parenthesized after a number', 'At n=2 $(x+y)$; total 1 500,00 $.', 'At n=2 $(x+y)$; total 1 500,00 \\$.'],
    [
      'parenthesized after a NBSP',
      'At n=2\u00a0$(x+y)$; total 1 500,00 $.',
      'At n=2\u00a0$(x+y)$; total 1 500,00 \\$.'
    ],
    ['parenthesized touching a number', 'At n=2$(x+y)$; total 1 500,00 $.', 'At n=2$(x+y)$; total 1 500,00 \\$.'],
    ['a variable after a number', 'At n=2 $x$; total 1 500,00 $.', 'At n=2 $x$; total 1 500,00 \\$.'],
    [
      'a TeX command after a number',
      'At n=2 $\\sqrt{4}$; total 1 500,00 $.',
      'At n=2 $\\sqrt{4}$; total 1 500,00 \\$.'
    ],
    ['spaced', 'Calcul $ 2 + 2 $ ; total 1 500,00 $.', 'Calcul $ 2 + 2 $ ; total 1 500,00 \\$.'],
    ['spaced numeric after a number', 'At n=2 $ 2 + 2 $; total 1 500,00 $.', 'At n=2 $ 2 + 2 $; total 1 500,00 \\$.'],
    [
      'spaced numeric after a NBSP',
      'At n=2\u00a0$ 2 + 2 $; total 1 500,00 $.',
      'At n=2\u00a0$ 2 + 2 $; total 1 500,00 \\$.'
    ],
    ['display', '$$\nE = mc^2\n$$\n\nTotal 1 500,00 $.', '$$\nE = mc^2\n$$\n\nTotal 1 500,00 \\$.']
  ])('preserves complete %s math beside a price on both surfaces', (_label, source, expected) => {
    const { container } = renderMarkdown(source, surface)

    expect(container.querySelectorAll('.katex')).toHaveLength(1)
    expect(container.textContent).toContain('1 500,00 $.')
    expect(preprocessMarkdown(source)).toBe(expected)
    expect(normalizeFilePreviewMath(source)).toBe(expected)
  })
})
