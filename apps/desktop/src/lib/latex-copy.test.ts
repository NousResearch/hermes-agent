import { describe, expect, it } from 'vitest'

import { selectionTextWithLatex } from './latex-copy'

function select(container: HTMLElement, start: Node, end: Node) {
  const range = document.createRange()
  range.setStart(start, 0)
  range.setEnd(end, end.textContent?.length ?? 0)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)

  return selection
}

describe('selectionTextWithLatex', () => {
  it('copies inline KaTeX as its original dollar-delimited source', () => {
    const container = document.createElement('div')
    container.innerHTML =
      '<p>Then <span class="katex"><span class="katex-mathml"><math><semantics><annotation encoding="application/x-tex">P(|B^H(1)| &gt; R_1)</annotation></semantics></math></span><span class="katex-html">visual</span></span> follows.</p>'
    document.body.append(container)

    const paragraph = container.querySelector('p')!
    const selection = select(container, paragraph.firstChild!, paragraph.lastChild!)

    expect(selectionTextWithLatex(container, selection)).toBe('Then $P(|B^H(1)| > R_1)$ follows.')
    container.remove()
  })

  it('copies display KaTeX with double-dollar delimiters', () => {
    const container = document.createElement('div')
    container.innerHTML =
      '<div class="katex-display"><span class="katex"><math><semantics><annotation encoding="application/x-tex">\\frac{\\varepsilon}{2}</annotation></semantics></math></span></div>'
    document.body.append(container)

    const annotation = container.querySelector('annotation')!.firstChild!
    const selection = select(container, annotation, annotation)

    expect(selectionTextWithLatex(container, selection)).toBe('$$ \\frac{\\varepsilon}{2} $$')
    container.remove()
  })

  // Prose, code and currency have no KaTeX to restore, so the serializer must
  // decline and leave Chromium's native copy in charge.
  it('declines selections that contain no rendered math', () => {
    const container = document.createElement('div')
    container.innerHTML = '<p>It costs $5 and <code>$HOME</code> is unset.</p>'
    document.body.append(container)

    const paragraph = container.querySelector('p')!
    const selection = select(container, paragraph.firstChild!, paragraph.lastChild!)

    expect(selectionTextWithLatex(container, selection)).toBeNull()
    container.remove()
  })
})
