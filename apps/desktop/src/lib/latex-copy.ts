const BLOCK_TAGS = new Set([
  'ADDRESS',
  'BLOCKQUOTE',
  'DIV',
  'H1',
  'H2',
  'H3',
  'H4',
  'H5',
  'H6',
  'LI',
  'OL',
  'P',
  'PRE',
  'TABLE',
  'TR',
  'UL'
])

function plainText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent ?? ''
  }

  if (!(node instanceof HTMLElement)) {
    return [...node.childNodes].map(plainText).join('')
  }

  if (node.tagName === 'BR') {
    return '\n'
  }

  const text = [...node.childNodes].map(plainText).join('')

  return BLOCK_TAGS.has(node.tagName) ? `${text}\n` : text
}

function texSource(element: Element): string | null {
  return element.querySelector('annotation[encoding="application/x-tex"]')?.textContent?.trim() || null
}

/**
 * Return the selected rendered Markdown as plaintext while restoring KaTeX
 * nodes to their original LaTeX source. A selection ending inside a formula
 * expands to include that complete formula rather than copying KaTeX's
 * accessibility text one glyph at a time.
 */
export function selectionTextWithLatex(container: HTMLElement, selection: Selection): string | null {
  if (selection.isCollapsed || selection.rangeCount === 0) {
    return null
  }

  const original = selection.getRangeAt(0)

  if (!container.contains(original.commonAncestorContainer)) {
    return null
  }

  const range = original.cloneRange()

  const formulas = [...container.querySelectorAll('.katex')].filter(element => {
    try {
      return range.intersectsNode(element)
    } catch {
      return false
    }
  })

  if (formulas.length === 0) {
    return null
  }

  const startFormula = formulas.find(element => element.contains(range.startContainer))
  const endFormula = [...formulas].reverse().find(element => element.contains(range.endContainer))

  if (startFormula) {
    range.setStartBefore(startFormula.closest('.katex-display') ?? startFormula)
  }

  if (endFormula) {
    range.setEndAfter(endFormula.closest('.katex-display') ?? endFormula)
  }

  const holder = document.createElement('div')
  holder.append(range.cloneContents())

  for (const display of [...holder.querySelectorAll('.katex-display')]) {
    const source = texSource(display)

    if (source) {
      display.replaceWith(document.createTextNode(`$$ ${source} $$`))
    }
  }

  for (const inline of [...holder.querySelectorAll('.katex')]) {
    const source = texSource(inline)

    if (source) {
      inline.replaceWith(document.createTextNode(`$${source}$`))
    }
  }

  return plainText(holder).replace(/\n{3,}/g, '\n\n').trim()
}
