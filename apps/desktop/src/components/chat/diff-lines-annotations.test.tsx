import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { FileDiffPanel } from './diff-lines'

describe('FileDiffPanel annotations', () => {
  it('emits old/new line anchors from the existing diff gutter', () => {
    const onAnnotateLine = vi.fn()

    const rendered = render(
      <FileDiffPanel
        annotateLineLabel={line => `${line}`}
        diff={'@@ -3,2 +3,2 @@\n-old\n+new\n context'}
        onAnnotateLine={onAnnotateLine}
        path="src/example.ts"
        showLineNumbers
      />
    )

    const lineButtons = rendered.getAllByRole('button', { name: '3' })

    fireEvent.click(lineButtons[0])
    fireEvent.click(lineButtons[1])

    expect(onAnnotateLine).toHaveBeenNthCalledWith(1, expect.objectContaining({ excerpt: 'old', line: 3, side: 'old' }))
    expect(onAnnotateLine).toHaveBeenNthCalledWith(2, expect.objectContaining({ excerpt: 'new', line: 3, side: 'new' }))
    expect(rendered.getByText('old')).toBeTruthy()
    expect(rendered.getByText('new')).toBeTruthy()
    expect(rendered.getByText('context')).toBeTruthy()

    for (const button of lineButtons) {
      const row = button.closest('[data-diff-row]')

      expect(row).toBeTruthy()
      expect(row?.querySelector('[data-diff-old-line], [data-diff-new-line]')).toBeTruthy()
    }
  })

  it('keeps full-file markdown content in the same virtualized rows as its line numbers', () => {
    const rendered = render(
      <FileDiffPanel
        diff={'@@ -2,2 +2,2 @@\n-old paragraph\n+new paragraph\n unchanged\n@@ -8,1 +8,1 @@\n-old tail\n+new tail'}
        fullText={'heading\nnew paragraph\nunchanged\nspace\nbetween\nhunks\nkept\nnew tail'}
        path="notes.md"
        showLineNumbers
      />
    )

    expect(rendered.getByText('heading').closest('[data-diff-row]')?.textContent).toContain('heading')
    expect(rendered.getByText('new paragraph').closest('[data-diff-row]')?.textContent).toContain('new paragraph')
    expect(rendered.getByText('between').closest('[data-diff-row]')?.textContent).toContain('between')
    expect(rendered.getByText('new tail').closest('[data-diff-row]')?.textContent).toContain('new tail')
  })
})
