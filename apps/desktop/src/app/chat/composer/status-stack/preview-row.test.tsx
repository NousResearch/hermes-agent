import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { PreviewStatusRow } from './preview-row'

describe('PreviewStatusRow', () => {
  afterEach(() => {
    cleanup()
  })

  it('keeps the preview tooltip label inline inside the portaled decoration', async () => {
    const view = render(
      <PreviewStatusRow
        item={{
          cwd: 'C:\\repo',
          id: 'preview.html',
          label: 'preview.html',
          target: 'preview.html'
        }}
        onDismiss={() => undefined}
      />
    )

    fireEvent.pointerMove(screen.getByText('preview.html'), { pointerType: 'mouse' })
    await screen.findByRole('tooltip')

    const content = document.querySelector<HTMLElement>('[data-slot="tooltip-content"]')
    const decoration = content?.firstElementChild
    const label = decoration?.firstElementChild

    expect(content).not.toBeNull()
    expect(view.container.contains(content)).toBe(false)
    expect(label?.classList.contains('inline-flex')).toBe(true)
    expect(label?.classList.contains('flex')).toBe(false)
  })
})
