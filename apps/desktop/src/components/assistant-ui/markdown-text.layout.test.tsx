import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

class OverflowingResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}

  observe(target: Element) {
    Object.defineProperty(target, 'scrollHeight', { configurable: true, value: 400 })
    this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }

  unobserve() {}
  disconnect() {}
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('assistant markdown layout', () => {
  it('lets fenced code fill the reading column and scroll horizontally without a height disclosure', async () => {
    vi.stubGlobal('ResizeObserver', OverflowingResizeObserver)

    const source = 'const wide = "'.concat('x'.repeat(240), '"')
    const { container } = render(<MarkdownTextContent isRunning={false} text={`\`\`\`js\n${source}\n\`\`\``} />)

    await screen.findByRole('button', { name: 'Copy code' })

    const card = container.querySelector<HTMLElement>('[data-slot="code-card"]')
    const body = container.querySelector<HTMLElement>('[data-slot="code-card-body"]')
    const pre = container.querySelector<HTMLElement>('pre.aui-shiki')

    expect(card).not.toBeNull()
    expect(card?.className).toContain('w-full')
    expect(card?.className).toContain('max-w-none')
    expect(body?.className).toContain('min-w-0')
    expect(pre?.className).toContain('overflow-x-auto')
    expect(screen.queryByRole('button', { name: 'Expand' })).toBeNull()
  })

  it('keeps markdown tables full-width with overflow owned by the table surface', async () => {
    const { container } = render(
      <MarkdownTextContent
        isRunning={false}
        text={'| Package | Description |\n| --- | --- |\n| hermes-agent | A deliberately wide table cell |'}
      />
    )

    await waitFor(() => expect(container.querySelector('.aui-md-table')).not.toBeNull())

    const surface = container.querySelector<HTMLElement>('.aui-md-table')
    const table = surface?.querySelector('table')

    expect(surface?.className).toContain('overflow-x-auto')
    expect(table?.className).toContain('w-full')
  })
})
