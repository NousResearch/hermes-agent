import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { bindApi } from './api'
import { Markdown } from './markdown'

const openExternal = vi.fn(async () => true)

function bindOs() {
  return bindApi(
    async () => ({}) as never,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn(),
    { os: { openExternal } as never }
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Markdown', () => {
  it('renders a numbered plan as a list, not as source text', () => {
    const { container } = render(<Markdown text={'1. first step\n2. second step'} />)

    const items = container.querySelectorAll('ol > li')
    expect(items).toHaveLength(2)
    expect(items[0].textContent).toBe('first step')
    // The literal "1." must not survive into the rendered text.
    expect(container.textContent).not.toContain('1.')
  })

  it('renders nested bullets, fenced code and emphasis', () => {
    const { container } = render(<Markdown text={'- top\n  - inner\n\n`code` and **bold**\n\n```\nraw *text*\n```'} />)

    expect(container.querySelector('ul ul li')?.textContent).toBe('inner')
    // Streamdown renders emphasis as its own span, not a bare <strong>.
    expect(container.querySelector('[data-streamdown="strong"]')?.textContent).toBe('bold')
    expect(container.querySelector('pre')?.textContent).toContain('raw *text*')
  })

  it('opens a link through the OS rather than navigating the renderer', () => {
    const dispose = bindOs()
    render(<Markdown text="see [the docs](https://example.com/docs)" />)

    const link = screen.getByRole('link', { name: 'the docs' })
    const event = new MouseEvent('click', { bubbles: true, cancelable: true })
    fireEvent(link, event)

    expect(openExternal).toHaveBeenCalledWith('https://example.com/docs')
    expect(event.defaultPrevented).toBe(true)
    dispose()
  })
})
