import { cleanup, render, screen } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const originalResizeObserver = globalThis.ResizeObserver

beforeAll(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      disconnect() {}
      observe() {}
      unobserve() {}
    }
  )
})

afterAll(() => {
  vi.stubGlobal('ResizeObserver', originalResizeObserver)
})

describe('MarkdownLink preview routing', () => {
  afterEach(() => cleanup())

  it.each([
    '/Users/andrewconsidine/Hermes-Workspace/context/report.md',
    'file:///Users/andrewconsidine/Hermes-Workspace/context/report.md',
    String.raw`C:\Users\Andrew\Documents\report.pdf`,
    './artifacts/chart.png'
  ])('leaves the ordinary local href %s to the capped message footer', href => {
    render(<MarkdownTextContent isRunning={false} text={`[artifact](${href})`} />)

    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it.each([
    ['inline code', '`[artifact](/tmp/example.md)`'],
    ['fenced code', '```markdown\n[artifact](/tmp/example.md)\n```']
  ])('does not promote local Markdown links inside %s', (_name, text) => {
    render(<MarkdownTextContent isRunning={false} text={text} />)

    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it('does not promote local Markdown images', () => {
    render(<MarkdownTextContent isRunning={false} text="![chart](/tmp/chart.png)" />)

    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it('preserves HTTP links as external anchors', async () => {
    render(<MarkdownTextContent isRunning={false} text="[docs](https://example.com/docs)" />)

    const link = await screen.findByRole('link', { name: 'docs' })

    expect(link.getAttribute('href')).toBe('https://example.com/docs')
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })

  it.each([
    '/Users/andrewconsidine/.ssh/config.pdf',
    '/tmp/client-private-key.pdf',
    'file:///Users/andrewconsidine/.aws/credentials.txt'
  ])('does not promote the secret-looking local link %s', href => {
    render(<MarkdownTextContent isRunning={false} text={`[secret](${href})`} />)

    expect(document.body.textContent).toContain('secret')
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })
})
