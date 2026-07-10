import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $filePreviewTarget, clearSessionPreviewRegistry } from '@/store/preview'
import { $connection, $currentCwd } from '@/store/session'

import { MarkdownTextContent } from '../assistant-ui/markdown-text'

import { MarkdownArtifactLink } from './markdown-artifact-link'

describe('MarkdownArtifactLink', () => {
  afterEach(() => {
    cleanup()
    clearSessionPreviewRegistry()
    $connection.set(null)
    $currentCwd.set('')
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
    vi.unstubAllGlobals()
  })

  it('opens a normal click in the read-only in-app artifact viewer', async () => {
    const normalizePreviewTarget = vi.fn(async () => ({
      kind: 'file' as const,
      label: 'brief.md',
      language: 'markdown',
      path: '/work/brief.md',
      previewKind: 'text' as const,
      source: './brief.md',
      url: 'file:///work/brief.md'
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget }
    })
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/work')

    const view = render(<MarkdownArtifactLink href="./brief.md">Read the brief</MarkdownArtifactLink>)
    fireEvent.click(view.getByRole('link', { name: 'Read the brief' }))

    await waitFor(() => expect($filePreviewTarget.get()?.artifact).toBe(true))
    expect($filePreviewTarget.get()).toMatchObject({
      artifact: true,
      path: '/work/brief.md'
    })
    expect(normalizePreviewTarget).toHaveBeenCalledWith('./brief.md', '/work')
  })

  it('routes file URLs and mentioned inline-code paths through the real transcript renderer', async () => {
    const normalizePreviewTarget = vi.fn(async (href: string) => ({
      kind: 'file' as const,
      label: 'brief.md',
      language: 'markdown',
      path: '/work/brief.md',
      previewKind: 'text' as const,
      source: href,
      url: 'file:///work/brief.md'
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget }
    })
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/work')

    const view = render(
      <MarkdownTextContent
        isRunning={false}
        text={'[Open file URL](file:///work/brief.md#decision) or inspect `/work/brief.md`.'}
      />
    )

    fireEvent.click(await view.findByRole('link', { name: 'Open file URL' }))
    await waitFor(() => expect($filePreviewTarget.get()?.artifact).toBe(true))
    expect(normalizePreviewTarget).toHaveBeenCalledWith('file:///work/brief.md', '/work')

    fireEvent.click(view.getByRole('link', { name: '/work/brief.md' }))
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledWith('/work/brief.md', '/work'))
  })

  it('keeps Markdown artifact links actionable inside lists and blockquotes', async () => {
    const normalizePreviewTarget = vi.fn(async (href: string) => ({
      kind: 'file' as const,
      label: 'brief.md',
      language: 'markdown',
      path: '/work/brief.md',
      previewKind: 'text' as const,
      source: href,
      url: 'file:///work/brief.md'
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget }
    })
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/work')

    const view = render(
      <MarkdownTextContent
        isRunning={false}
        text={'- [List artifact](file:///work/brief.md)\n\n> [Quoted artifact](file:///work/brief.md)'}
      />
    )

    fireEvent.click(await view.findByRole('link', { name: 'List artifact' }))
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledWith('file:///work/brief.md', '/work'))

    fireEvent.click(view.getByRole('link', { name: 'Quoted artifact' }))
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(2))
  })

  it('uses parsed Markdown destinations for parentheses, angle brackets, and references while leaving code inert', async () => {
    const normalizePreviewTarget = vi.fn(async (href: string) => ({
      kind: 'file' as const,
      label: 'brief.md',
      language: 'markdown',
      path: href,
      previewKind: 'text' as const,
      source: href,
      url: 'file:///work/brief.md'
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget }
    })
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/work')

    const view = render(
      <MarkdownTextContent
        isRunning={false}
        text={[
          '[Balanced](file:///work/Q1(report).md)',
          '[Angle](<file:///work/angle%20report.md>)',
          '[Reference][brief]',
          '',
          '[brief]: file:///work/reference.md',
          '',
          '`` `[Code](file:///work/code.md)` ``'
        ].join('\n')}
      />
    )

    for (const [name, href] of [
      ['Balanced', 'file:///work/Q1(report).md'],
      ['Angle', 'file:///work/angle%20report.md'],
      ['Reference', 'file:///work/reference.md']
    ]) {
      fireEvent.click(await view.findByRole('link', { name }))
      await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledWith(href, '/work'))
    }

    expect(view.queryByRole('link', { name: 'Code' })).toBeNull()
  })
})
