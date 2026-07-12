import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

const MARKDOWN = [
  '# Safe report',
  '<script>window.pwned = true</script>',
  '<img src="https://example.com/tracker.png" onerror="window.pwned = true">',
  '[unsafe](javascript:alert(1))',
  '[local handler](file:///tmp/other.md)',
  '```mermaid',
  'graph TD; A-->B',
  '```'
].join('\n')

const target = {
  artifact: true,
  filesystemKey: 'local',
  kind: 'file' as const,
  label: 'report.md',
  language: 'markdown',
  path: '/work/report.md',
  previewKind: 'text' as const,
  source: '/work/report.md',
  url: 'file:///work/report.md'
}

describe('read-only Markdown artifact preview', () => {
  const openExternal = vi.fn(async () => {})

  const readFileText = vi.fn(async (_path: string, options?: { complete?: boolean }) => ({
    byteSize: MARKDOWN.length,
    language: 'markdown',
    path: '/work/report.md',
    text: options?.complete ? `${MARKDOWN}\nComplete tail` : MARKDOWN,
    truncated: false
  }))

  const revealPath = vi.fn(async () => true)
  const writeClipboard = vi.fn(async () => true)
  const gitRoot = vi.fn(async () => null)

  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path.includes('complete=true')) {
      return { path: '/remote/report.md', text: '# Remote complete', truncated: false }
    }

    return { byteSize: 8, language: 'markdown', path: '/remote/report.md', text: '# Remote', truncated: false }
  })

  beforeEach(() => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api, gitRoot, openExternal, readFileText, revealPath, writeClipboard }
    })
    $connection.set({ mode: 'local' } as never)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
    vi.clearAllMocks()
  })

  it('renders Markdown as inert data and exposes all local artifact actions without editing', async () => {
    const view = render(<LocalFilePreview reloadKey={0} target={target} />)

    await view.findByRole('heading', { name: 'Safe report' })
    expect((view.getByRole('button', { name: 'Open' }) as HTMLButtonElement).disabled).toBe(false)
    expect((view.getByRole('button', { name: 'Reveal' }) as HTMLButtonElement).disabled).toBe(false)
    expect((view.getByRole('button', { name: 'Copy path' }) as HTMLButtonElement).disabled).toBe(false)
    expect((view.getByRole('button', { name: 'Copy contents' }) as HTMLButtonElement).disabled).toBe(false)
    expect(view.queryByRole('button', { name: /edit/i })).toBeNull()
    expect(view.container.querySelector('script, img, a, [data-streamdown="mermaid"]')).toBeNull()

    fireEvent.click(view.getByRole('button', { name: 'Copy contents' }))
    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith(`${MARKDOWN}\nComplete tail`))
  })

  it('keeps local OS actions visible but disabled for remote artifacts', async () => {
    $connection.set({ mode: 'remote' } as never)

    const view = render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          ...target,
          filesystemKey: 'remote::',
          path: '/remote/report.md',
          url: 'file:///remote/report.md'
        }}
      />
    )

    await view.findByRole('heading', { name: 'Remote' })
    expect((view.getByRole('button', { name: 'Open' }) as HTMLButtonElement).disabled).toBe(true)
    expect((view.getByRole('button', { name: 'Reveal' }) as HTMLButtonElement).disabled).toBe(true)
    expect((view.getByRole('button', { name: 'Copy path' }) as HTMLButtonElement).disabled).toBe(false)
    expect((view.getByRole('button', { name: 'Copy contents' }) as HTMLButtonElement).disabled).toBe(false)
    expect(openExternal).not.toHaveBeenCalled()
    expect(revealPath).not.toHaveBeenCalled()
  })

  it('keeps path recovery visible when artifact contents cannot be read', async () => {
    readFileText.mockRejectedValueOnce(new Error('ENOENT'))
    const view = render(<LocalFilePreview reloadKey={0} target={target} />)

    await view.findByText('ENOENT')
    expect((view.getByRole('button', { name: 'Copy path' }) as HTMLButtonElement).disabled).toBe(false)
    expect((view.getByRole('button', { name: 'Copy contents' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('keeps Copy contents disabled when a binary Markdown file is force-previewed', async () => {
    readFileText.mockResolvedValueOnce({
      binary: true,
      byteSize: 4,
      language: 'markdown',
      path: '/work/report.md',
      text: '\u0000BIN',
      truncated: false
    } as never)
    const view = render(<LocalFilePreview reloadKey={0} target={target} />)

    await view.findByRole('button', { name: /preview anyway/i })
    expect((view.getByRole('button', { name: 'Copy contents' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(view.getByRole('button', { name: /preview anyway/i }))
    await waitFor(() =>
      expect((view.getByRole('button', { name: 'Copy contents' }) as HTMLButtonElement).disabled).toBe(true)
    )
  })

  it('does not read a persisted artifact after its filesystem or profile changes', async () => {
    const view = render(
      <LocalFilePreview
        reloadKey={0}
        target={{ ...target, filesystemKey: 'remote:prod:https://gateway.example', path: '/srv/report.md' }}
      />
    )

    await view.findByText(/Reopen this artifact after switching filesystems/i)
    expect(readFileText).not.toHaveBeenCalled()
    expect((view.getByRole('button', { name: 'Copy path' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('exits edit mode when an existing file tab is reopened as a read-only artifact', async () => {
    const regularTarget = { ...target, artifact: undefined, filesystemKey: undefined }
    const view = render(<LocalFilePreview reloadKey={0} target={regularTarget} />)

    await view.findByRole('heading', { name: 'Safe report' })
    fireEvent.click(view.getByRole('button', { name: /edit/i }))
    expect(view.getByRole('button', { name: /cancel/i })).toBeTruthy()

    view.rerender(<LocalFilePreview reloadKey={0} target={target} />)

    await waitFor(() => expect(view.queryByRole('button', { name: /cancel/i })).toBeNull())
    expect(view.queryByRole('button', { name: /edit/i })).toBeNull()
    expect(view.getByRole('toolbar', { name: 'Markdown artifact actions' })).toBeTruthy()
  })

  it('reloads a same-path artifact when its remote filesystem provenance changes', async () => {
    let remoteText = '# Remote A'

    api.mockImplementation(async () => ({
      byteSize: remoteText.length,
      language: 'markdown',
      path: '/srv/report.md',
      text: remoteText,
      truncated: false
    }))

    $connection.set({ baseUrl: 'https://gateway.example/a', mode: 'remote', profile: 'a' } as never)

    const remoteA = {
      ...target,
      filesystemKey: 'remote:a:https://gateway.example/a',
      path: '/srv/report.md',
      url: 'file:///srv/report.md'
    }

    const view = render(<LocalFilePreview reloadKey={0} target={remoteA} />)

    await view.findByRole('heading', { name: 'Remote A' })
    remoteText = '# Remote B'

    const remoteB = { ...remoteA, filesystemKey: 'remote:b:https://gateway.example/b' }

    await act(async () => {
      $connection.set({ baseUrl: 'https://gateway.example/b', mode: 'remote', profile: 'b' } as never)
      view.rerender(<LocalFilePreview reloadKey={0} target={remoteB} />)
    })

    await view.findByRole('heading', { name: 'Remote B' })
    const readCalls = api.mock.calls.filter(([request]) => request.path.includes('/api/fs/read-text'))

    expect(readCalls).toHaveLength(2)
  })
})
