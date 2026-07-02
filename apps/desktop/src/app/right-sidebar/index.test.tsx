import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $connection, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders the tree whenever the session has a working dir (repo or not) — no picker', async () => {
    setCurrentCwd('/repo')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })

    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))

    // The freeform folder picker is retired.
    expect(screen.queryByRole('button', { name: 'Open folder' })).toBeNull()
  })

  it('marks file tree rows with metadata for the native context menu', async () => {
    const rect = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 240,
      height: 240,
      left: 0,
      right: 320,
      toJSON: () => ({}),
      top: 0,
      width: 320,
      x: 0,
      y: 0
    })

    try {
      setCurrentCwd('/repo')

      render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

      const label = await screen.findByText('README.md')
      const row = label.closest('[data-hermes-file-tree-path]')

      expect(row?.getAttribute('data-hermes-file-tree-path')).toBe('/repo/README.md')
      expect(row?.getAttribute('data-hermes-file-tree-name')).toBe('README.md')
      expect(row?.getAttribute('data-hermes-file-tree-is-directory')).toBe('false')
    } finally {
      rect.mockRestore()
    }
  })

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })
})
