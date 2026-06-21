import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $panesFlipped } from '@/store/layout'
import { $connection, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const selectPaths = vi.fn()

function ok(entries: { name: string; path: string; isDirectory: boolean }[]): HermesReadDirResult {
  return { entries }
}

function installBridge() {
  ;(
    window as unknown as {
      hermesDesktop: {
        readDir: typeof readDir
        selectPaths: typeof selectPaths
      }
    }
  ).hermesDesktop = { readDir, selectPaths }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    $panesFlipped.set(false)
    resetProjectTreeState()
    setCurrentCwd('/repo')
    readDir.mockReset()
    selectPaths.mockReset()
    readDir.mockResolvedValue(ok([{ name: 'README.md', path: '/repo/README.md', isDirectory: false }]))
    selectPaths.mockResolvedValue(['/repo-next'])
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    $panesFlipped.set(false)
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('refreshes the current tree without opening the folder picker', async () => {
    const onChangeCwd = vi.fn()

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={onChangeCwd} />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refresh tree' }).hasAttribute('disabled')).toBe(false)
    )

    readDir.mockClear()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh tree' }))

    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))
    expect(selectPaths).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Open folder' }))

    await waitFor(() =>
      expect(selectPaths).toHaveBeenCalledWith({
        defaultPath: '/repo',
        directories: true,
        multiple: false,
        title: 'Change working directory'
      })
    )
    await waitFor(() => expect(onChangeCwd).toHaveBeenCalledWith('/repo-next'))
  })

  it('keeps the file browser edge below the titlebar controls', () => {
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={vi.fn()} />)

    const pane = screen.getByLabelText('Right sidebar')

    expect(pane.className).toContain('before:top-(--titlebar-height)')
    expect(pane.className).toContain('before:left-0')
    expect(pane.className).not.toMatch(/\bborder-l\b|\bborder-r\b/)
  })

  it('keeps the clipped edge on the main-column side when panes are flipped', () => {
    $panesFlipped.set(true)

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={vi.fn()} />)

    const pane = screen.getByLabelText('Right sidebar')

    expect(pane.className).toContain('before:top-(--titlebar-height)')
    expect(pane.className).toContain('before:right-0')
    expect(pane.className).not.toContain('before:left-0')
    expect(pane.className).not.toMatch(/\bborder-l\b|\bborder-r\b/)
  })
})
