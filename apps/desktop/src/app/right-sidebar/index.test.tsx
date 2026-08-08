import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $panesFlipped } from '@/store/layout'
import { $connection, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'
import { ReviewPane } from './review'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    $panesFlipped.set(false)
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
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

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('keeps the file browser edge below the titlebar controls', () => {
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const pane = screen.getByLabelText('Right sidebar')

    expect(pane.className).toContain('before:top-(--titlebar-height)')
    expect(pane.className).toContain('before:left-0')
    expect(pane.className).not.toMatch(/\bborder-l\b|\bborder-r\b/)
  })

  it('keeps the clipped edge on the main-column side when panes are flipped', () => {
    $panesFlipped.set(true)

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const pane = screen.getByLabelText('Right sidebar')

    expect(pane.className).toContain('before:top-(--titlebar-height)')
    expect(pane.className).toContain('before:right-0')
    expect(pane.className).not.toContain('before:left-0')
    expect(pane.className).not.toMatch(/\bborder-l\b|\bborder-r\b/)
  })

  it('keeps the review edge below the titlebar controls', () => {
    render(<ReviewPane />)

    const pane = screen.getByLabelText('Review')

    expect(pane.className).toContain('before:top-(--titlebar-height)')
    expect(pane.className).toContain('before:left-0')
    expect(pane.className).not.toMatch(/\bborder-l\b|\bborder-r\b/)
  })

  it('keeps the review edge on the main-column side when panes are flipped', () => {
    $panesFlipped.set(true)

    render(<ReviewPane />)

    const pane = screen.getByLabelText('Review')

    expect(pane.className).toContain('before:top-(--titlebar-height)')
    expect(pane.className).toContain('before:right-0')
    expect(pane.className).not.toContain('before:left-0')
    expect(pane.className).not.toMatch(/\bborder-l\b|\bborder-r\b/)
  })
})
