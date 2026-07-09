import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $connection, setCurrentCwd } from '@/store/session'

import { FileActionDialogs } from './file-actions'
import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

function renderSidebar() {
  return render(
    <>
      <RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />
      <FileActionDialogs />
    </>
  )
}

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const createTextFile = vi.fn<(path: string, content?: string) => Promise<{ path: string }>>()
const createFolder = vi.fn<(path: string) => Promise<{ path: string }>>()

function installBridge() {
  ;(
    window as unknown as {
      hermesDesktop: {
        createFolder: typeof createFolder
        createTextFile: typeof createTextFile
        readDir: typeof readDir
      }
    }
  ).hermesDesktop = { createFolder, createTextFile, readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 640,
      height: 640,
      left: 0,
      right: 320,
      top: 0,
      width: 320,
      x: 0,
      y: 0,
      toJSON: () => ({})
    } as DOMRect)
    $connection.set(null)
    resetProjectTreeState()
    readDir.mockReset()
    createTextFile.mockReset()
    createFolder.mockReset()
    createTextFile.mockResolvedValue({ path: '/repo/TODO.md' })
    createFolder.mockResolvedValue({ path: '/repo/docs' })
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    setCurrentCwd('')
    resetProjectTreeState()
    vi.restoreAllMocks()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders the tree whenever the session has a working dir (repo or not) — no picker', async () => {
    setCurrentCwd('/repo')

    renderSidebar()

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })

    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))

    // The freeform folder picker is retired.
    expect(screen.queryByRole('button', { name: 'Open folder' })).toBeNull()
  })

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    renderSidebar()

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('exposes standard file-manager toolbar controls and filters by file search', async () => {
    readDir.mockResolvedValue({
      entries: [
        { isDirectory: true, name: 'src', path: '/repo/src' },
        { isDirectory: false, name: 'README.md', path: '/repo/README.md' },
        { isDirectory: false, name: 'package.json', path: '/repo/package.json' }
      ]
    })
    setCurrentCwd('/repo')

    renderSidebar()

    const search = await screen.findByRole('textbox', { name: 'Search files' })
    expect(screen.getByRole('button', { name: 'Expand all folders' }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('button', { name: 'Collapse all folders' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'New file' }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('button', { name: 'New folder' }).hasAttribute('disabled')).toBe(false)

    await screen.findByText('README.md')
    expect(document.body.contains(screen.getByText('package.json'))).toBe(true)

    fireEvent.change(search, { target: { value: 'readme' } })

    expect(document.body.contains(screen.getByText('README.md'))).toBe(true)
    expect(screen.queryByText('package.json')).toBeNull()
  })

  it('creates files and folders from the file-manager toolbar', async () => {
    setCurrentCwd('/repo')

    renderSidebar()

    fireEvent.click(await screen.findByRole('button', { name: 'New file' }))
    fireEvent.change(await screen.findByLabelText('File name'), { target: { value: 'TODO.md' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create file' }))

    await waitFor(() => expect(createTextFile).toHaveBeenCalledWith('/repo/TODO.md', ''))

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    fireEvent.change(await screen.findByLabelText('Folder name'), { target: { value: 'docs' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create folder' }))

    await waitFor(() => expect(createFolder).toHaveBeenCalledWith('/repo/docs'))
    expect(readDir).toHaveBeenCalledWith('/repo')
  })
})
