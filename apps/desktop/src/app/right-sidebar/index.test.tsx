import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $connection, setCurrentCwd } from '@/store/session'

import { resetBrowserWorkspace } from './files/browser-workspace'
import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const selectPaths = vi.fn(async () => ['/picked'])
const createFile = vi.fn(async (parent: string, name: string) => ({ path: `${parent}/${name}` }))
const createDirectory = vi.fn(async (parent: string, name: string) => ({ path: `${parent}/${name}` }))

function installBridge() {
  ;(
    window as unknown as {
      hermesDesktop: {
        createDirectory: typeof createDirectory
        createFile: typeof createFile
        readDir: typeof readDir
        selectPaths: typeof selectPaths
      }
    }
  ).hermesDesktop = { createDirectory, createFile, readDir, selectPaths }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    resetBrowserWorkspace()
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockImplementation(async path => ({
      entries: [{ isDirectory: false, name: 'README.md', path: `${path}/README.md` }]
    }))
    selectPaths.mockClear()
    createFile.mockClear()
    createDirectory.mockClear()
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    setCurrentCwd('')
    resetBrowserWorkspace()
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders compact inline Explorer navigation without a primary folder picker', async () => {
    setCurrentCwd('/repo')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await screen.findByRole('button', { name: 'Refresh tree' })

    expect((screen.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Forward' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Up' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Session root' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Current location' }).getAttribute('title')).toBe('/repo')
    expect(screen.queryByRole('button', { name: 'Choose folder' })).toBeNull()
    expect((screen.getByRole('button', { name: 'New file' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'New folder' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('reveals the full path with Ctrl+L and validates before navigating on Enter', async () => {
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)
    await screen.findByRole('button', { name: 'Refresh tree' })

    fireEvent.keyDown(window, { ctrlKey: true, key: 'l' })
    const location = screen.getByRole('textbox', { name: 'Location' })
    expect((location as HTMLInputElement).value).toBe('/repo')

    fireEvent.change(location, { target: { value: '/repo/src' } })
    fireEvent.keyDown(location, { key: 'Enter' })

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Current location' }).getAttribute('title')).toBe('/repo/src')
    )
    expect(readDir).toHaveBeenCalledWith('/repo/src')
    expect((screen.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('keeps the current tree and address editor open when a typed path is invalid', async () => {
    readDir.mockImplementation(async path =>
      path === '/missing'
        ? { entries: [], error: 'ENOENT' }
        : { entries: [{ isDirectory: false, name: 'README.md', path: `${path}/README.md` }] }
    )
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)
    await screen.findByRole('button', { name: 'Refresh tree' })

    fireEvent.click(screen.getByRole('button', { name: 'Current location' }))
    const location = screen.getByRole('textbox', { name: 'Location' })
    fireEvent.change(location, { target: { value: '/missing' } })
    fireEvent.keyDown(location, { key: 'Enter' })

    expect((await screen.findByRole('alert')).textContent).toContain('Could not open location')
    expect((location as HTMLInputElement).value).toBe('/missing')

    fireEvent.keyDown(location, { key: 'Escape' })
    expect(screen.getByRole('button', { name: 'Current location' }).getAttribute('title')).toBe('/repo')
  })

  it('cancels inline location editing with Escape', async () => {
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)
    await screen.findByRole('button', { name: 'Refresh tree' })

    fireEvent.click(screen.getByRole('button', { name: 'Current location' }))
    const location = screen.getByRole('textbox', { name: 'Location' })
    fireEvent.change(location, { target: { value: '/other' } })
    fireEvent.keyDown(location, { key: 'Escape' })

    expect(screen.queryByRole('textbox', { name: 'Location' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Current location' }).getAttribute('title')).toBe('/repo')
    expect(readDir).not.toHaveBeenCalledWith('/other')
  })

  it('does not follow session cwd updates after the initial browser seed', async () => {
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)
    await screen.findByRole('button', { name: 'Current location' })

    setCurrentCwd('/agent-updated-cwd')

    expect(screen.getByRole('button', { name: 'Current location' }).getAttribute('title')).toBe('/repo')
  })

  it('uses a folder picker only as the detached-chat fallback', async () => {
    setCurrentCwd('')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const choose = screen.getByRole('button', { name: 'Choose folder' })
    fireEvent.click(choose)

    await waitFor(() => expect(selectPaths).toHaveBeenCalledWith({ directories: true, multiple: false }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Current location' }).getAttribute('title')).toBe('/picked')
    )
  })

  it('creates files and folders in the browser location without disabling unrelated controls', async () => {
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)
    await screen.findByRole('button', { name: 'Refresh tree' })

    fireEvent.click(screen.getByRole('button', { name: 'New file' }))
    expect(screen.getByText('/repo')).toBeTruthy()
    const fileName = screen.getByRole('textbox', { name: 'File name' })
    fireEvent.change(fileName, { target: { value: 'notes.md' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create file' }))

    await waitFor(() => expect(createFile).toHaveBeenCalledWith('/repo', 'notes.md'))
    expect((screen.getByRole('button', { name: 'New folder' }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    const folderName = screen.getByRole('textbox', { name: 'Folder name' })
    fireEvent.change(folderName, { target: { value: 'docs' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create folder' }))

    await waitFor(() => expect(createDirectory).toHaveBeenCalledWith('/repo', 'docs'))
  })
})
