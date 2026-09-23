import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopFsRemotePicker } from '@/lib/desktop-fs'

const fsMocks = vi.hoisted(() => ({
  picker: null as DesktopFsRemotePicker | null,
  readDesktopDir: vi.fn(),
  setDesktopFsRemotePicker: vi.fn((picker: DesktopFsRemotePicker | null) => {
    fsMocks.picker = picker
  })
}))

vi.mock('@/lib/desktop-fs', () => ({
  readDesktopDir: fsMocks.readDesktopDir,
  setDesktopFsRemotePicker: fsMocks.setDesktopFsRemotePicker
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      rightSidebar: {
        emptyBody: 'No folders',
        loadingFiles: 'Loading',
        remotePickerDescription: 'Browse folders on the connected backend.',
        remotePickerPathLabel: 'Folder path',
        remotePickerPathPlaceholder: 'Type a path or folder name',
        remotePickerSelect: 'Select folder',
        remotePickerTitle: 'Choose remote folder',
        unreadableBody: (error: string) => `Unreadable: ${error}`
      }
    }
  })
}))

import { filterAndRankFolders, planFolderQuery, RemoteFolderPicker } from './remote-picker'

describe('remote folder picker query planning', () => {
  it('keeps a simple query in the current directory for fuzzy filtering', () => {
    expect(planFolderQuery('hagt', '/srv')).toEqual({ browsePath: '/srv', filter: 'hagt' })
  })

  it('discovers children under the deepest explicit parent path', () => {
    expect(planFolderQuery('/srv/herm', '/home/kosta')).toEqual({ browsePath: '/srv', filter: 'herm' })
    expect(planFolderQuery('/srv/hermes/', '/home/kosta')).toEqual({ browsePath: '/srv/hermes', filter: '' })
    expect(planFolderQuery('~/LocalDev/herm', '/home/kosta')).toEqual({ browsePath: '~/LocalDev', filter: 'herm' })
    expect(planFolderQuery('C:\\Users\\Kosta\\herm', 'C:\\Users\\Kosta')).toEqual({
      browsePath: 'C:\\Users\\Kosta',
      filter: 'herm'
    })
    expect(planFolderQuery('C:/Users/Kosta/herm', 'C:/Users/Kosta')).toEqual({
      browsePath: 'C:/Users/Kosta',
      filter: 'herm'
    })
  })

  it('resolves relative subdirectory queries from the directory being browsed', () => {
    expect(planFolderQuery('projects/herm', '/home/kosta')).toEqual({
      browsePath: '/home/kosta/projects',
      filter: 'herm'
    })
  })
})

describe('remote folder picker fuzzy matching', () => {
  const folders = [
    { name: 'hermes-agent', path: '/srv/hermes-agent' },
    { name: 'home-agent-tools', path: '/srv/home-agent-tools' },
    { name: 'agent-harness', path: '/srv/agent-harness' },
    { name: 'docs', path: '/srv/docs' }
  ]

  it('supports non-contiguous matches and ranks tighter matches first', () => {
    expect(filterAndRankFolders(folders, 'hagt').map(entry => entry.name)).toEqual([
      'home-agent-tools',
      'hermes-agent'
    ])
  })

  it('prefers prefix and contiguous matches without changing the unfiltered order', () => {
    expect(filterAndRankFolders(folders, '').map(entry => entry.name)).toEqual(folders.map(entry => entry.name))
    expect(filterAndRankFolders(folders, 'agent').map(entry => entry.name)).toEqual([
      'agent-harness',
      'home-agent-tools',
      'hermes-agent'
    ])
  })
})

describe('RemoteFolderPicker', () => {
  beforeEach(() => {
    fsMocks.picker = null
    fsMocks.readDesktopDir.mockReset()
    fsMocks.setDesktopFsRemotePicker.mockClear()
    fsMocks.readDesktopDir.mockImplementation(async (path: string) => {
      if (path === '/srv') {
        return {
          entries: [
            { isDirectory: true, name: 'hermes-agent', path: '/srv/hermes-agent' },
            { isDirectory: true, name: 'docs', path: '/srv/docs' }
          ]
        }
      }

      if (path === '/srv/hermes-agent') {
        return {
          entries: [{ isDirectory: true, name: 'packages', path: '/srv/hermes-agent/packages' }]
        }
      }

      return { entries: [], error: 'ENOENT' }
    })
  })

  it('accepts a path, fuzzy-matches its final segment, and discovers the selected directory children', async () => {
    render(<RemoteFolderPicker />)
    await waitFor(() => expect(fsMocks.picker).not.toBeNull())

    let selection!: Promise<string[]>
    act(() => {
      selection = fsMocks.picker!.selectPaths({ defaultPath: '/srv', directories: true })
    })

    const input = await screen.findByRole('combobox', { name: 'Folder path' })
    await screen.findByText('docs')
    fireEvent.change(input, { target: { value: '/srv/herm' } })

    await screen.findByText('hermes-agent')
    await waitFor(() => expect(screen.queryByText('docs')).toBeNull())

    fireEvent.keyDown(input, { key: 'Enter' })
    await screen.findByText('packages')
    expect((input as HTMLInputElement).value).toBe('/srv/hermes-agent')
    expect(fsMocks.readDesktopDir).toHaveBeenCalledWith('/srv/hermes-agent')

    fireEvent.keyDown(input, { key: 'Enter' })
    await expect(selection).resolves.toEqual(['/srv/hermes-agent'])
  })

  it('keeps the initial directory read alive when the user starts fuzzy-searching immediately', async () => {
    let resolveInitial!: (value: unknown) => void
    fsMocks.readDesktopDir.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveInitial = resolve
        })
    )
    render(<RemoteFolderPicker />)
    await waitFor(() => expect(fsMocks.picker).not.toBeNull())

    act(() => {
      void fsMocks.picker!.selectPaths({ defaultPath: '/srv', directories: true })
    })
    const input = await screen.findByRole('combobox', { name: 'Folder path' })
    await waitFor(() => expect(fsMocks.readDesktopDir).toHaveBeenCalledWith('/srv'))
    fireEvent.change(input, { target: { value: 'herm' } })

    await act(async () => {
      resolveInitial({
        entries: [{ isDirectory: true, name: 'hermes-agent', path: '/srv/hermes-agent' }]
      })
    })

    expect(await screen.findByText('hermes-agent')).toBeTruthy()
  })

  it('does not allow an unmatched partial path to select its readable parent', async () => {
    render(<RemoteFolderPicker />)
    await waitFor(() => expect(fsMocks.picker).not.toBeNull())
    act(() => {
      void fsMocks.picker!.selectPaths({ defaultPath: '/srv', directories: true })
    })

    const input = await screen.findByRole('combobox', { name: 'Folder path' })
    await screen.findByText('docs')
    fireEvent.change(input, { target: { value: '/srv/not-a-folder' } })
    expect((screen.getByRole('button', { name: 'Select folder' }) as HTMLButtonElement).disabled).toBe(true)
    await screen.findByText('No folders')

    expect((screen.getByRole('button', { name: 'Select folder' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('clears loading when an in-flight path lookup is replaced by a fuzzy query', async () => {
    let resolveSlow!: (value: unknown) => void
    fsMocks.readDesktopDir.mockImplementation(async (path: string) => {
      if (path === '/srv') {
        return { entries: [{ isDirectory: true, name: 'hermes-agent', path: '/srv/hermes-agent' }] }
      }

      return new Promise(resolve => {
        resolveSlow = resolve
      })
    })
    render(<RemoteFolderPicker />)
    await waitFor(() => expect(fsMocks.picker).not.toBeNull())
    act(() => {
      void fsMocks.picker!.selectPaths({ defaultPath: '/srv', directories: true })
    })

    const input = await screen.findByRole('combobox', { name: 'Folder path' })
    await screen.findByText('hermes-agent')
    fireEvent.change(input, { target: { value: '/srv/slow' } })
    await screen.findByText('Loading')
    fireEvent.change(input, { target: { value: 'herm' } })

    await waitFor(() => expect(screen.queryByText('Loading')).toBeNull())
    expect(await screen.findByText('hermes-agent')).toBeTruthy()
    await act(async () => {
      resolveSlow({ entries: [] })
    })
  })
})
