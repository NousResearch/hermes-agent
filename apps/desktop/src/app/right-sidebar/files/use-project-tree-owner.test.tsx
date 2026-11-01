/// <reference types="node" />

import '@/store/session-states'

import { Buffer } from 'node:buffer'

import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type { HermesReadDirEntry } from '@/global'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { $showIgnoredRoots } from './prefs'
import { RemoteFolderPicker } from './remote-picker'
import { resetProjectTreeState, useProjectTree } from './use-project-tree'

// One gateway serving several profiles, each with its own SSH identity: a profile reads
// its own workspace and nothing else. The window stays on `default` throughout.
const CONNECTION = 'lab'
const SESSION_OWNERS: Record<string, string> = { 'ops-chat': 'ops', 'qa-chat': 'qa' }

const entry = (path: string, isDirectory = false): HermesReadDirEntry => ({
  isDirectory,
  name: path.slice(path.lastIndexOf('/') + 1),
  path
})

// What each identity can see. Both Docker-backed Bots call their workspace `/workspace`.
const FILES: Record<string, Record<string, HermesReadDirEntry[]>> = {
  ops: {
    '/workspace': [
      entry('/workspace/.gitignore'),
      entry('/workspace/build', true),
      entry('/workspace/src', true),
      entry('/workspace/ops.md')
    ]
  },
  qa: { '/workspace': [entry('/workspace/qa.md')] }
}

const GITIGNORE: Record<string, string> = { ops: 'build/\n' }

interface ApiRequest {
  body?: Record<string, unknown>
  method?: string
  path: string
  profile?: string
}

const api = vi.fn(async ({ body, path, profile }: ApiRequest) => {
  const identity = profile ?? 'default'
  const url = new URL(path, 'http://gateway')
  const target = url.searchParams.get('path') ?? ''
  const files = FILES[identity] ?? {}

  switch (url.pathname) {
    case '/api/fs/list':
      // Like the route: an unreadable folder is HTTP 200 with the errno in the JSON.
      return files[target] ? { entries: files[target] } : { entries: [], error: 'EACCES' }

    case '/api/fs/git-root':
      // The SSH backend's walk finds no readable .git and answers null, never an error.
      return { root: GITIGNORE[identity] && target.startsWith('/workspace') ? '/workspace' : null }

    case '/api/fs/read-data-url':
      if (target === '/workspace/.gitignore' && GITIGNORE[identity]) {
        return { dataUrl: `data:text/plain;base64,${Buffer.from(GITIGNORE[identity]).toString('base64')}` }
      }

      throw new Error('404: File not found')

    case '/api/files/mkdir':
      files[String(body?.path)] = []

      return { path: body?.path }

    default:
      throw new Error(`unexpected ${path}`)
  }
})

const profilesAsked = () => [...new Set(api.mock.calls.map(([request]) => request.profile ?? 'default'))]

beforeEach(() => {
  api.mockClear()
  resetProjectTreeState()
  $showIgnoredRoots.set([])
  _resetSessionOwnerHintsForTests()
  setApiRequestConnection(CONNECTION)

  for (const [sessionId, profile] of Object.entries(SESSION_OWNERS)) {
    setSessionOwnerHint(sessionId, { connectionId: CONNECTION, profile })
  }

  $connection.set({ connectionId: CONNECTION, mode: 'remote', profile: 'default' } as never)
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { api, readDir: vi.fn() }
})

afterEach(() => {
  cleanup()
  resetProjectTreeState()
  _resetSessionOwnerHintsForTests({ storage: true })
  setApiRequestConnection(null)
  $connection.set(null)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe("a Bot chat's workspace on a window focused on another profile", () => {
  it('lists the tree and applies its .gitignore as the Bot that owns the chat', async () => {
    const { result } = renderHook(() => useProjectTree('/workspace', 'ops-chat'))

    await waitFor(() => expect(result.current.rootLoading).toBe(false))

    expect(result.current.rootError).toBeNull()
    expect(result.current.data.map(node => node.name)).toEqual(['.gitignore', 'src', 'ops.md'])
    expect(profilesAsked()).toEqual(['ops'])
  })

  it('re-reads the same path for each owner as focus moves A → B → A', async () => {
    const { rerender, result } = renderHook(({ sessionId }) => useProjectTree('/workspace', sessionId), {
      initialProps: { sessionId: 'ops-chat' }
    })

    await waitFor(() => expect(result.current.data.map(node => node.name)).toContain('ops.md'))

    act(() => rerender({ sessionId: 'qa-chat' }))
    await waitFor(() => expect(result.current.data.map(node => node.name)).toEqual(['qa.md']))

    act(() => rerender({ sessionId: 'ops-chat' }))
    await waitFor(() => expect(result.current.data.map(node => node.name)).toContain('ops.md'))

    expect(result.current.rootError).toBeNull()
    expect(profilesAsked().sort()).toEqual(['ops', 'qa'])
  })

  it('browses the folder picker, and creates its folders, as the Bot that owns the chat', async () => {
    render(<RemoteFolderPicker />)
    let selection: Promise<string[]> = Promise.resolve([])

    act(() => {
      selection = selectDesktopPaths({ defaultPath: '/workspace', directories: true }, { sessionId: 'ops-chat' })
    })

    await screen.findByText('src')
    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    const input = screen.getByRole('textbox', { name: 'Folder name' })
    fireEvent.change(input, { target: { value: 'notes' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Folder name' })).toBeNull())
    fireEvent.click(screen.getByRole('button', { name: 'Select folder' }))

    await expect(selection).resolves.toEqual(['/workspace/notes'])
    expect(profilesAsked()).toEqual(['ops'])
  })
})
