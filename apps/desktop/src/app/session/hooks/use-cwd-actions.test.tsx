import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ensureProjectForFolder, pickProjectFolder } from '@/store/projects'
import {
  $currentBranch,
  $currentCwd,
  $newChatWorkspaceTarget,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentCwdTransient,
  setNewChatWorkspaceTarget
} from '@/store/session'

import { useCwdActions } from './use-cwd-actions'

vi.mock('@/store/projects', () => ({
  ensureProjectForFolder: vi.fn(),
  pickProjectFolder: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const ensureProjectForFolderMock = vi.mocked(ensureProjectForFolder)
const pickProjectFolderMock = vi.mocked(pickProjectFolder)
const notifications = await import('@/store/notifications')
const notifyErrorMock = vi.mocked(notifications.notifyError)

type CwdActionsHandle = ReturnType<typeof useCwdActions>

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function Harness({
  activeSessionIdRef,
  onReady,
  requestGateway
}: {
  activeSessionIdRef: MutableRefObject<string | null>
  onReady: (handle: CwdActionsHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const actions = useCwdActions({
    activeSessionId: activeSessionIdRef.current,
    activeSessionIdRef,
    requestGateway
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

describe('useCwdActions draft workspace target', () => {
  beforeEach(() => {
    ensureProjectForFolderMock.mockReset()
    pickProjectFolderMock.mockReset()
    notifyErrorMock.mockClear()
    setCurrentCwd('')
    setCurrentBranch('')
    setNewChatWorkspaceTarget(undefined)
  })

  afterEach(() => {
    cleanup()
    setCurrentCwd('')
    setCurrentBranch('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  it('resolves a Project before staging a folder as a new-chat workspace', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ branch: 'main', cwd: '/workspace/selected-folder' })
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null
    ensureProjectForFolderMock.mockResolvedValue('/workspace/selected-folder')

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.startProjectFromFolder('/picked/folder')
    })

    expect(ensureProjectForFolderMock).toHaveBeenCalledWith('/picked/folder', expect.any(Function))
    expect(requestGateway).toHaveBeenCalledWith('config.get', { cwd: '/workspace/selected-folder', key: 'project' })
    expect($newChatWorkspaceTarget.get()).toBe('/workspace/selected-folder')
    expect($currentCwd.get()).toBe('/workspace/selected-folder')
  })

  it('routes the folder picker selection through the Project resolver', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ branch: 'main', cwd: '/workspace/selected-folder' })
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null
    pickProjectFolderMock.mockResolvedValue('/picked/folder')
    ensureProjectForFolderMock.mockResolvedValue('/workspace/selected-folder')

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.chooseProjectFolder()
    })

    expect(pickProjectFolderMock).toHaveBeenCalledOnce()
    expect(ensureProjectForFolderMock).toHaveBeenCalledWith('/picked/folder', expect.any(Function))
    expect($currentCwd.get()).toBe('/workspace/selected-folder')
  })

  it('surfaces folder-picker failures without attempting Project resolution', async () => {
    const requestGateway = vi.fn()
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null
    pickProjectFolderMock.mockRejectedValue(new Error('picker unavailable'))

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.chooseProjectFolder()
    })

    expect(ensureProjectForFolderMock).not.toHaveBeenCalled()
    expect(notifyErrorMock).toHaveBeenCalled()
  })

  it('ignores a repeated folder action while the picker is already open', async () => {
    const folder = deferred<null | string>()
    const requestGateway = vi.fn()
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null
    pickProjectFolderMock.mockReturnValue(folder.promise)

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const first = handle!.chooseProjectFolder()
    const repeated = handle!.chooseProjectFolder()

    expect(pickProjectFolderMock).toHaveBeenCalledOnce()

    folder.resolve(null)
    await Promise.all([first, repeated])
  })

  it('does not touch the workspace and surfaces an error when the Project resolve fails', async () => {
    const requestGateway = vi.fn()
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null
    ensureProjectForFolderMock.mockRejectedValue(new Error('boom'))

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.startProjectFromFolder('/picked/folder')
    })

    expect(requestGateway).not.toHaveBeenCalled()
    expect($newChatWorkspaceTarget.get()).toBeUndefined()
    expect($currentCwd.get()).toBe('')
    expect(notifyErrorMock).toHaveBeenCalled()
  })

  it('does not create or enter a Project when a session is already active on entry', async () => {
    const requestGateway = vi.fn()
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'active-session' }
    let handle: CwdActionsHandle | null = null

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.startProjectFromFolder('/picked/folder')
    })

    expect(ensureProjectForFolderMock).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('does not mutate cwd when a session starts while the folder action is pending', async () => {
    const folder = deferred<string>()
    const requestGateway = vi.fn()
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null
    ensureProjectForFolderMock.mockReturnValue(folder.promise)

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const starting = handle!.startProjectFromFolder('/picked/folder')
    activeSessionIdRef.current = 'active-session'
    folder.resolve('/workspace/selected-folder')
    await starting

    expect(requestGateway).not.toHaveBeenCalled()
    expect($newChatWorkspaceTarget.get()).toBeUndefined()
    expect($currentCwd.get()).toBe('')
  })

  it('ignores stale draft cwd normalization after a newer no-workspace target wins', async () => {
    const projectInfo = deferred<{ branch?: string; cwd?: string }>()
    const requestGateway = vi.fn(async () => projectInfo.promise as never)
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null

    render(
      <Harness activeSessionIdRef={activeSessionIdRef} onReady={h => (handle = h)} requestGateway={requestGateway} />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    let pendingChange!: Promise<void>

    await act(async () => {
      pendingChange = handle!.changeSessionCwd('/stale-workspace')
    })

    expect($newChatWorkspaceTarget.get()).toBe('/stale-workspace')

    setNewChatWorkspaceTarget(null)
    setCurrentCwdTransient('')
    projectInfo.resolve({ branch: 'main', cwd: '/normalized-stale-workspace' })

    await act(async () => {
      await pendingChange
    })

    expect($newChatWorkspaceTarget.get()).toBeNull()
    expect($currentCwd.get()).toBe('')
    expect($currentBranch.get()).toBe('')
  })
})
