import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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

  it('ignores stale draft cwd normalization after a newer no-workspace target wins', async () => {
    const projectInfo = deferred<{ branch?: string; cwd?: string }>()
    const requestGateway = vi.fn(async () => projectInfo.promise as never)
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let handle: CwdActionsHandle | null = null

    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
      />
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
