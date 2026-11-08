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

import { deferred } from '../../../test/deferred'

import { useCwdActions } from './use-cwd-actions'

type CwdActionsHandle = ReturnType<typeof useCwdActions>

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
    activeSessionIdRef,
    requestGateway
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

describe('useCwdActions session target', () => {
  afterEach(() => {
    cleanup()
    setCurrentCwd('')
    setCurrentBranch('')
  })

  it('re-homes the named tile, not the primary, and leaves the primary readout alone', async () => {
    const requestGateway = vi.fn(async () => ({ branch: 'dev', cwd: '/opt/data/profiles/austin' }) as never)
    const onSessionRuntimeInfo = vi.fn()
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'primary-rt' }
    let handle: CwdActionsHandle | null = null
    setCurrentCwd('/primary-workspace')

    function TargetHarness() {
      const actions = useCwdActions({ activeSessionIdRef, onSessionRuntimeInfo, requestGateway })

      useEffect(() => {
        handle = actions
      }, [actions])

      return null
    }

    render(<TargetHarness />)
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.changeSessionCwd('/opt/data/profiles/austin', 'tile-rt')
    })

    expect(requestGateway).toHaveBeenCalledWith('session.cwd.set', {
      cwd: '/opt/data/profiles/austin',
      session_id: 'tile-rt'
    })
    expect(onSessionRuntimeInfo).toHaveBeenCalledWith('tile-rt', { branch: 'dev', cwd: '/opt/data/profiles/austin' })
    expect($currentCwd.get()).toBe('/primary-workspace')

    await act(async () => {
      await handle!.changeSessionCwd('/opt/data/profiles/austin')
    })

    expect(requestGateway).toHaveBeenLastCalledWith('session.cwd.set', {
      cwd: '/opt/data/profiles/austin',
      session_id: 'primary-rt'
    })
    expect($currentCwd.get()).toBe('/opt/data/profiles/austin')
  })
})

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
