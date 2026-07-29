import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $currentBranch,
  $currentCwd,
  $newChatWorkspaceTarget,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'

import { newSessionOpensTab, startWorkspaceSession } from './workspace-session-target'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

/**
 * The sidebar "+" is a create affordance, not a discard one: with a chat
 * already loaded it must stack a tab (⌘T) rather than replace the surface
 * (⌘N), which would throw away a conversation that may still be mid-turn.
 */
describe('newSessionOpensTab', () => {
  it('opens a tab once a conversation is on screen', () => {
    expect(newSessionOpensTab('runtime-a', 'stored-a')).toBe(true)
  })

  it('opens a tab for a live runtime whose stored id has not landed yet', () => {
    expect(newSessionOpensTab('runtime-a', null)).toBe(true)
  })

  it('opens a tab for a selected session still resuming into a runtime', () => {
    expect(newSessionOpensTab(null, 'stored-a')).toBe(true)
  })

  it('replaces the empty surface when nothing is open', () => {
    expect(newSessionOpensTab(null, null)).toBe(false)
  })
})

describe('startWorkspaceSession', () => {
  afterEach(() => {
    setCurrentBranch('')
    setCurrentCwd('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  it('keeps a newer sidebar target when an older project lookup resolves', async () => {
    const first = deferred<{ branch?: string; cwd?: string }>()
    const second = deferred<{ branch?: string; cwd?: string }>()

    const requestGateway = vi
      .fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)

    const activeSessionIdRef = { current: null }

    const startFreshSessionDraft = vi.fn((options?: { workspaceTarget: string }) => {
      setNewChatWorkspaceTarget(options?.workspaceTarget)
      setCurrentCwd(options?.workspaceTarget || '')
    })

    const followActiveSessionCwd = vi.fn()

    startWorkspaceSession({
      activeSessionIdRef,
      followActiveSessionCwd,
      path: '/workspace-a',
      requestGateway,
      startFreshSessionDraft
    })
    startWorkspaceSession({
      activeSessionIdRef,
      followActiveSessionCwd,
      path: '/workspace-b',
      requestGateway,
      startFreshSessionDraft
    })

    first.resolve({ branch: 'stale', cwd: '/normalized-a' })
    await first.promise
    await Promise.resolve()

    expect($newChatWorkspaceTarget.get()).toBe('/workspace-b')
    expect($currentCwd.get()).toBe('/workspace-b')
    expect($currentBranch.get()).not.toBe('stale')

    second.resolve({ branch: 'main', cwd: '/normalized-b' })
    await second.promise
    await Promise.resolve()

    expect($newChatWorkspaceTarget.get()).toBe('/normalized-b')
    expect($currentCwd.get()).toBe('/normalized-b')
    expect($currentBranch.get()).toBe('main')
  })
})
