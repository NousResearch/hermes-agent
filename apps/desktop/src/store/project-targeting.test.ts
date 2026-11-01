import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $terminals, closeAllTerminals, createTerminal } from '@/app/right-sidebar/terminal/terminals'
import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import type { HermesReviewFile } from '@/global'
import type { SessionInfo } from '@/hermes'

import { refreshRepoStatus } from './coding-status'
import { stampSessionPrBranch } from './pull-requests'
import {
  $reviewFiles,
  $reviewOpen,
  $reviewScopeTarget,
  closeReview,
  confirmRevert,
  createOrOpenPr,
  openReview,
  refreshReview,
  requestRevert,
  stageReviewFile
} from './review'
import {
  $connection,
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  $unlistedSessionOwnerRows,
  $workspaceCwdOwner,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint
} from './session'
import { $focusedWorkspaceCwd } from './session-states'
import {
  activeWorkspaceExecutionOwner,
  focusedWorkspaceExecutionOwner,
  workspaceExecutionOwnerForSession,
  workspaceExecutionOwnerMatchesActiveBackend
} from './workspace-execution-owner'

vi.mock('@/lib/oneshot', () => ({ requestOneShot: vi.fn() }))
vi.mock('./coding-status', () => ({
  refreshRepoStatus: vi.fn(),
  repoStatusForCwd: () => ({ get: () => ({ branch: 'feature' }) })
}))
vi.mock('./pull-requests', () => ({ stampSessionPrBranch: vi.fn() }))

function focus(id: string) {
  noteActiveTreeGroup(id)
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>(res => (resolve = res))

  return { promise, resolve }
}

function bridge(over: { list?: (cwd: string) => Promise<{ files: HermesReviewFile[] }> } = {}) {
  const review = {
    list: vi.fn(async (cwd: string) => ({
      files: [{ path: `${cwd.split('/').pop()}.ts`, status: 'modified', staged: false, added: 1, removed: 0 }]
    })),
    stage: vi.fn(async (): Promise<void> => undefined),
    revert: vi.fn(async () => undefined),
    createPr: vi.fn(async () => ({ url: '' })),
    shipInfo: vi.fn(async () => ({ ghReady: false, pr: null })),
    ...over
  }

  vi.stubGlobal('hermesDesktop', {
    git: { review },
    api: vi.fn(async ({ path, body }: { path: string; body?: { path: string; file?: string } }) => {
      const url = new URL(path, 'http://test.invalid')

      switch (url.pathname) {
        case '/api/git/review/list':
          return review.list(url.searchParams.get('path') ?? '')

        case '/api/git/review/ship-info':
          return review.shipInfo()

        case '/api/git/review/stage':
          return review.stage()

        case '/api/git/review/revert':
          return review.revert()

        default:
          throw new Error(`Unexpected API request: ${url.pathname} ${body?.path ?? ''}`)
      }
    })
  })

  return review
}

beforeEach(() => {
  vi.useFakeTimers()
  setWorkspaceScope('sessions')
  closeReview()
  closeAllTerminals()
  $selectedStoredSessionId.set('primary')
  $workspaceCwdOwner.set('primary')
  $currentCwd.set('/repo/primary')
  $sessions.set([
    { id: 'project-a', cwd: '/repo/a', git_repo_root: '/repo/a' },
    { id: 'project-b', cwd: '/repo/b', git_repo_root: '/repo/b' }
  ] as SessionInfo[])
  setSessionOwnerHint('project-a', { connectionId: 'local', mode: 'local', profile: 'default' })
  setSessionOwnerHint('project-b', { connectionId: 'local', mode: 'local', profile: 'default' })
  $layoutTree.set(
    split('row', [
      group(['workspace'], { active: 'workspace', id: 'primary' }),
      group(['session-tile:project-a'], { active: 'session-tile:project-a', id: 'a' }),
      group(['session-tile:project-b'], { active: 'session-tile:project-b', id: 'b' })
    ])
  )
  focus('a')
})

afterEach(() => {
  closeReview()
  closeAllTerminals()
  noteActiveTreeGroup(null)
  $layoutTree.set(null)
  $selectedStoredSessionId.set(null)
  $workspaceCwdOwner.set(null)
  $currentCwd.set('')
  $sessions.set([])
  $connection.set(null)
  _resetSessionOwnerHintsForTests({ storage: true })
  $unlistedSessionOwnerRows.set([])
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('project targeting follows the focused chat', () => {
  it('moves unpinned review reads and mutations with Files, retaining an explicit review pin', async () => {
    const review = bridge()
    expect($focusedWorkspaceCwd.get()).toBe('/repo/a')
    openReview()
    await refreshReview()
    expect($reviewFiles.get().map(file => file.path)).toEqual(['a.ts'])
    expect($reviewScopeTarget.get()).toBe('tile:project-a')

    focus('b')
    expect($focusedWorkspaceCwd.get()).toBe('/repo/b')
    expect($reviewFiles.get()).toEqual([])
    await vi.advanceTimersByTimeAsync(100)
    expect($reviewFiles.get().map(file => file.path)).toEqual(['b.ts'])
    expect($reviewScopeTarget.get()).toBe('tile:project-b')
    await stageReviewFile('b.ts')
    expect(review.stage).toHaveBeenCalledWith('/repo/b', 'b.ts')

    openReview('/repo/a', 'tile:project-a')
    await refreshReview()
    focus('primary')
    await vi.advanceTimersByTimeAsync(100)
    expect($reviewFiles.get().map(file => file.path)).toEqual(['a.ts'])
    expect($reviewScopeTarget.get()).toBe('tile:project-a')
  })

  it('does not carry a pending destructive confirmation into another project', async () => {
    const review = bridge()
    openReview()
    await refreshReview()
    requestRevert(null)
    focus('b')
    await confirmRevert()
    expect(review.revert).not.toHaveBeenCalled()
  })

  it('creates shells in the focused project without changing existing shell directories', () => {
    bridge()
    const first = createTerminal()
    focus('b')
    const second = createTerminal()
    focus('primary')
    const third = createTerminal()
    expect($terminals.get().map(({ id, cwd }) => ({ id, cwd }))).toEqual([
      { id: first, cwd: '/repo/a' },
      { id: second, cwd: '/repo/b' },
      { id: third, cwd: '/repo/primary' }
    ])
  })

  it('keeps PR completion attached to the session that started it after focus changes', async () => {
    const review = bridge()
    let finish!: (value: { url: string }) => void
    review.createPr.mockImplementation(() => new Promise(resolve => (finish = resolve)))
    // Old/hidden chats need not be on the recent-session page.
    $unlistedSessionOwnerRows.set($sessions.get())
    $sessions.set([])
    openReview()
    const pending = createOrOpenPr()
    expect(review.createPr).toHaveBeenCalledWith('/repo/a')
    focus('b')
    finish({ url: '' })
    await pending
    expect(stampSessionPrBranch).toHaveBeenCalledWith('project-a', '/repo/a', 'feature')
  })

  it('refuses Git and terminal dispatch when a focused tile belongs to another gateway, even at the same path', async () => {
    _resetSessionOwnerHintsForTests({ storage: true })
    const review = bridge()
    const repo = '/workspace/shared-name'
    $connection.set({ connectionId: 'gateway-a', mode: 'remote', profile: 'default', remoteKind: 'url', baseUrl: 'https://gateway-a.test' } as never)
    $sessions.set([
      { id: 'project-a', cwd: repo, git_repo_root: repo },
      { id: 'project-b', cwd: repo, git_repo_root: repo }
    ] as SessionInfo[])
    $currentCwd.set(repo)
    setSessionOwnerHint('project-a', { connectionId: 'gateway-a', mode: 'remote', profile: 'default' })
    setSessionOwnerHint('project-b', { connectionId: 'gateway-b', mode: 'remote', profile: 'default' })
    focus('a')
    expect(activeWorkspaceExecutionOwner()).toMatchObject({ connectionId: 'gateway-a', mode: 'remote', profile: 'default' })
    expect(workspaceExecutionOwnerForSession('project-a')).toMatchObject({
      connectionId: 'gateway-a',
      mode: 'remote',
      profile: 'default'
    })
    expect(focusedWorkspaceExecutionOwner()).toMatchObject({ connectionId: 'gateway-a', mode: 'remote' })
    $reviewOpen.set(true)
    await refreshReview()
    expect(review.list).toHaveBeenCalledTimes(1)
    requestRevert(null)
    const staged = deferred<void>()
    review.stage.mockImplementationOnce(() => staged.promise)
    vi.mocked(refreshRepoStatus).mockClear()
    const pendingStage = stageReviewFile(null)

    focus('b')
    staged.resolve()
    await pendingStage
    expect(refreshRepoStatus).not.toHaveBeenCalled()
    expect($reviewFiles.get()).toEqual([])
    const terminalId = createTerminal()
    const terminal = $terminals.get().find(entry => entry.id === terminalId)!

    expect(terminal.executionOwner).toMatchObject({ connectionId: 'gateway-b', mode: 'remote', profile: 'default' })
    expect(workspaceExecutionOwnerMatchesActiveBackend(terminal.executionOwner)).toBe(false)

    await confirmRevert()
    await stageReviewFile(null)
    expect(review.revert).not.toHaveBeenCalled()
    expect(review.stage).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(100)
    expect(review.list).toHaveBeenCalledTimes(1)

    $connection.set({ connectionId: 'gateway-b', mode: 'remote', profile: 'default', remoteKind: 'url', baseUrl: 'https://gateway-b.test' } as never)
    // A terminal created before its backend was known cannot acquire authority
    // just because the selected connection later changes. Open one on its owner.
    expect(workspaceExecutionOwnerMatchesActiveBackend(terminal.executionOwner)).toBe(false)
    const ownedTerminal = createTerminal()
    expect(workspaceExecutionOwnerMatchesActiveBackend($terminals.get().find(t => t.id === ownedTerminal)!.executionOwner)).toBe(true)
  })

  it('rejects stale same-path Review results and another profile on the same remote gateway', async () => {
    _resetSessionOwnerHintsForTests({ storage: true })
    const result = deferred<{ files: Array<{ path: string; status: string; staged: boolean; added: number; removed: number }> }>()
    const review = bridge({ list: vi.fn(() => result.promise) })
    const repo = '/workspace/same'
    $connection.set({ connectionId: 'gateway-a', mode: 'remote', profile: 'default', remoteKind: 'url', baseUrl: 'https://gateway-a.test' } as never)
    $sessions.set([
      { id: 'project-a', cwd: repo, git_repo_root: repo },
      { id: 'project-b', cwd: repo, git_repo_root: repo }
    ] as SessionInfo[])
    $currentCwd.set(repo)
    setSessionOwnerHint('project-a', { connectionId: 'gateway-a', mode: 'remote', profile: 'default' })
    setSessionOwnerHint('project-b', { connectionId: 'gateway-a', mode: 'remote', profile: 'worker' })
    focus('a')
    $reviewOpen.set(true)
    const pending = refreshReview()

    focus('b')
    result.resolve({ files: [{ path: 'from-default.ts', status: 'modified', staged: false, added: 1, removed: 0 }] })
    await pending
    await vi.advanceTimersByTimeAsync(100)

    expect($reviewFiles.get()).toEqual([])
    expect(review.list).toHaveBeenCalledTimes(1)
    expect(workspaceExecutionOwnerMatchesActiveBackend(workspaceExecutionOwnerForSession('project-b'))).toBe(false)
  })

  it.each([undefined, 'gateway-a'])('does not reuse a saved shell when SSH configuration changes behind %s', connectionId => {
    focus('primary')

    const connection = {
      connectionId, mode: 'remote', profile: 'default', remoteKind: 'ssh', remoteIdentity: 'same-desktop-ownership-id',
      ssh: { effectiveConfigFingerprint: 'host-a', remoteHermesPath: '/opt/hermes', remoteProfile: '' }
    }

    $connection.set(connection as never)
    const id = createTerminal('/workspace/same')
    const owner = $terminals.get().find(t => t.id === id)!.executionOwner
    expect(workspaceExecutionOwnerMatchesActiveBackend(owner)).toBe(true)

    $connection.set({ ...connection, ssh: { ...connection.ssh, effectiveConfigFingerprint: 'host-b' } } as never)
    expect(workspaceExecutionOwnerMatchesActiveBackend(owner)).toBe(false)
    $connection.set({ ...connection, ssh: { ...connection.ssh, remoteHermesPath: '/other/hermes' } } as never)
    expect(workspaceExecutionOwnerMatchesActiveBackend(owner)).toBe(false)
    $connection.set(connection as never)
    expect(workspaceExecutionOwnerMatchesActiveBackend(owner)).toBe(true)
  })
})
