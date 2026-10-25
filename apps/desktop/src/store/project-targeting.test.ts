import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $terminals, closeAllTerminals, createTerminal } from '@/app/right-sidebar/terminal/terminals'
import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import type { SessionInfo } from '@/hermes'

import { stampSessionPrBranch } from './pull-requests'
import {
  $reviewFiles,
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
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  $unlistedSessionOwnerRows,
  $workspaceCwdOwner
} from './session'
import { $focusedWorkspaceCwd } from './session-states'

vi.mock('@/lib/oneshot', () => ({ requestOneShot: vi.fn() }))
vi.mock('./coding-status', () => ({
  refreshRepoStatus: vi.fn(),
  repoStatusForCwd: () => ({ get: () => ({ branch: 'feature' }) })
}))
vi.mock('./pull-requests', () => ({ stampSessionPrBranch: vi.fn() }))

function focus(id: string) {
  noteActiveTreeGroup(id)
}

function bridge() {
  const review = {
    list: vi.fn(async (cwd: string) => ({
      files: [{ path: `${cwd.split('/').pop()}.ts`, status: 'modified', staged: false, added: 1, removed: 0 }]
    })),
    stage: vi.fn(async () => undefined),
    revert: vi.fn(async () => undefined),
    createPr: vi.fn(async () => ({ url: '' })),
    shipInfo: vi.fn(async () => ({ ghReady: false, pr: null }))
  }

  vi.stubGlobal('hermesDesktop', { git: { review } })

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
})
