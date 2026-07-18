import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesRepoStatus } from '@/global'

import { $repoStatus, $repoStatusLoading, refreshRepoStatus } from './coding-status'
import { $currentCwd } from './session'

const sampleStatus: HermesRepoStatus = {
  branch: 'feature/login',
  defaultBranch: 'main',
  detached: false,
  ahead: 1,
  behind: 0,
  staged: 1,
  unstaged: 2,
  untracked: 0,
  conflicted: 0,
  changed: 3,
  added: 12,
  removed: 4,
  files: []
}

function stubProbe(impl: (cwd: string) => Promise<HermesRepoStatus | null>) {
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { git: { repoStatus: impl } }
}

describe('refreshRepoStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $repoStatus.set(null)
    $currentCwd.set('')
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('populates $repoStatus from the probe for an explicit cwd', async () => {
    stubProbe(async () => sampleStatus)
    await refreshRepoStatus('/repo')
    expect($repoStatus.get()).toEqual(sampleStatus)
  })

  it('falls back to the active session cwd when none is passed', async () => {
    const probe = vi.fn(async () => sampleStatus)
    stubProbe(probe)
    $currentCwd.set('/active/repo')
    await refreshRepoStatus()
    expect(probe).toHaveBeenCalledWith('/active/repo')
  })

  it('clears status when there is no cwd', async () => {
    stubProbe(async () => sampleStatus)
    $repoStatus.set(sampleStatus)
    await refreshRepoStatus('   ')
    expect($repoStatus.get()).toBeNull()
  })

  it('clears status when the probe is unavailable (remote backend)', async () => {
    $repoStatus.set(sampleStatus)
    await refreshRepoStatus('/repo')
    expect($repoStatus.get()).toBeNull()
  })

  it('clears status when the probe throws', async () => {
    stubProbe(async () => {
      throw new Error('not a repo')
    })
    $repoStatus.set(sampleStatus)
    await refreshRepoStatus('/repo')
    expect($repoStatus.get()).toBeNull()
  })

  it('runs one probe at a time and coalesces overlap into one trailing refresh', async () => {
    const resolvers: Array<(status: HermesRepoStatus | null) => void> = []
    const calls: string[] = []
    let active = 0
    let maxActive = 0

    stubProbe(
      cwd =>
        new Promise(resolve => {
          calls.push(cwd)
          active++
          maxActive = Math.max(maxActive, active)
          resolvers.push(status => {
            active--
            resolve(status)
          })
        })
    )

    const first = refreshRepoStatus('/repo-a')
    const second = refreshRepoStatus('/repo-b')
    const third = refreshRepoStatus('/repo-c')

    expect(calls).toEqual(['/repo-a'])
    expect(maxActive).toBe(1)
    expect($repoStatusLoading.get()).toBe(true)

    resolvers.shift()?.(sampleStatus)
    await Promise.resolve()
    await Promise.resolve()

    expect(calls).toEqual(['/repo-a', '/repo-c'])
    expect(maxActive).toBe(1)
    expect($repoStatus.get()).toBeNull()

    resolvers.shift()?.(sampleStatus)
    await Promise.all([first, second, third])

    expect(maxActive).toBe(1)
    expect($repoStatus.get()).toEqual(sampleStatus)
    expect($repoStatusLoading.get()).toBe(false)
  })
})
