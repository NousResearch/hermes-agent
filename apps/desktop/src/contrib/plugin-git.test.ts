import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from './plugin'

const worktrees = [
  { branch: 'main', detached: false, isMain: true, locked: false, path: '/srv/app' },
  { branch: 'feature', detached: false, isMain: false, locked: false, path: '/srv/app/.worktrees/feature' }
]

const door = {
  worktreeAdd: vi.fn(async () => ({ branch: 'feature', path: '/srv/app/.worktrees/feature', repoRoot: '/srv/app' })),
  worktreeList: vi.fn(async () => worktrees),
  worktreeRemove: vi.fn(async () => ({ removed: '/srv/app/.worktrees/feature' }))
}

beforeEach(() => {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { git: door }
})

afterEach(() => {
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ctx.git', () => {
  it('lists the worktrees of the path it was given', async () => {
    const result = await createPluginContext('board').git.worktreeList('/srv/app')

    expect(result).toEqual({ ok: true, worktrees })
    expect(door.worktreeList).toHaveBeenCalledWith('/srv/app')
  })

  it('passes the add options through and returns where git landed', async () => {
    const result = await createPluginContext('board').git.worktreeAdd('/srv/app', {
      base: 'origin/feature',
      branch: 'feature',
      name: 'feature'
    })

    expect(result).toEqual({
      branch: 'feature',
      ok: true,
      path: '/srv/app/.worktrees/feature',
      repoRoot: '/srv/app'
    })
    expect(door.worktreeAdd).toHaveBeenCalledWith('/srv/app', {
      base: 'origin/feature',
      branch: 'feature',
      name: 'feature'
    })
  })

  it('removes by path and forwards the force flag', async () => {
    const result = await createPluginContext('board').git.worktreeRemove('/srv/app', '/srv/app/.worktrees/x', {
      force: true
    })

    expect(result).toEqual({ ok: true, removed: '/srv/app/.worktrees/feature' })
    expect(door.worktreeRemove).toHaveBeenCalledWith('/srv/app', '/srv/app/.worktrees/x', { force: true })
  })

  it('answers unavailable on a shell without the door, and never throws', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    const git = createPluginContext('board').git

    expect(await git.worktreeList('/srv/app')).toEqual({
      error: 'unavailable',
      message: 'This Desktop build has no git door for plugins.',
      ok: false
    })
    expect((await git.worktreeAdd('/srv/app')).ok).toBe(false)
    expect((await git.worktreeRemove('/srv/app', '/srv/app/.worktrees/x')).ok).toBe(false)
    expect(door.worktreeList).not.toHaveBeenCalled()
  })

  it('turns a refused path into the sentence git gave, not a throw', async () => {
    door.worktreeAdd.mockRejectedValueOnce(new Error('Branch name is required.'))

    const result = await createPluginContext('board').git.worktreeAdd('/srv/app', { name: 'x' })

    expect(result).toEqual({ error: 'failed', message: 'Branch name is required.', ok: false })
  })

  it('reads a malformed list as empty rather than handing a plugin a non-array', async () => {
    door.worktreeList.mockResolvedValueOnce(null as never)

    expect(await createPluginContext('board').git.worktreeList('/srv/app')).toEqual({
      ok: true,
      worktrees: []
    })
  })
})
