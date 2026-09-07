/**
 * Tests for electron/healed-branch.ts — the self-heal decision that keeps
 * never-pushed local branches pinned instead of silently re-pinning the
 * desktop updater to main.
 *
 * The heal path exists for merged-and-deleted upstream branches (bb/gui):
 * ls-remote exit 2 is treated as "branch gone" and the pin flips to main.
 * That inference is wrong for a branch that was never pushed — the local ref
 * is then the only copy of the user's commits, and re-pinning moves the
 * running code off them without any visible error.
 */

import { describe, expect, it } from 'vitest'

import { decideHealedBranch } from './healed-branch'

describe('decideHealedBranch', () => {
  it('keeps a never-pushed local branch pinned', () => {
    const decision = decideHealedBranch('hermes-local', {
      remoteTrackingRefExists: false,
      commitsBeyondMain: 3
    })

    expect(decision.branch).toBe('hermes-local')
    expect(decision.reason).toBe('kept-local-branch')
  })

  it('keeps a pushed branch that carries commits beyond main', () => {
    const decision = decideHealedBranch('feature/with-work', {
      remoteTrackingRefExists: true,
      commitsBeyondMain: 1
    })

    expect(decision.branch).toBe('feature/with-work')
    expect(decision.reason).toBe('kept-local-branch')
  })

  it('keeps the pin when commits are unknown but the branch was never pushed', () => {
    const decision = decideHealedBranch('hermes-local', {
      remoteTrackingRefExists: false,
      commitsBeyondMain: null
    })

    expect(decision.branch).toBe('hermes-local')
    expect(decision.reason).toBe('kept-local-branch')
  })

  it('heals a fully merged branch even after fetch --prune removed the tracking ref', () => {
    const decision = decideHealedBranch('bb/gui', {
      remoteTrackingRefExists: false,
      commitsBeyondMain: 0
    })

    expect(decision.branch).toBe('main')
    expect(decision.reason).toBe('healed-to-main')
  })

  it('heals a deleted-after-merge branch with nothing beyond main', () => {
    const decision = decideHealedBranch('bb/gui', {
      remoteTrackingRefExists: true,
      commitsBeyondMain: 0
    })

    expect(decision.branch).toBe('main')
    expect(decision.reason).toBe('healed-to-main')
  })

  it('heals when the commit count is unavailable but the branch was pushed', () => {
    const decision = decideHealedBranch('bb/gui', {
      remoteTrackingRefExists: true,
      commitsBeyondMain: null
    })

    expect(decision.branch).toBe('main')
    expect(decision.reason).toBe('healed-to-main')
  })
})
