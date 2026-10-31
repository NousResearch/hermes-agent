import { describe, expect, it } from 'vitest'

import type { DesktopSyncReceipt } from '@/global'

import { deriveStashOutcome, deriveSyncStatusSummary } from './sync-status'

const stashReceipt = (detail: string): DesktopSyncReceipt => ({
  pm_steps: [{ name: 'local_changes_stash', ok: !detail.startsWith('parked'), detail }]
})

describe('deriveStashOutcome', () => {
  it('returns null when the update never touched local changes', () => {
    expect(deriveStashOutcome(null)).toBeNull()
    expect(deriveStashOutcome({})).toBeNull()
    expect(deriveStashOutcome({ plugin_checks: [] })).toBeNull()
    expect(deriveStashOutcome({ pm_steps: [{ name: 'sync', ok: true, detail: 'in sync' }] })).toBeNull()
  })

  it('parses the parked disposition and points at the stash entry', () => {
    const outcome = deriveStashOutcome(stashReceipt('parked: abc1234 (--keep-stash)'))

    expect(outcome?.disposition).toBe('parked')
    expect(outcome?.ref).toBe('abc1234')
    expect(outcome?.reason).toBe('--keep-stash')
    expect(outcome?.level).toBe('warn')
    expect(outcome?.detail).toContain('git stash apply abc1234')
  })

  it('warns on the conflict variant of parked', () => {
    const outcome = deriveStashOutcome(stashReceipt('parked: deadbeef (restore hit conflicts)'))

    expect(outcome?.level).toBe('warn')
    expect(outcome?.headline).toContain('not re-applied')
  })

  it('treats a restored stash as the happy path', () => {
    const outcome = deriveStashOutcome(stashReceipt('restored: cafe123'))

    expect(outcome?.disposition).toBe('restored')
    expect(outcome?.level).toBe('ok')
  })

  it('warns when the configured discard dropped the changes', () => {
    const outcome = deriveStashOutcome(stashReceipt('discarded: f00d000'))

    expect(outcome?.disposition).toBe('discarded')
    expect(outcome?.level).toBe('warn')
    expect(outcome?.headline).toContain('discarded')
  })

  it('ignores a stash step whose detail does not match the producer contract', () => {
    expect(deriveStashOutcome(stashReceipt(''))).toBeNull()
    expect(deriveStashOutcome(stashReceipt('mystery outcome'))).toBeNull()
  })
})

describe('deriveSyncStatusSummary stash ranking', () => {
  it('surfaces a parked stash above plugin warnings, below a failed rebuild', () => {
    const parked: DesktopSyncReceipt = {
      plugin_checks: [{ name: 'bad-url', needs_fixing: 'update_url points at a fork' }],
      pm_steps: [{ name: 'local_changes_stash', ok: false, detail: 'parked: abc1234 (--keep-stash)' }]
    }

    const stashOnly = deriveSyncStatusSummary(parked)

    expect(stashOnly.headline).toContain('stashed')
    expect(stashOnly.level).toBe('warn')
    expect(stashOnly.stash?.ref).toBe('abc1234')

    const failedRebuild: DesktopSyncReceipt = {
      ...parked,
      venv_rebuild: { ok: false, reason: 'uv sync exited 1' }
    }

    expect(deriveSyncStatusSummary(failedRebuild).headline).toContain('rebuild failed')
  })
})
