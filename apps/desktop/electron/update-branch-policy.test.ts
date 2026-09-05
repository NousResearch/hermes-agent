import assert from 'node:assert/strict'

import { test } from 'vitest'

import { shouldHealMissingUpdateBranch } from './update-branch-policy'

test('missing custom update branch is retained when it is the active maintained checkout', () => {
  assert.equal(
    shouldHealMissingUpdateBranch({ configuredBranch: 'hermes/live-worktree-upgrade', currentBranch: 'hermes/live-worktree-upgrade' }),
    false
  )
})

test('stale missing update branch heals to main when another branch is active', () => {
  assert.equal(
    shouldHealMissingUpdateBranch({ configuredBranch: 'release/deleted', currentBranch: 'main' }),
    true
  )
})
