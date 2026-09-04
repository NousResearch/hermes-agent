/**
 * Tests for electron/update-root-policy.ts — the pure classifier that decides
 * whether the desktop's git-based self-update may run against a resolved
 * update root. A checkout the install contract does not manage (steward-owned
 * `external`/`electron-updater` mechanisms) must be refused with a user-action
 * pointer instead of the desktop pulling into it.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyUpdateRoot } from './update-root-policy'

test('a non-git root is not updatable and carries no user advice', () => {
  const result = classifyUpdateRoot({ isGitTree: false, updateMechanism: null })

  assert.equal(result.updatable, false)
  assert.equal(result.verdict, 'not-a-checkout')
  assert.equal(result.provenance, 'not-a-checkout')
  assert.equal(result.advice, null)
  assert.ok(result.message)
})

test('a self-managed checkout is updatable', () => {
  const result = classifyUpdateRoot({ isGitTree: true, updateMechanism: 'self' })

  assert.equal(result.updatable, true)
  assert.equal(result.verdict, 'updatable')
  assert.equal(result.provenance, 'managed-self')
  assert.equal(result.advice, null)
  assert.equal(result.message, null)
})

test('a checkout owned by the external steward is refused with a git pull pointer', () => {
  const result = classifyUpdateRoot({ isGitTree: true, updateMechanism: 'external' })

  assert.equal(result.updatable, false)
  assert.equal(result.verdict, 'steward-owned-git-tree')
  assert.equal(result.provenance, 'steward-owned')
  assert.equal(result.advice, 'git pull')
  assert.ok(result.message?.includes('external'))
})

test('a checkout owned by electron-updater is refused the same way', () => {
  const result = classifyUpdateRoot({ isGitTree: true, updateMechanism: 'electron-updater' })

  assert.equal(result.updatable, false)
  assert.equal(result.verdict, 'steward-owned-git-tree')
  assert.equal(result.advice, 'git pull')
  assert.ok(result.message?.includes('electron-updater'))
})

test('an unstamped git tree (dev checkout) stays updatable with unknown provenance', () => {
  const result = classifyUpdateRoot({ isGitTree: true, updateMechanism: null })

  assert.equal(result.updatable, true)
  assert.equal(result.verdict, 'updatable')
  assert.equal(result.provenance, 'unknown')
  assert.equal(result.advice, null)
})

test('the classification is a pure function of its inputs', () => {
  const facts = { isGitTree: true, updateMechanism: 'external' as const }
  assert.deepEqual(classifyUpdateRoot(facts), classifyUpdateRoot({ ...facts }))
})
