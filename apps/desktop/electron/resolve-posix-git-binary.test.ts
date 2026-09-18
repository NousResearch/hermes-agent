import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolvePosixGitBinary } from './resolve-posix-git-binary'

const PATH_ENV = '/usr/local/bin:/usr/bin'
const INTEL_GIT = '/usr/local/bin/git'
const SYSTEM_GIT = '/usr/bin/git'

test('an existing-but-unrunnable earlier candidate is skipped for one that runs', () => {
  // The reported incident: a leftover Intel git earlier on PATH than a
  // working native one exists but fails "Bad CPU type in executable".
  const result = resolvePosixGitBinary({
    pathEnv: PATH_ENV,
    fileExists: p => p === INTEL_GIT || p === SYSTEM_GIT,
    canExecute: p => p !== INTEL_GIT
  })

  assert.equal(result, SYSTEM_GIT)
})

test('the first candidate wins when it both exists and runs', () => {
  const result = resolvePosixGitBinary({
    pathEnv: PATH_ENV,
    fileExists: p => p === INTEL_GIT || p === SYSTEM_GIT,
    canExecute: () => true
  })

  assert.equal(result, INTEL_GIT)
})

test('when no candidate runs, fall back to the first that exists', () => {
  // Probing may be impossible (sandboxed spawn); preserve the pre-probe
  // behaviour rather than skipping a git that would have worked.
  const result = resolvePosixGitBinary({
    pathEnv: PATH_ENV,
    fileExists: p => p === INTEL_GIT || p === SYSTEM_GIT,
    canExecute: () => false
  })

  assert.equal(result, INTEL_GIT)
})

test('no candidate on PATH returns null', () => {
  const result = resolvePosixGitBinary({
    pathEnv: PATH_ENV,
    fileExists: () => false,
    canExecute: () => false
  })

  assert.equal(result, null)
})

test('empty PATH returns null without throwing', () => {
  const result = resolvePosixGitBinary({
    pathEnv: undefined,
    fileExists: () => true,
    canExecute: () => true
  })

  assert.equal(result, null)
})

test('the probe is not consulted for candidates that do not exist', () => {
  const probed: string[] = []

  resolvePosixGitBinary({
    pathEnv: PATH_ENV,
    fileExists: p => p === SYSTEM_GIT,
    canExecute: p => {
      probed.push(p)

      return true
    }
  })

  assert.deepEqual(probed, [SYSTEM_GIT])
})
