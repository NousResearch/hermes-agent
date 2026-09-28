import assert from 'node:assert/strict'

import { test } from 'vitest'

import { describeGitSpawnFailure, selectRunnableBinary } from './select-runnable-binary'

const yes = () => true
const no = () => false

test.each([
  {
    // The reported machine: Intel-only /usr/local/bin/git ahead on PATH of a
    // working /usr/bin/git — it exists, so existence-only selection commits to
    // it, and it fails at spawn with errno -86 (Bad CPU type in executable).
    name: 'an existing but unlaunchable earlier candidate is skipped for a later one that runs',
    candidates: ['/usr/local/bin/git', '/usr/bin/git'],
    binaryRuns: (p: string) => p === '/usr/bin/git',
    expected: '/usr/bin/git'
  },
  {
    name: 'the first candidate wins when it both exists and runs',
    candidates: ['/opt/homebrew/bin/gh', '/usr/local/bin/gh'],
    binaryRuns: yes,
    expected: '/opt/homebrew/bin/gh'
  },
  {
    // Preserves pre-probe behaviour where the probe itself cannot run
    // (locked-down execution policy, AV interposing on spawn) instead of
    // skipping a binary that would have worked.
    name: 'when no candidate runs, fall back to the first that exists',
    candidates: ['/usr/local/bin/git', '/usr/bin/git'],
    binaryRuns: no,
    expected: '/usr/local/bin/git'
  }
])('$name', ({ candidates, binaryRuns, expected }) => {
  assert.equal(selectRunnableBinary({ candidates, fileExists: yes, binaryRuns }), expected)
})

test('missing candidates are never probed, and nothing existing yields null for the caller fallback', () => {
  // Probing a non-existent path would cost a failed spawn per candidate.
  const probed: string[] = []

  const result = selectRunnableBinary({
    candidates: ['/opt/homebrew/bin/gh', '/usr/local/bin/gh', '/usr/bin/gh'],
    fileExists: (p: string) => p === '/usr/bin/gh',
    binaryRuns: (p: string) => {
      probed.push(p)

      return true
    }
  })

  assert.equal(result, '/usr/bin/gh')
  assert.deepEqual(probed, ['/usr/bin/gh'])

  assert.equal(selectRunnableBinary({ candidates: ['/opt/homebrew/bin/gh'], fileExists: no, binaryRuns: no }), null)
  assert.equal(selectRunnableBinary({ candidates: [], fileExists: yes, binaryRuns: yes }), null)
})

test.each([
  {
    // The reported machine: Darwin errno 86 (Bad CPU type) reaches Node as a
    // bare `errno: -86` with no `code` — "spawn Unknown system error -86".
    name: 'EBADARCH via errno -86 names the binary and the CPU-type cause',
    error: { errno: -86, syscall: 'spawn', message: 'spawn Unknown system error -86' },
    expected: /\/usr\/local\/bin\/git.*Bad CPU type/
  },
  {
    name: 'ENOENT reads as a missing binary',
    error: { code: 'ENOENT', errno: -2, syscall: 'spawn /usr/local/bin/git' },
    expected: /\/usr\/local\/bin\/git: not found/
  }
])('$name', ({ error, expected }) => {
  assert.match(describeGitSpawnFailure(error, '/usr/local/bin/git') ?? '', expected)
})

test('a git that ran and exited nonzero is not a spawn failure and keeps its own wording', () => {
  // Nonzero exits never fire the child 'error' event; only spawn-level
  // failures (binary missing / not executable / wrong architecture) do.
  assert.equal(describeGitSpawnFailure(new Error('fatal: not a git repository'), '/usr/bin/git'), null)
  assert.equal(describeGitSpawnFailure({ code: 'ETIMEDOUT' }, '/usr/bin/git'), null)
})
