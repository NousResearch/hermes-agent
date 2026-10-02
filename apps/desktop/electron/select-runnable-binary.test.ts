import assert from 'node:assert/strict'

import { test } from 'vitest'

import { selectRunnableBinary } from './select-runnable-binary'

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
