import assert from 'node:assert/strict'

import { test } from 'vitest'

import { selectRunnableBinary } from './select-runnable-binary'

const yes = () => true
const no = () => false

test('an existing but unlaunchable earlier candidate is skipped for a later one that runs', () => {
  // The reported machine: Intel-only /usr/local/bin/git ahead on PATH of a
  // working /usr/bin/git — it exists, so existence-only selection commits to
  // it, and it fails at spawn with errno -86 (Bad CPU type in executable).
  const result = selectRunnableBinary({
    candidates: ['/usr/local/bin/git', '/usr/bin/git'],
    fileExists: yes,
    binaryRuns: (p: string) => p === '/usr/bin/git'
  })

  assert.equal(result, '/usr/bin/git')
})

test('the first candidate wins when it both exists and runs', () => {
  const result = selectRunnableBinary({
    candidates: ['/opt/homebrew/bin/gh', '/usr/local/bin/gh'],
    fileExists: yes,
    binaryRuns: yes
  })

  assert.equal(result, '/opt/homebrew/bin/gh')
})

test('the probe is not consulted for candidates that do not exist', () => {
  // Probing a non-existent path costs a failed spawn per candidate.
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
})

test('when no candidate runs, fall back to the first that exists', () => {
  // Preserves pre-probe behaviour where the probe itself cannot run
  // (locked-down execution policy, AV interposing on spawn) instead of
  // skipping a binary that would have worked.
  const result = selectRunnableBinary({
    candidates: ['/usr/local/bin/git', '/usr/bin/git'],
    fileExists: yes,
    binaryRuns: no
  })

  assert.equal(result, '/usr/local/bin/git')
})

test('all candidates broken on arch except a later one still selects the later one', () => {
  // gh resolver's hand-ordered list: the Intel /usr/local/bin/gh would be
  // picked ahead of /usr/bin/gh by existence alone on the reported machine.
  const result = selectRunnableBinary({
    candidates: ['/opt/homebrew/bin/gh', '/usr/local/bin/gh', '/usr/bin/gh'],
    fileExists: yes,
    binaryRuns: (p: string) => p === '/usr/bin/gh'
  })

  assert.equal(result, '/usr/bin/gh')
})

test('no existing candidate returns null so the caller keeps its own fallback', () => {
  const result = selectRunnableBinary({
    candidates: ['/opt/homebrew/bin/gh', '/usr/local/bin/gh'],
    fileExists: no,
    binaryRuns: no
  })

  assert.equal(result, null)
})

test('an empty candidate list returns null', () => {
  const result = selectRunnableBinary({
    candidates: [],
    fileExists: yes,
    binaryRuns: yes
  })

  assert.equal(result, null)
})
