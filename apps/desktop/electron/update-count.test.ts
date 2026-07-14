import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { resolveBehindCount, resolveCommitLogSelection, shouldCountCommits } from './update-count'

function createTempGitRepo() {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-count-'))
  const git = (...args: string[]) => execFileSync('git', args, { cwd, encoding: 'utf8', timeout: 10_000 }).trim()

  try {
    git('init', '--quiet')
    git('config', 'commit.gpgSign', 'false')
    git('config', 'core.hooksPath', '.git/no-hooks')
    git('config', 'user.name', 'Hermes Test')
    git('config', 'user.email', 'hermes@example.invalid')

    return { cwd, git }
  } catch (error) {
    fs.rmSync(cwd, { recursive: true, force: true })
    throw error
  }
}

// FAIL-BEFORE: pre-fix the function did `Number.parseInt(countStr) || 0`
// unconditionally, so a shallow checkout with no merge-base surfaced the bogus
// rev-list count (e.g. 12104). This asserts the original #51922 regression.
test('shallow checkout with no merge-base does NOT trust the bogus rev-list count', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '12104',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true
    }),
    1
  )
})

test('shallow checkout with no merge-base but identical SHA reports up-to-date', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '12104',
      currentSha: 'abc',
      targetSha: 'abc',
      isShallow: true
    }),
    0
  )
})

test('shallow local-ahead checkout reports up-to-date when origin is a known ancestor', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'local-child',
      targetSha: 'origin-parent',
      isShallow: true,
      targetIsAncestorOfHead: true
    }),
    0
  )
})

test('shallow Git graph proves the remote tip is an ancestor of a local commit', () => {
  const { cwd, git } = createTempGitRepo()

  try {
    git('commit', '--allow-empty', '-m', 'origin tip')

    const targetSha = git('rev-parse', 'HEAD')

    git('update-ref', 'refs/remotes/origin/main', targetSha)
    fs.writeFileSync(path.join(cwd, '.git', 'shallow'), `${targetSha}\n`)
    git('commit', '--allow-empty', '-m', 'local child')

    const currentSha = git('rev-parse', 'HEAD')

    git('merge-base', '--is-ancestor', 'origin/main', 'HEAD')
    assert.notEqual(currentSha, targetSha)
    assert.equal(
      resolveBehindCount({
        countStr: '',
        currentSha,
        targetSha,
        isShallow: true,
        targetIsAncestorOfHead: true
      }),
      0
    )
  } finally {
    fs.rmSync(cwd, { recursive: true, force: true })
  }
}, 30_000)

test('shallow checkout with a merge-base does not trust an inflated rev-list count', () => {
  const { cwd, git } = createTempGitRepo()

  try {
    git('commit', '--allow-empty', '-m', 'root')
    git('commit', '--allow-empty', '-m', 'ancestor')

    const redundantParent = git('rev-parse', 'HEAD')

    git('commit', '--allow-empty', '-m', 'installed head')

    const currentSha = git('rev-parse', 'HEAD')
    const tree = git('rev-parse', 'HEAD^{tree}')

    const targetSha = execFileSync('git', ['commit-tree', tree, '-p', currentSha, '-p', redundantParent], {
      cwd,
      encoding: 'utf8',
      input: 'remote merge\n',
      timeout: 10_000
    }).trim()

    git('update-ref', 'refs/remotes/origin/main', targetSha)

    const completeCount = git('rev-list', 'HEAD..origin/main', '--count')

    assert.equal(completeCount, '1')

    fs.writeFileSync(path.join(cwd, '.git', 'shallow'), `${currentSha}\n`)

    assert.equal(git('rev-parse', '--is-shallow-repository'), 'true')
    assert.equal(git('merge-base', 'HEAD', 'origin/main'), currentSha)

    const shallowCount = git('rev-list', 'HEAD..origin/main', '--count')

    assert.ok(Number.parseInt(shallowCount, 10) > Number.parseInt(completeCount, 10))
    assert.equal(
      resolveBehindCount({
        countStr: shallowCount,
        currentSha,
        targetSha,
        isShallow: true
      }),
      1
    )
  } finally {
    fs.rmSync(cwd, { recursive: true, force: true })
  }
}, 30_000)

test('shallow checkout with a merge-base still uses presence-only status', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '3',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true
    }),
    1
  )
})

test('full (non-shallow) clone keeps the exact count path unchanged', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '7',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: false
    }),
    7
  )
})

test('up-to-date full clone reports 0', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '0',
      currentSha: 'x',
      targetSha: 'x',
      isShallow: false
    }),
    0
  )
})

test('non-numeric count falls back to 0 (defensive, unchanged behaviour)', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: false
    }),
    0
  )
})

// shouldCountCommits gates the expensive `rev-list --count` in checkUpdates().
// Every shallow graph is incomplete, so a visible merge-base is not enough to
// prove that the count is exact.
test('shallow checkouts skip the rev-list count', () => {
  assert.equal(shouldCountCommits({ isShallow: true }), false)
})

test('full (non-shallow) clones run the rev-list count', () => {
  assert.equal(shouldCountCommits({ isShallow: false }), true)
})

test('shallow commit logs select only the fetched remote tip', () => {
  assert.deepEqual(resolveCommitLogSelection({ branch: 'main', isShallow: true }), {
    limit: 1,
    revision: 'origin/main'
  })
})

test('full-clone commit logs keep the complete behind range', () => {
  assert.deepEqual(resolveCommitLogSelection({ branch: 'release', isShallow: false }), {
    limit: 40,
    revision: 'HEAD..origin/release'
  })
})

// The skip path produces an empty countStr; resolveBehindCount must NOT trust
// it and must fall through to the SHA compare (mirrors the live call site).
test('skipped-count path resolves via SHA compare, never via empty countStr', () => {
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'aaa',
      targetSha: 'bbb',
      isShallow: true
    }),
    1
  )
  assert.equal(
    resolveBehindCount({
      countStr: '',
      currentSha: 'same',
      targetSha: 'same',
      isShallow: true
    }),
    0
  )
})
