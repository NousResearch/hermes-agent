import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { splitDiffHunks } from '../../shared/src/diff-hunks'

import { reviewHunk, STALE_HUNK_ERROR } from './git-review-hunk'

const tempDirs: string[] = []

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

const git = (cwd: string, ...args: string[]) => execFileSync('git', args, { cwd, encoding: 'utf8' })

// A tracked sub/f.txt with two separate edits (lines 3 and 25), so its diff has two hunks.
function twoHunkRepo() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-review-hunk-'))

  tempDirs.push(dir)
  git(dir, 'init', '-q')
  fs.mkdirSync(path.join(dir, 'sub'))
  const file = path.join(dir, 'sub', 'f.txt')
  const lines = Array.from({ length: 30 }, (_, i) => String(i + 1))
  fs.writeFileSync(file, `${lines.join('\n')}\n`)
  git(dir, 'add', '.')
  git(dir, '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', 'commit', '-qm', 'init')
  lines[2] = 'three'
  lines[24] = 'twentyfive'
  fs.writeFileSync(file, `${lines.join('\n')}\n`)

  return { dir, file, hunks: splitDiffHunks(git(dir, 'diff', '--', 'sub/f.txt')).hunks }
}

test('one change is staged or reverted alone, from a session cwd below the repo root', async () => {
  const { dir, file, hunks } = twoHunkRepo()
  const sub = path.join(dir, 'sub')

  assert.equal(hunks.length, 2)
  await reviewHunk(sub, 'f.txt', hunks[1], 'stage', 'git')
  assert.match(git(dir, 'diff', '--cached'), /\+twentyfive/)
  assert.doesNotMatch(git(dir, 'diff', '--cached'), /three/)

  await reviewHunk(sub, 'f.txt', hunks[0], 'revert', 'git')
  assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /three/)
  assert.match(fs.readFileSync(file, 'utf8'), /twentyfive/)
  assert.deepEqual(
    fs.readdirSync(path.join(dir, '.git')).filter(name => name.startsWith('hermes-review-hunk-')),
    []
  )
})

test('a change the file no longer holds is refused and the file is left untouched', async () => {
  const { dir, file, hunks } = twoHunkRepo()

  fs.writeFileSync(file, fs.readFileSync(file, 'utf8').replace('three', 'THREE'))
  const before = fs.readFileSync(file, 'utf8')

  await assert.rejects(reviewHunk(dir, 'sub/f.txt', hunks[0], 'revert', 'git'), { message: STALE_HUNK_ERROR })
  assert.equal(fs.readFileSync(file, 'utf8'), before)
})
