import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import {
  gitFor,
  repoStatus,
  resolveRenamePath,
  REVIEW_FILE_CAP,
  reviewDiff,
  reviewList,
  reviewRevert,
  reviewStage,
  reviewUnstage
} from './git-review-ops'

const tempDirs: string[] = []

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

function makeRepo() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-git-status-'))

  tempDirs.push(dir)
  execFileSync('git', ['init', '-q'], { cwd: dir })
  execFileSync('git', ['config', 'user.email', 'hermes-test@example.com'], { cwd: dir })
  execFileSync('git', ['config', 'user.name', 'Hermes Test'], { cwd: dir })
  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'tracked\n')
  execFileSync('git', ['add', 'tracked.txt'], { cwd: dir })
  execFileSync('git', ['commit', '-qm', 'initial'], { cwd: dir })

  return dir
}

test('resolveRenamePath: plain path is unchanged', () => {
  assert.equal(resolveRenamePath('src/a.ts'), 'src/a.ts')
})

test('gitFor accepts an internally resolved git binary path containing spaces', () => {
  assert.doesNotThrow(() => gitFor(process.cwd(), 'C:\\Program Files\\Git\\cmd\\git.exe'))
})

test('gitFor runs git through a spaced binary path', async () => {
  if (process.platform !== 'win32') {
    return
  }

  const gitBin = path.join(process.env.ProgramFiles || String.raw`C:\Program Files`, 'Git', 'cmd', 'git.exe')

  if (!fs.existsSync(gitBin)) {
    return
  }

  const repo = makeRepo()

  fs.writeFileSync(path.join(repo, 'changed.txt'), 'review me\n')

  const status = await gitFor(repo, gitBin).status()

  assert.equal(status.not_added.includes('changed.txt'), true)
})

test('resolveRenamePath: simple rename resolves to the new path', () => {
  assert.equal(resolveRenamePath('old.ts => new.ts'), 'new.ts')
})

test('resolveRenamePath: brace rename resolves to the new path', () => {
  assert.equal(resolveRenamePath('src/{old => new}/file.ts'), 'src/new/file.ts')
})

test('resolveRenamePath: brace rename collapsing a segment', () => {
  assert.equal(resolveRenamePath('src/{lib => }/file.ts'), 'src/file.ts')
})

test('repoStatus reports an untracked directory without recursively listing its contents', async () => {
  const dir = makeRepo()
  const nested = path.join(dir, 'generated', 'deep')

  fs.mkdirSync(nested, { recursive: true })
  fs.writeFileSync(path.join(nested, 'large-output.txt'), 'generated\n')

  const status = await repoStatus(dir, 'git')

  assert.ok(status)
  assert.equal(status.untracked, 1)
  assert.equal(status.changed, 1)
  assert.deepEqual(
    status.files.map(file => file.path),
    ['generated/']
  )
})

test('reviewList reports an untracked directory without recursively listing its contents', async () => {
  const dir = makeRepo()
  const nested = path.join(dir, 'browser-profile', 'Default', 'Cache')

  fs.mkdirSync(nested, { recursive: true })

  for (let i = 0; i < 20; i++) {
    fs.writeFileSync(path.join(nested, `cache-${i}.bin`), 'generated\n')
  }

  const result = await reviewList(dir, 'uncommitted', null, 'git')

  assert.deepEqual(
    result.files.map(file => file.path),
    ['browser-profile/']
  )
})

test('reviewList caps the file payload returned to the renderer', async () => {
  const dir = makeRepo()

  for (let i = 0; i < REVIEW_FILE_CAP + 10; i++) {
    fs.writeFileSync(path.join(dir, `untracked-${String(i).padStart(4, '0')}.txt`), 'generated\n')
  }

  const result = await reviewList(dir, 'uncommitted', null, 'git')

  assert.equal(result.files.length, REVIEW_FILE_CAP)
})

function makeNestedRepo() {
  const dir = makeRepo()
  const sub = path.join(dir, 'sub')

  fs.mkdirSync(sub)
  fs.writeFileSync(path.join(sub, 'nested.txt'), 'nested\n')
  execFileSync('git', ['add', 'sub/nested.txt'], { cwd: dir })
  execFileSync('git', ['commit', '-qm', 'nested'], { cwd: dir })
  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'tracked changed\n')
  fs.writeFileSync(path.join(sub, 'nested.txt'), 'nested changed\n')

  return { dir, sub }
}

test('Review list and diff use repo-root-relative paths from a nested cwd', async () => {
  const { sub } = makeNestedRepo()
  const listed = await reviewList(sub, 'uncommitted', null, 'git')

  assert.deepEqual(
    listed.files.map(file => file.path),
    ['sub/nested.txt', 'tracked.txt']
  )
  assert.match(String(await reviewDiff(sub, 'tracked.txt', 'uncommitted', null, false, 'git')), /tracked changed/)
})

test('Review stage and unstage target repo-root-relative paths from a nested cwd', async () => {
  const { dir, sub } = makeNestedRepo()

  await reviewStage(sub, 'tracked.txt', 'git')
  assert.ok((await gitFor(dir, 'git').status()).staged.includes('tracked.txt'))

  await reviewUnstage(sub, 'tracked.txt', 'git')
  assert.ok(!(await gitFor(dir, 'git').status()).staged.includes('tracked.txt'))
})

test('Review revert targets a repo-root-relative path from a nested cwd', async () => {
  const { dir, sub } = makeNestedRepo()

  await reviewRevert(sub, 'tracked.txt', 'git')

  assert.equal(fs.readFileSync(path.join(dir, 'tracked.txt'), 'utf8'), 'tracked\n')
  assert.equal(fs.readFileSync(path.join(sub, 'nested.txt'), 'utf8'), 'nested changed\n')
})
