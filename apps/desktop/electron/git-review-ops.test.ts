import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import {
  resolveRenamePath,
  REVIEW_SNAPSHOT_LIMIT,
  reviewDiff,
  reviewList,
  reviewReleaseSnapshot,
  reviewSnapshot
} from './git-review-ops'

const roots: string[] = []

function git(cwd: string, ...args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile('git', args, { cwd, encoding: 'utf8' }, (error, stdout) =>
      error ? reject(error) : resolve(String(stdout || ''))
    )
  })
}

async function repository(): Promise<string> {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-review-'))
  roots.push(root)
  await git(root, 'init', '-q')
  await git(root, 'config', 'user.email', 'test@example.com')
  await git(root, 'config', 'user.name', 'Test')
  fs.writeFileSync(path.join(root, 'tracked.txt'), 'committed\n')
  await git(root, 'add', '-A')
  await git(root, 'commit', '-qm', 'initial')

  return root
}

function objectFiles(root: string): string[] {
  const objects = path.join(root, '.git', 'objects')

  return fs
    .readdirSync(objects, { recursive: true, withFileTypes: true })
    .filter(entry => entry.isFile())
    .map(entry => path.join(entry.parentPath, entry.name))
    .sort()
}

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('resolveRenamePath: plain path is unchanged', () => {
  assert.equal(resolveRenamePath('src/a.ts'), 'src/a.ts')
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

test('last-turn trees exclude unchanged pre-existing dirt while preserving the real index', async () => {
  const root = await repository()
  fs.writeFileSync(path.join(root, 'tracked.txt'), 'dirty before turn\n')
  fs.writeFileSync(path.join(root, 'pre-existing.txt'), 'already here\n')
  const statusBefore = await git(root, 'status', '--porcelain=v1')
  const objectsBefore = objectFiles(root)
  const baseline = await reviewSnapshot(root, undefined)

  assert.match(baseline ?? '', /^[0-9a-f]{40,64}$/)
  assert.equal(await git(root, 'status', '--porcelain=v1'), statusBefore)
  assert.deepEqual(objectFiles(root), objectsBefore)

  fs.writeFileSync(path.join(root, 'tracked.txt'), 'changed during turn\n')
  fs.writeFileSync(path.join(root, 'turn-only.txt'), 'new this turn\n')

  const result = await reviewList(root, 'lastTurn', baseline, undefined)
  const paths = result.files.map(file => file.path)

  assert.deepEqual(paths, ['tracked.txt', 'turn-only.txt'])
  assert.equal(await reviewDiff(root, 'pre-existing.txt', 'lastTurn', baseline, false, undefined), '')
  assert.match(String(await reviewDiff(root, 'turn-only.txt', 'lastTurn', baseline, false, undefined)), /new this turn/)
  assert.match(await git(root, 'status', '--porcelain=v1'), /pre-existing\.txt/)
})

test('last-turn baseline survives more transient snapshots than the retention limit', async () => {
  const root = await repository()
  const baseline = await reviewSnapshot(root, undefined, 'session-a')
  const transientSnapshotCount = REVIEW_SNAPSHOT_LIMIT + 1

  assert.match(baseline ?? '', /^[0-9a-f]{40,64}$/)

  for (let index = 0; index < transientSnapshotCount; index++) {
    fs.writeFileSync(path.join(root, 'tracked.txt'), `refresh ${index}\n`)
    await reviewList(root, 'uncommitted', null, undefined)
  }

  const diff = await reviewDiff(root, 'tracked.txt', 'lastTurn', baseline, false, undefined)

  assert.match(String(diff), new RegExp(`\\+refresh ${transientSnapshotCount - 1}`))
  assert.deepEqual(reviewReleaseSnapshot(root, 'session-a'), { ok: true })
}, 15_000)

test('branch scope includes uncommitted and untracked workspace changes', async () => {
  const root = await repository()
  fs.writeFileSync(path.join(root, 'tracked.txt'), 'working tree\n')
  fs.writeFileSync(path.join(root, 'new.txt'), 'untracked\n')

  const result = await reviewList(root, 'branch', null, undefined)

  assert.deepEqual(
    result.files.map(file => file.path),
    ['new.txt', 'tracked.txt']
  )
})

test('review snapshots do not execute repository-configured filters', async () => {
  const root = await repository()
  const marker = path.join(root, 'filter-ran')
  fs.writeFileSync(path.join(root, '.gitattributes'), '*.txt filter=unsafe\n')
  await git(root, 'config', 'filter.unsafe.clean', `sh -c 'echo invoked > ${marker}; cat'`)
  fs.writeFileSync(path.join(root, 'tracked.txt'), 'snapshot me\n')

  assert.match((await reviewSnapshot(root, undefined)) ?? '', /^[0-9a-f]{40,64}$/)
  assert.equal(fs.existsSync(marker), false)
})
