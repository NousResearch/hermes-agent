import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import simpleGit from 'simple-git'
import { afterEach, test } from 'vitest'

import {
  gitFor,
  repoStatus,
  resolveRenamePath,
  REVIEW_FILE_CAP,
  reviewList,
  SIMPLE_GIT_UNSAFE_BINARY_WARN
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

test('gitFor accepts internally resolved git paths with restricted non-space characters', () => {
  // simple-git's whitelist is `/^([a-z]:)?([a-z0-9/.\_~-]+)$/i`, so parentheses
  // (`Program Files (x86)`), `+`, and accented profile dirs (`C:\Users\João\...`)
  // are rejected exactly like a space — a `/\s/` guess still throws on them.
  const restrictedBinaries = [
    String.raw`C:\Git(x86)\cmd\git.exe`,
    String.raw`C:\tools\git+portable\cmd\git.exe`,
    String.raw`C:\Users\João\AppData\Local\hermes\git\cmd\git.exe`
  ]

  for (const binary of restrictedBinaries) {
    assert.doesNotThrow(() => gitFor(process.cwd(), binary), `should accept ${binary}`)
  }
})

test('gitFor suppresses only the known custom-binary warning and restores console.warn', () => {
  const spacedBin = String.raw`C:\Program Files\Git\cmd\git.exe`
  // `windowsGitHost()` resolves nothing in this process (no configured roots, no
  // HERMES_DESKTOP_PYTHON), so `gitBin` itself is what simple-git validates — the
  // spaced `Program Files` path, which warns once per factory call.
  const warnings: unknown[][] = []
  const originalWarn = console.warn

  const recordingWarn = (...args: unknown[]) => {
    warnings.push(args)
  }

  console.warn = recordingWarn

  try {
    for (let i = 0; i < 5; i += 1) {
      gitFor(process.cwd(), spacedBin)
    }

    assert.equal(console.warn, recordingWarn)

    // The escape hatch used directly still warns: the message gitFor filters is a
    // live emission of the installed simple-git, so the filter cannot go stale
    // silently (an upgrade that rewords it fails this test, not production).
    simpleGit({ baseDir: process.cwd(), binary: spacedBin, unsafe: { allowUnsafeCustomBinary: true } })
    console.warn('unrelated warning')
  } finally {
    console.warn = originalWarn
  }

  assert.deepEqual(warnings, [[SIMPLE_GIT_UNSAFE_BINARY_WARN], ['unrelated warning']])
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
