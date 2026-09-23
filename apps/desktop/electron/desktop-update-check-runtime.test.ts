import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { createDesktopUpdateCheckRuntime } from './desktop-update-check-runtime'

const fixtureRoots: string[] = []
const originalUpdateRoot = process.env.HERMES_DESKTOP_HERMES_ROOT

function git(cwd: string, args: string[]): string {
  return execFileSync('git', args, {
    cwd,
    encoding: 'utf8',
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    windowsHide: true
  }).trim()
}

function makeFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-e07-update-'))
  fixtureRoots.push(root)
  const remote = path.join(root, 'remote.git')
  const local = path.join(root, 'local')
  const userData = path.join(root, 'user-data')
  fs.mkdirSync(local)
  fs.mkdirSync(userData)
  git(root, ['init', '--bare', '-q', '-b', 'main', remote])
  git(local, ['init', '-q', '-b', 'main'])
  git(local, ['config', 'user.name', 'Fixture'])
  git(local, ['config', 'user.email', 'fixture@example.test'])
  fs.writeFileSync(path.join(local, 'readme.txt'), 'initial\n')
  git(local, ['add', 'readme.txt'])
  git(local, ['commit', '-qm', 'initial'])
  git(local, ['remote', 'add', 'origin', remote])
  git(local, ['push', '-q', '-u', 'origin', 'main'])
  process.env.HERMES_DESKTOP_HERMES_ROOT = local

  const progress = vi.fn()
  const logs: string[] = []

  const runtime = createDesktopUpdateCheckRuntime({
    ACTIVE_HERMES_ROOT: local,
    BrowserWindow: { getAllWindows: () => [{ webContents: { send: progress } }] },
    DEFAULT_UPDATE_BRANCH: 'main',
    DESKTOP_UPDATE_CHECK_CACHE_PATH: path.join(userData, 'update-check-cache.json'),
    DESKTOP_UPDATE_CONFIG_PATH: path.join(userData, 'updates.json'),
    IS_PACKAGED: true,
    IS_WINDOWS: process.platform === 'win32',
    SOURCE_REPO_ROOT: root,
    directoryExists: filePath => {
      try {
        return fs.statSync(filePath).isDirectory()
      } catch {
        return false
      }
    },
    isHermesSourceRoot: filePath => filePath === local,
    rememberLog: entry => logs.push(entry),
    resolveGitBinary: () => 'git',
    writeFileAtomic: (targetPath, data, encoding) => {
      fs.writeFileSync(`${targetPath}.tmp`, data, encoding)
      fs.renameSync(`${targetPath}.tmp`, targetPath)
    }
  })

  return { git: (args: string[]) => git(local, args), local, logs, progress, remote, root, runtime, userData }
}

function advanceRemote(fixture: ReturnType<typeof makeFixture>): string {
  const peer = path.join(fixture.root, 'peer')
  git(fixture.root, ['clone', '-q', fixture.remote, peer])
  git(peer, ['config', 'user.name', 'Fixture'])
  git(peer, ['config', 'user.email', 'fixture@example.test'])
  fs.writeFileSync(path.join(peer, 'next.txt'), 'remote only\n')
  git(peer, ['add', 'next.txt'])
  git(peer, ['commit', '-qm', 'remote advance'])
  git(peer, ['push', '-q', 'origin', 'main'])

  return git(peer, ['rev-parse', 'HEAD'])
}

afterEach(() => {
  if (originalUpdateRoot === undefined) {
    delete process.env.HERMES_DESKTOP_HERMES_ROOT
  } else {
    process.env.HERMES_DESKTOP_HERMES_ROOT = originalUpdateRoot
  }

  for (const root of fixtureRoots.splice(0)) {
    const resolved = path.resolve(root)

    if (
      path.dirname(resolved) !== path.resolve(os.tmpdir()) ||
      !path.basename(resolved).startsWith('hermes-e07-update-')
    ) {
      throw new Error(`Refusing to remove unexpected test directory: ${resolved}`)
    }

    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('passive check uses the cache and a forced check sees a moving non-GitHub tip without fetch', async () => {
  const fixture = makeFixture()
  const currentSha = fixture.git(['rev-parse', 'HEAD'])

  const first = await fixture.runtime.checkUpdates()
  assert.equal(first.supported, true)
  assert.equal(first.currentSha, currentSha)
  assert.equal(first.updateAvailable, false)
  assert.equal(first.behind, 0)
  assert.ok(fs.existsSync(path.join(fixture.userData, 'update-check-cache.json')))

  const remoteSha = advanceRemote(fixture)
  const passive = await fixture.runtime.checkUpdates()
  assert.equal(passive.updateAvailable, false)
  assert.equal(passive.currentSha, currentSha)

  const forced = await fixture.runtime.checkUpdates({ force: true })
  assert.equal(forced.updateAvailable, true)
  assert.equal(forced.targetSha, remoteSha)
  assert.equal(forced.behind, null)
  assert.equal(fixture.git(['rev-parse', 'HEAD']), currentSha)
  assert.equal(fs.existsSync(path.join(fixture.local, '.git', 'FETCH_HEAD')), false)
  assert.equal(fixture.git(['status', '--porcelain']), '')
})

test('missing tracked branch heals to main only after a definitive absent-ref probe', async () => {
  const fixture = makeFixture()
  fixture.runtime.writeDesktopUpdateConfig({ branch: 'retired' })

  const result = await fixture.runtime.checkUpdates({ force: true })
  assert.equal(result.branch, 'main')
  assert.equal(result.updateAvailable, false)
  assert.deepEqual(fixture.runtime.readDesktopUpdateConfig(), { branch: 'main' })
  assert.match(fixture.logs.join('\n'), /falling back to main/)
})

test('malformed config falls back to main and progress reaches every current window', () => {
  const fixture = makeFixture()
  fs.writeFileSync(path.join(fixture.userData, 'updates.json'), '{')
  assert.deepEqual(fixture.runtime.readDesktopUpdateConfig(), { branch: 'main' })

  fixture.runtime.emitUpdateProgress({ stage: 'checking', message: 'Checking' })
  assert.equal(fixture.progress.mock.calls.length, 1)
  assert.equal(fixture.progress.mock.calls[0][0], 'hermes:updates:progress')
  assert.equal(fixture.progress.mock.calls[0][1].stage, 'checking')
  assert.equal(fixture.progress.mock.calls[0][1].message, 'Checking')
})
