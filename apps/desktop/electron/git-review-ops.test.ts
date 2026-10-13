import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import type * as ChildProcess from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import {
  gitFor,
  repoStatus,
  resolveRenamePath,
  REVIEW_FILE_CAP,
  reviewCommit,
  reviewCreatePr,
  reviewDiff,
  reviewList,
  reviewStage,
  reviewUnstage
} from './git-review-ops'
import type * as NoConsoleGit from './no-console-git'

// `runGh` shells to the `gh` CLI via execFile. Mock it so reviewCreatePr's gh
// invocation is controllable (real `gh` may be absent or slow in CI) while the
// repo setup below still uses the real execFileSync.
vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<{ execFile: unknown; execFileSync: unknown }>()

  return { ...actual, execFile: vi.fn() }
})

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

test('gitFor accepts a Windows no-console host tuple with restricted characters', async () => {
  vi.resetModules()
  vi.doMock('./no-console-git', async importOriginal => {
    const actual = await importOriginal<typeof NoConsoleGit>()

    return {
      ...actual,
      windowsGitHost: () => ({
        isWindows: true,
        pythonBin: String.raw`C:\Tools\python-3.14+build\python.exe`,
        scriptPath: String.raw`C:\Hermes\hermes-no-console-git.py`
      })
    }
  })

  try {
    const { gitFor: gitForWithHostTuple } = await import('./git-review-ops')

    assert.doesNotThrow(() => gitForWithHostTuple(process.cwd(), 'git'))
  } finally {
    vi.doUnmock('./no-console-git')
    vi.resetModules()
  }
})

test('gitFor leaves console.warn intact for internally resolved binaries', () => {
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

    console.warn('unrelated warning')
  } finally {
    console.warn = originalWarn
  }

  assert.deepEqual(warnings, [['unrelated warning']])
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

const mockExecFile = vi.mocked(await import('node:child_process')).execFile

type ExecFileCallback = (error: Error | null, stdout?: string, stderr?: string) => void

// `execFile` has overloaded declarations returning ChildProcess; the mock
// implementation only needs to drive its callback, so view it as a plain
// callable and set the implementation through the Mock typing.
function failGh(stderr: string): void {
  ;(
    mockExecFile as unknown as {
      mockImplementation: (
        impl: (file: string, args: string[], options: object, callback: ExecFileCallback) => void
      ) => unknown
    }
  ).mockImplementation((_bin: string, _args: string[], _opts: object, callback: ExecFileCallback) => {
    const error = new Error('command failed')

    if (stderr) {
      ;(error as Error & { stderr?: string }).stderr = stderr
    }

    callback(error, '', stderr)
  })
}

test('reviewCreatePr surfaces gh stderr when pr create fails', async () => {
  const dir = makeRepo()

  failGh('no commits between main and feature')

  await assert.rejects(reviewCreatePr(dir, 'git', 'gh'), /no commits between main and feature/)
})

test('reviewCreatePr falls back to the generic message when gh reports no stderr', async () => {
  const dir = makeRepo()

  failGh('')

  await assert.rejects(reviewCreatePr(dir, 'git', 'gh'), /is gh installed and authenticated\?/)
})

test('v4 performs review diff, stage, unstage and local commit without a push', async () => {
  const actual = await vi.importActual<typeof ChildProcess>('node:child_process')
  mockExecFile.mockImplementation(actual.execFile)
  const dir = makeRepo()
  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'changed\n')
  await reviewStage(dir, 'tracked.txt', 'git')
  assert.match(await reviewDiff(dir, 'tracked.txt', 'uncommitted', null, true, 'git'), /changed/)
  await reviewUnstage(dir, 'tracked.txt', 'git')
  assert.equal((await repoStatus(dir, 'git'))?.files[0]?.staged, false)
  await reviewCommit(dir, 'local regression commit', false, 'git')
  assert.equal(
    execFileSync('git', ['log', '-1', '--pretty=%s'], { cwd: dir, encoding: 'utf8' }).trim(),
    'local regression commit'
  )
  assert.equal((await repoStatus(dir, 'git'))?.changed, 0)
})

test('v4 refuses unsafe config includes and trailer commands through the production factory', async () => {
  const dir = makeRepo()
  const includes = path.join(dir, 'ordinary-config')
  fs.writeFileSync(includes, '[user]\nname = unused\n')

  for (const config of [`include.path=${includes}`, 'trailer.audit.command=not-a-real-command']) {
    await assert.rejects(gitFor(dir, 'git').raw(['-c', config, 'status', '--short']), /blocked|not permitted|unsafe/i)
  }
})

test('the no-console binary tuple forwards safe environment and strips command hooks', async () => {
  const dir = makeRepo()
  const script = path.join(dir, 'inspect-env.cjs')
  fs.writeFileSync(
    script,
    `process.stdout.write(JSON.stringify({path:process.env.PATH,home:process.env.HOME,agent:process.env.SSH_AUTH_SOCK,token:process.env.GH_TOKEN,argv0:process.env.HERMES_GIT_ARGV0,prompt:process.env.GIT_TERMINAL_PROMPT,config:process.env.GIT_CONFIG_GLOBAL,ssh:process.env.GIT_SSH_COMMAND,editor:process.env.VISUAL}));`
  )
  vi.resetModules()
  vi.doMock('./no-console-git', async importOriginal => {
    const actual = await importOriginal<typeof NoConsoleGit>()

    return { ...actual, windowsGitHost: () => ({ isWindows: true, pythonBin: process.execPath, scriptPath: script }) }
  })
  vi.stubEnv('PATH', 'trusted-path')
  vi.stubEnv('HOME', 'trusted-home')
  vi.stubEnv('SSH_AUTH_SOCK', 'trusted-agent')
  vi.stubEnv('GH_TOKEN', 'test-token')
  vi.stubEnv('GIT_CONFIG_GLOBAL', '/untrusted/config')
  vi.stubEnv('GIT_SSH_COMMAND', 'untrusted-command')
  vi.stubEnv('VISUAL', 'untrusted-editor')

  try {
    const { gitFor: hostedGitFor } = await import('./git-review-ops')
    const env = JSON.parse(await hostedGitFor(dir, 'trusted-git').raw(['status']))
    assert.deepEqual(env, {
      path: 'trusted-path',
      home: 'trusted-home',
      agent: 'trusted-agent',
      token: 'test-token',
      argv0: JSON.stringify('trusted-git'),
      prompt: '0'
    })
  } finally {
    vi.unstubAllEnvs()
    vi.doUnmock('./no-console-git')
    vi.resetModules()
  }
})
