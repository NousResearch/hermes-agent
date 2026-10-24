import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import {
  gitFor,
  repoStatus,
  resolveRenamePath,
  REVIEW_FILE_CAP,
  reviewCreatePr,
  reviewList,
  simpleGitHostEnv
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

test('simpleGitHostEnv forwards the allow-listed git variables and drops the other guarded ones', () => {
  const env = simpleGitHostEnv(
    {
      PATH: '/bin',
      HOME: '/home/u',
      GIT_SSH_COMMAND: 'ssh -i key',
      GIT_SSH: '/usr/bin/ssh',
      GIT_ASKPASS: '/usr/bin/askpass',
      SSH_ASKPASS: '/usr/bin/ssh-askpass',
      GIT_TERMINAL_PROMPT: '1',
      GIT_PAGER: 'cat',
      GIT_DIR: '/elsewhere/.git',
      git_work_tree: '/elsewhere',
      GIT_CONFIG_GLOBAL: '/tmp/gitconfig',
      EDITOR: 'vim',
      VISUAL: 'code',
      PAGER: 'less',
      PREFIX: '/usr'
    },
    'git'
  )

  assert.deepEqual(env, {
    PATH: '/bin',
    HOME: '/home/u',
    GIT_SSH_COMMAND: 'ssh -i key',
    GIT_SSH: '/usr/bin/ssh',
    GIT_ASKPASS: '/usr/bin/askpass',
    SSH_ASKPASS: '/usr/bin/ssh-askpass',
    GIT_TERMINAL_PROMPT: '1',
    HERMES_GIT_ARGV0: '"git"'
  })
})

test('simpleGitHostEnv still defaults GIT_TERMINAL_PROMPT off for the host tuple', () => {
  assert.equal(simpleGitHostEnv({ PATH: '/bin' }, 'git').GIT_TERMINAL_PROMPT, '0')
})

// An ssh:// remote whose transport is a script recording that git ran it, so a
// test can tell whether the user's GIT_SSH_COMMAND reached the git process.
function makeRepoWithSshRemote() {
  const dir = makeRepo()
  const toolDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fake-ssh-'))
  const marker = path.join(toolDir, 'ssh-command-ran')
  const sshScript = path.join(toolDir, 'fake-ssh.sh')

  tempDirs.push(toolDir)

  execFileSync('git', ['remote', 'add', 'origin', 'ssh://git@example.invalid/repo.git'], { cwd: dir })
  fs.writeFileSync(sshScript, `#!/bin/sh\necho "$@" > "${marker}"\nexit 1\n`, { mode: 0o755 })

  return { dir, marker, sshScript }
}

// Guarded variables a user's shell commonly carries; simple-git 4 rejects an
// explicit env holding any of them and strips them when inherited.
function stubUserGitEnv(sshScript: string) {
  vi.stubEnv('GIT_SSH_COMMAND', sshScript)
  vi.stubEnv('GIT_PAGER', 'cat')
  vi.stubEnv('GIT_DIR', '/nonexistent/.git')
  vi.stubEnv('EDITOR', 'vi')
}

test.skipIf(process.platform === 'win32')(
  "gitFor runs git with the user's GIT_SSH_COMMAND while other inherited git variables are dropped",
  async () => {
    const { dir, marker, sshScript } = makeRepoWithSshRemote()

    stubUserGitEnv(sshScript)

    try {
      // ls-remote fails (the fake transport exits 1); what matters is that git
      // ran the configured ssh command, and that the inherited GIT_DIR did not
      // point it away from the repository.
      await assert.rejects(gitFor(dir, 'git').raw(['ls-remote', 'origin']), /Could not read from remote repository/)
      assert.ok(fs.existsSync(marker), 'GIT_SSH_COMMAND did not reach git')
    } finally {
      vi.unstubAllEnvs()
    }
  }
)

const HOST_PYTHON =
  process.env.HERMES_PYTHON || process.env.UV_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')

test.skipIf(process.platform === 'win32')(
  "gitFor's Windows host-tuple path forwards GIT_SSH_COMMAND and does not throw on inherited guarded variables",
  async () => {
    // The host tuple is the only path that hands simple-git an explicit env,
    // which simple-git 4 rejects when it carries a guarded key. Drive it for real
    // with the python host script.
    execFileSync(HOST_PYTHON, ['--version'], { stdio: 'ignore' })

    const { dir, marker, sshScript } = makeRepoWithSshRemote()
    const actualNoConsole = await vi.importActual<typeof NoConsoleGit>('./no-console-git')
    const scriptDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-git-host-'))

    tempDirs.push(scriptDir)

    const scriptPath = actualNoConsole.ensureNoConsoleGitScript(scriptDir)

    vi.resetModules()
    vi.doMock('./no-console-git', async importOriginal => ({
      ...(await importOriginal<typeof NoConsoleGit>()),
      windowsGitHost: () => ({ isWindows: true, pythonBin: HOST_PYTHON, scriptPath })
    }))
    stubUserGitEnv(sshScript)

    try {
      const { gitFor: gitForWithHostTuple, repoStatus: repoStatusWithHostTuple } = await import('./git-review-ops')

      fs.writeFileSync(path.join(dir, 'tracked.txt'), 'changed\n')

      const status = await repoStatusWithHostTuple(dir, 'git')

      assert.ok(status)
      assert.deepEqual(
        status.files.map(file => file.path),
        ['tracked.txt']
      )

      await assert.rejects(
        gitForWithHostTuple(dir, 'git').raw(['ls-remote', 'origin']),
        /Could not read from remote repository/
      )
      assert.ok(fs.existsSync(marker), 'GIT_SSH_COMMAND did not reach git through the host tuple')
    } finally {
      vi.unstubAllEnvs()
      vi.doUnmock('./no-console-git')
      vi.resetModules()
    }
  }
)

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
