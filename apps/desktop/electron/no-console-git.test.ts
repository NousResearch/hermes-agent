import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import { connect } from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import {
  CREATE_NO_WINDOW,
  execGit,
  killTimedGitChildren,
  NO_CONSOLE_GIT_SCRIPT,
  planNoConsoleGitSpawn,
  resolveNoConsolePython,
  simpleGitBinary
} from './no-console-git'

const gitArgs = ['-c', 'windows.appendAtomically=false', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD']
const gitBin = 'C:\\Program Files\\Git\\cmd\\git.exe'

// A node "git" that spawns a grandchild owning a TCP listener, then hangs:
// the listener only goes away if the whole tree is killed.
function hangingTreeFixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'git-timeout-'))
  const pids = path.join(dir, 'pids.json')
  const fixture = path.join(dir, 'hang.cjs')

  fs.writeFileSync(
    fixture,
    `
    const { spawn } = require('node:child_process');
    const worker = String.raw\`const fs = require('node:fs');
      const server = require('node:net').createServer(socket => socket.end());
      server.listen(0, '127.0.0.1', () => fs.writeFileSync(process.argv[1],
        JSON.stringify({ pids: [Number(process.argv[2]), process.pid], port: server.address().port })));\`;
    spawn(process.execPath, ['-e', worker, process.argv[2], String(process.pid)], { stdio: 'inherit' });
    setInterval(() => {}, 1000);
  `
  )

  if (process.platform === 'win32') {
    const python = execFileSync('python3', ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' }).trim()
    vi.stubEnv('HERMES_DESKTOP_PYTHON', python)
  }

  const listening = (port: number): Promise<boolean> =>
    new Promise(resolve => {
      const socket = connect(port, '127.0.0.1')

      const done = (alive: boolean) => {
        socket.destroy()
        resolve(alive)
      }

      socket.once('connect', () => done(true))
      socket.once('error', () => done(false))
      socket.setTimeout(2000, () => done(true))
    })

  const port = async (): Promise<number> => {
    await vi.waitFor(() => expect(fs.existsSync(pids)).toBe(true), { timeout: 5000 })

    return JSON.parse(fs.readFileSync(pids, 'utf8')).port
  }

  const cleanup = () => {
    if (fs.existsSync(pids)) {
      for (const pid of JSON.parse(fs.readFileSync(pids, 'utf8')).pids) {
        try {
          process.kill(pid, 'SIGKILL')
        } catch {
          /* already reaped */
        }
      }
    }

    vi.unstubAllEnvs()
    fs.rmSync(dir, { recursive: true, force: true })
  }

  return { args: [fixture, pids], cleanup, listening, port }
}

test('a timed-out git command reaps descendants, including the Windows Python host', async () => {
  const tree = hangingTreeFixture()

  try {
    const success = await execGit(process.execPath, ['-e', 'process.stdout.write("ok")'], { timeoutMs: 5000 })
    expect(success).toMatchObject({ code: 0, stdout: 'ok' })

    const error = await execGit(process.execPath, tree.args, { timeoutMs: 5000 }).catch(error => error)
    const port = await tree.port()
    await vi.waitFor(async () => expect(await tree.listening(port)).toBe(false), { timeout: 3000 })
    expect(error).toMatchObject({ code: 'ETIMEDOUT' })
  } finally {
    tree.cleanup()
  }
}, 20_000)

test('app quit kills a timed git command still running, descendants included', async () => {
  const tree = hangingTreeFixture()

  try {
    const pending = execGit(process.execPath, tree.args, { timeoutMs: 60_000 }).catch(error => error)
    const port = await tree.port()
    expect(await tree.listening(port)).toBe(true)

    killTimedGitChildren()

    await vi.waitFor(async () => expect(await tree.listening(port)).toBe(false), { timeout: 3000 })
    await pending
  } finally {
    tree.cleanup()
  }
}, 20_000)

test('windows git spawn uses a CREATE_NO_WINDOW host and does not rewrite git argv', () => {
  const plan = planNoConsoleGitSpawn({
    gitBin,
    args: gitArgs,
    isWindows: true,
    pythonBin: 'C:\\hermes\\venv\\Scripts\\python.exe',
    scriptPath: 'C:\\hermes\\no-console-git.py',
    env: { GIT_TERMINAL_PROMPT: '0', PATH: 'C:\\Windows' }
  })

  assert.equal(plan.command, 'C:\\hermes\\venv\\Scripts\\python.exe')
  assert.deepEqual(plan.args, ['C:\\hermes\\no-console-git.py', ...gitArgs])
  assert.equal(plan.env.HERMES_GIT_ARGV0, JSON.stringify(gitBin))
  assert.equal(plan.env.GIT_TERMINAL_PROMPT, '0')
  assert.equal(plan.windowsHide, true)
  assert.deepEqual(plan.stdio, ['ignore', 'pipe', 'pipe'])
  assert.equal(plan.creationFlags, CREATE_NO_WINDOW)
  assert.equal(CREATE_NO_WINDOW, 0x08000000)
})

test('non-windows git spawn keeps the git binary and argv', () => {
  const plan = planNoConsoleGitSpawn({
    gitBin: '/usr/bin/git',
    args: ['status', '--porcelain'],
    isWindows: false,
    pythonBin: '/usr/bin/python3',
    scriptPath: '/tmp/host.py'
  })

  assert.equal(plan.command, '/usr/bin/git')
  assert.deepEqual(plan.args, ['status', '--porcelain'])
  assert.equal(plan.creationFlags, 0)
})

test('missing python does not rewrite git argv', () => {
  const plan = planNoConsoleGitSpawn({
    gitBin: 'git.exe',
    args: gitArgs,
    isWindows: true,
    pythonBin: null,
    scriptPath: 'C:\\hermes\\no-console-git.py'
  })

  assert.equal(plan.command, 'git.exe')
  assert.deepEqual(plan.args, gitArgs)
})

test('simple-git binary is the python host tuple on windows', () => {
  assert.deepEqual(
    simpleGitBinary('git.exe', {
      isWindows: true,
      pythonBin: 'C:\\hermes\\venv\\Scripts\\python.exe',
      scriptPath: 'C:\\hermes\\no-console-git.py'
    }),
    ['C:\\hermes\\venv\\Scripts\\python.exe', 'C:\\hermes\\no-console-git.py']
  )
})

test('python resolver skips pythonw and the WindowsApps stub', () => {
  const python = resolveNoConsolePython({
    isWindows: true,
    env: {
      HERMES_DESKTOP_PYTHON: 'C:\\Users\\me\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe',
      HERMES_DESKTOP_HERMES_ROOT: 'D:\\hermes'
    },
    roots: ['E:\\src'],
    fileExists: () => true
  })

  assert.equal(python, path.win32.join('D:\\hermes', '.venv', 'Scripts', 'python.exe'))
  assert.equal(
    resolveNoConsolePython({
      isWindows: true,
      env: { HERMES_DESKTOP_PYTHON: 'C:\\hermes\\venv\\Scripts\\pythonw.exe' },
      roots: [],
      fileExists: () => true
    }),
    null
  )
  assert.equal(
    resolveNoConsolePython({
      isWindows: true,
      env: { HERMES_DESKTOP_PYTHON: 'C:\\Users\\me\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe' },
      roots: [],
      fileExists: () => true
    }),
    null
  )
})

test.skipIf(process.platform === 'win32')('host script forwards git argv unchanged and sets CREATE_NO_WINDOW', () => {
  // The host script is invoked through a literal `python3`, which does not
  // exist on Windows (the Store alias prints a banner instead). The argv /
  // CREATE_NO_WINDOW contract is covered by resolveNoConsolePython's own
  // Windows-specific specs above.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-no-console-git-'))

  try {
    const script = path.join(dir, 'host.py')

    fs.writeFileSync(script, NO_CONSOLE_GIT_SCRIPT)

    const out = execFileSync('python3', [script, ...gitArgs], {
      encoding: 'utf8',
      env: {
        ...process.env,
        HERMES_GIT_ARGV0: JSON.stringify(gitBin),
        HERMES_GIT_DRY_RUN: '1',
        HERMES_GIT_NO_CONSOLE: '1'
      }
    })

    const parsed = JSON.parse(out)

    assert.deepEqual(parsed.argv, [gitBin, ...gitArgs])
    assert.equal(parsed.creationflags, CREATE_NO_WINDOW)
  } finally {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})
