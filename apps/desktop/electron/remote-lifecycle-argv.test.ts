// Regression tests for the no-/proc ownership-probe fallback (#133970):
// Darwin reads the exact argv from psutil or KERN_PROCARGS2 (via ctypes)
// instead of re-tokenizing the lossy `ps -o command=` join. The synthetic
// KERN_PROCARGS2 buffers drive the parser on Linux and macOS; the chain tests
// force the fallback with a provably-free pid so no /proc is ever read.

import assert from 'node:assert/strict'
import { exec as execCallback, execFile as execFileCallback, spawn } from 'node:child_process'
import { chmod, mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { KERN_PROCARGS2_ARGV_PY, pidIsOurDashboard, shq, spawnTokenPath } from './remote-lifecycle'

const exec = promisify(execCallback)
const execFile = promisify(execFileCallback)

const OWNERSHIP_ID = '0123456789abcdef0123456789abcdef'
const SPAWN_NONCE = '0123456789abcdef'

// A pid that provably does not exist forces the no-/proc branch, so the tests
// exercise the fallback chain on Linux and macOS alike.
function ghostPid(): number {
  for (const candidate of [3999999, 2999999, 1999999, 999999, 499999]) {
    try {
      process.kill(candidate, 0)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ESRCH') {
        return candidate
      }
    }
  }

  throw new Error('no free ghost pid for the fallback test')
}

test.skipIf(process.platform === 'win32')(
  'KERN_PROCARGS2 parser recovers the exact argv from synthetic buffers',
  async (): Promise<void> => {
    const python: string = (await exec('command -v python3')).stdout.trim()

    const driver: string =
      KERN_PROCARGS2_ARGV_PY + 'import json,sys\n' + 'print(json.dumps(_procargs2_argv(bytes.fromhex(sys.argv[1]))))'

    const parse = async (raw: Buffer): Promise<unknown> =>
      JSON.parse((await execFile(python, ['-c', driver, raw.toString('hex')])).stdout.trim())

    // The kernel layout: [argc int32][exec_path NUL][NUL padding][argv..., NUL-terminated].
    const blob = (execPath: string, args: string[], pad = 3, argc = args.length): Buffer => {
      const head = Buffer.alloc(4)

      head.writeInt32LE(argc)

      const parts: Buffer[] = [head, Buffer.from(`${execPath}\0`, 'utf8'), Buffer.alloc(pad)]

      for (const arg of args) {
        parts.push(Buffer.from(`${arg}\0`, 'utf8'))
      }

      return Buffer.concat(parts)
    }

    assert.deepEqual(await parse(blob('/x/hermes', ['/x/hermes', 'serve', '--isolated'])), [
      '/x/hermes',
      'serve',
      '--isolated'
    ])

    // The bug this guards: elements containing spaces must survive as single
    // elements, which is exactly what the ps + shlex fallback cannot do.
    const spaced = '/tmp/hermes wrapper ownership abc/.hermes/desktop-ssh/token file'
    assert.deepEqual(await parse(blob('/x/hermes launcher', ['/x/hermes launcher', 'serve', spaced])), [
      '/x/hermes launcher',
      'serve',
      spaced
    ])

    // Zero padding: the exec path's terminator alone ends the string area.
    assert.deepEqual(await parse(blob('/x/hermes', ['/x/hermes', 'serve'], 0)), ['/x/hermes', 'serve'])

    // argc 0 is an empty argv, not a parse miss.
    assert.deepEqual(await parse(blob('/x/hermes', [])), [])

    // A buffer holding fewer NUL-terminated args than argc claims is a miss,
    // not a short list.
    assert.equal(await parse(blob('/x/hermes', ['a', 'b'], 2, 4)), null)

    // A missing exec-path terminator, a runt header, and a negative argc are
    // misses too.
    assert.equal(await parse(Buffer.concat([Buffer.alloc(4), Buffer.from('/x/hermes', 'utf8')])), null)
    assert.equal(await parse(Buffer.from([1, 0, 0])), null)
    assert.equal(await parse(blob('/x/hermes', [], 0, -1)), null)
  }
)

test.skipIf(process.platform === 'win32')(
  'pidIsOurDashboard reads the exact argv from psutil and keeps the ps fallback',
  async (): Promise<void> => {
    const shell: string = (await exec('command -v bash', { shell: 'bash' })).stdout.trim()
    const temp: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-argv-chain-'))
    const shadow: string = path.join(temp, 'shadow')
    const fakeBin: string = path.join(temp, 'bin')
    const tokenPath: string = path.join(temp, spawnTokenPath(OWNERSHIP_ID, SPAWN_NONCE).replace(/^~\//, ''))

    const sshWith = (env: Record<string, string | undefined>) => ({
      exec: async (command: string): Promise<string> =>
        (await exec(command, { shell, env: { ...process.env, HOME: temp, ...env } })).stdout
    })

    await mkdir(shadow, { recursive: true })
    await mkdir(fakeBin, { recursive: true })

    try {
      // Phase 1: the shadowed psutil is the only exact-argv source (the ghost
      // pid forces the no-/proc branch); OWNED proves the probe consulted it,
      // with the spaced launcher path intact.
      await writeFile(
        path.join(shadow, 'psutil.py'),
        'class Process:\n' +
          '    def __init__(self, pid):\n' +
          '        self.pid = pid\n' +
          '    def cmdline(self):\n' +
          `        return ${JSON.stringify([
            '/x/hermes launcher',
            '--profile',
            'ops',
            'serve',
            '--isolated',
            '--ssh-session-token-file',
            tokenPath,
            '--ssh-owner-nonce',
            SPAWN_NONCE
          ])}\n`,
        'utf8'
      )

      assert.equal(
        await pidIsOurDashboard(
          sshWith({ PYTHONPATH: shadow }),
          ghostPid(),
          SPAWN_NONCE,
          '/x/hermes launcher',
          temp,
          OWNERSHIP_ID,
          'ops'
        ),
        true
      )

      // Phase 2: when psutil raises and no exact source answers, the legacy ps
      // join still runs - a spaceless argv splits back into exact tokens.
      await writeFile(path.join(shadow, 'psutil.py'), 'raise RuntimeError("psutil unavailable")\n', 'utf8')

      const line = [
        '/x/hermes',
        '--profile',
        'ops',
        'serve',
        '--isolated',
        '--ssh-session-token-file',
        tokenPath,
        '--ssh-owner-nonce',
        SPAWN_NONCE
      ].join(' ')

      await writeFile(path.join(fakeBin, 'ps'), `#!/bin/sh\nprintf '%s\\n' ${shq(line)}\n`, {
        encoding: 'utf8',
        mode: 0o755
      })

      assert.equal(
        await pidIsOurDashboard(
          sshWith({ PYTHONPATH: shadow, PATH: `${fakeBin}:${process.env.PATH}` }),
          ghostPid(),
          SPAWN_NONCE,
          '/x/hermes',
          temp,
          OWNERSHIP_ID,
          'ops'
        ),
        true
      )
    } finally {
      await rm(temp, { force: true, recursive: true })
    }
  }
)

test.skipIf(process.platform !== 'darwin')(
  'pidIsOurDashboard reads the exact argv via KERN_PROCARGS2 when psutil is unavailable',
  async (): Promise<void> => {
    const shell: string = (await exec('command -v bash', { shell: 'bash' })).stdout.trim()
    const temp: string = await mkdtemp(path.join(os.tmpdir(), 'hermes kern procargs2 '))
    const installDir = path.join(temp, 'install dir')
    const venvBin = path.join(installDir, 'venv', 'bin')
    const pythonLink = path.join(venvBin, 'python')
    const entrypoint = path.join(installDir, 'hermes')
    const launcher = path.join(temp, 'hermes launcher')
    const python: string = (await exec('command -v python3', { shell })).stdout.trim()
    const tokenPath: string = path.join(temp, spawnTokenPath(OWNERSHIP_ID, SPAWN_NONCE).replace(/^~\//, ''))
    const shadow: string = path.join(temp, 'shadow')
    const env: NodeJS.ProcessEnv = { ...process.env, HOME: temp, HERMES_HOME: temp, PYTHONPATH: shadow }

    await mkdir(venvBin, { recursive: true })
    await mkdir(shadow, { recursive: true })
    await symlink(python, pythonLink)
    await writeFile(entrypoint, 'import time\ntime.sleep(30)\n', 'utf8')
    await writeFile(launcher, `#!${shell}\nexec "${pythonLink}" "${entrypoint}" "$@"\n`, 'utf8')
    await chmod(launcher, 0o755)
    // Force the ctypes leg: the probe's python must not find a usable psutil.
    await writeFile(path.join(shadow, 'psutil.py'), 'raise RuntimeError("psutil unavailable")\n', 'utf8')

    const child: ReturnType<typeof spawn> = spawn(
      launcher,
      [
        '--profile',
        'ops',
        'serve',
        '--isolated',
        '--host',
        '127.0.0.1',
        '--port',
        '0',
        '--ssh-session-token-file',
        tokenPath,
        '--ssh-owner-nonce',
        SPAWN_NONCE
      ],
      { stdio: 'ignore', env }
    )

    try {
      let execed = false

      for (let attempt: number = 0; attempt < 100 && !execed; attempt += 1) {
        const command: string = (await exec(`ps -ww -o command= -p ${child.pid}`, { shell, env })).stdout

        execed = command.includes(entrypoint)

        if (!execed) {
          await new Promise(resolve => setTimeout(resolve, 25))
        }
      }

      assert.equal(execed, true, 'wrapper must exec into the fake installer entrypoint')

      const ssh = {
        exec: async (command: string): Promise<string> => (await exec(command, { shell, env })).stdout
      }

      assert.equal(
        await pidIsOurDashboard(ssh, child.pid, SPAWN_NONCE, launcher, '/unrelated/hermes-home', OWNERSHIP_ID, 'ops'),
        true
      )
    } finally {
      child.kill('SIGKILL')
      await rm(temp, { force: true, recursive: true })
    }
  }
)
