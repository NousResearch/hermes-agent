import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { probeHermesVersion, remoteSupportsSshOwnership } from './remote-lifecycle'
import { withRemoteTimeout } from './ssh-connection'

const execFileAsync = promisify(execFile)
const shells = ['/bin/sh', '/bin/bash', '/bin/zsh'].filter(existsSync)

test.skipIf(process.platform === 'win32')(
  'SSH probes retain capability checks under noninteractive shells',
  async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'hermes-probe-shell-'))
    const cli = path.join(dir, "hermes ' fixture")
    const version = 'Hermes Agent v0.0.0 (shell fixture)'

    try {
      await writeFile(
        cli,
        `#!/bin/sh\ncase "$*" in\n  --version) printf '%s\\n' '${version}';;\n  'serve --help') printf '%s\\n' "$PROBE_HELP";;\n  *) exit 64;;\nesac\n`,
        { mode: 0o700 }
      )

      for (const shell of shells) {
        for (const flags of [
          '--ssh-session-token-file PATH --ssh-owner-nonce NONCE',
          '--ssh-session-token-file PATH',
          '--ssh-owner-nonce NONCE',
          '--port PORT'
        ]) {
          const ssh = {
            exec: async (command: string) =>
              (
                await execFileAsync(shell, ['-c', command], {
                  env: { HOME: dir, PATH: '/usr/bin:/bin', PROBE_HELP: flags },
                  timeout: 5000
                })
              ).stdout
          }

          assert.equal(
            await remoteSupportsSshOwnership(ssh, cli),
            flags.includes('--ssh-session-token-file') && flags.includes('--ssh-owner-nonce'),
            `${shell}: ${flags}`
          )
          assert.equal(await probeHermesVersion(ssh, cli), version, `${shell}: version probe`)
        }
      }
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  }
)

test.skipIf(process.platform === 'win32')(
  'SSH watchdog preserves exit status and reaps hung probes under noninteractive shells',
  async () => {
    for (const shell of shells) {
      const options = { env: { HOME: os.tmpdir(), PATH: '/usr/bin:/bin' }, timeout: 5000 }

      await assert.rejects(execFileAsync(shell, ['-c', withRemoteTimeout('exit 7', 2)], options), { code: 7 })

      let pid = 0

      try {
        await assert.rejects(
          execFileAsync(shell, ['-c', withRemoteTimeout(`sh -c 'printf "%s\\n" "$$"; exec sleep 30'`, 1)], options),
          (error: any) => {
            pid = Number(String(error.stdout).trim())

            return typeof error.code === 'number' && error.code !== 0
          }
        )
        assert.ok(Number.isInteger(pid) && pid > 1, `${shell}: probe must have started`)
        assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' }, `${shell}: probe must be reaped`)
        pid = 0
      } finally {
        if (Number.isInteger(pid) && pid > 1) {
          try {
            process.kill(pid, 'SIGKILL')
          } catch (error: any) {
            assert.equal(error.code, 'ESRCH', `${shell}: cleanup must remove only the tracked probe`)
          }
        }
      }
    }
  },
  20_000
)
