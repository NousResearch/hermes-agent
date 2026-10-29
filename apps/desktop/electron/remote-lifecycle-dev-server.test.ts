import assert from 'node:assert/strict'
import { exec as execCallback, execFile as execFileCallback } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { buildSpawnCommand, expandRemotePath } from './remote-lifecycle'

const exec = promisify(execCallback)
const execFile = promisify(execFileCallback)

test.skipIf(process.platform === 'win32')(
  'a dev build hands its renderer origin to the remote serve (real sh parse)',
  async (): Promise<void> => {
    // The backend trusts a dev renderer's http origin only when it inherits HERMES_DESKTOP_DEV_SERVER,
    // as a locally spawned backend does. A packaged build (no dev server) must not set it at all.
    const shell: string = (await exec('command -v bash', { shell: 'bash' })).stdout.trim()
    const directory: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-dev-server-env-'))
    const hermesPath: string = path.join(directory, 'hermes')
    const reportPath: string = path.join(directory, 'dev-server-report')

    try {
      // The serve is detached: it reports through a FIFO, whose read blocks until it has written.
      await execFile('mkfifo', [reportPath])
      await writeFile(
        hermesPath,
        `#!${shell}\nprintf '%s' "\${HERMES_DESKTOP_DEV_SERVER-unset}" > ${expandRemotePath(reportPath)}\n`,
        { encoding: 'utf8', mode: 0o700 }
      )

      for (const [devServer, expected] of [
        ["http://127.0.0.1:5175/'$(x)", "http://127.0.0.1:5175/'$(x)"],
        ['', 'unset']
      ]) {
        const command: string = buildSpawnCommand(hermesPath, '', {
          devServer,
          hermesHome: path.join(directory, 'home'),
          logPath: path.join(directory, 'spawn.log')
        })

        // A runner launched from a dev shell may export the variable itself; the spawn must not inherit it.
        const env: NodeJS.ProcessEnv = { ...process.env, HOME: directory, HERMES_HOME: directory }
        delete env.HERMES_DESKTOP_DEV_SERVER
        await exec(command, { shell, env })

        assert.equal(await readFile(reportPath, 'utf8'), expected)
      }
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  }
)
