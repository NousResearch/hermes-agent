import { spawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, test } from 'vitest'

import { desktopMainProcesses, type InstallFacts, installProcesses, releaseDir } from './harness'

// Electron 44 zeroes its processes' /proc/<pid>/environ, so the installed app shows no
// HERMES_HOME. A copy of node under the install's release dir, started with an empty
// environment, is that process.
test.skipIf(process.platform !== 'linux')(
  'the install census still sees the packaged app when its environment names no HERMES_HOME',
  async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'update census '))

    const facts = {
      hermesHome: path.join(root, 'home', '.hermes'),
      checkout: path.join(root, 'checkout')
    } as InstallFacts

    fs.mkdirSync(releaseDir(facts), { recursive: true })
    const exe = path.join(releaseDir(facts), 'Hermes')
    fs.copyFileSync(process.execPath, exe)
    const app = spawn(exe, ['-e', 'setTimeout(() => {}, 60_000)'], { env: {}, stdio: 'ignore' })

    try {
      await expect.poll(() => installProcesses(facts).map(proc => proc.pid)).toContain(app.pid)
      expect(desktopMainProcesses(facts).map(proc => proc.pid)).toEqual([app.pid])
    } finally {
      app.kill('SIGKILL')
      await expect.poll(() => installProcesses(facts).length).toBe(0)
      fs.rmSync(root, { recursive: true, force: true })
    }
  }
)
