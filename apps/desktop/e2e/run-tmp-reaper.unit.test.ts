import { type ChildProcess, spawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { reapAbandonedRuns, recordRunOwner } from './run-tmp-reaper'

const children: ChildProcess[] = []
const dirs: string[] = []

afterEach(() => {
  for (const child of children.splice(0)) {
    child.kill('SIGKILL')
  }

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

function exited(child: ChildProcess): Promise<NodeJS.Signals | null> {
  return child.exitCode !== null || child.signalCode !== null
    ? Promise.resolve(child.signalCode)
    : new Promise(resolve => child.once('exit', (_code, signal) => resolve(signal)))
}

/** A stand-in backend: a real process whose environment names `root` the way a sandbox gateway's does. */
function backendUnder(root: string): ChildProcess {
  const child = spawn('sleep', ['300'], {
    env: { PATH: process.env.PATH, HERMES_E2E_RUN_TMP: root, HERMES_HOME: path.join(root, 'sandbox', 'hermes-home') },
    stdio: 'ignore'
  })

  children.push(child)

  return child
}

describe.runIf(process.platform === 'linux')('reapAbandonedRuns', () => {
  it("stops a killed runner's backends and removes its root; a live runner's backends are untouched", async () => {
    const base = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'reaper-')))
    dirs.push(base)
    const deadRoot = fs.mkdtempSync(path.join(base, 'pw-'))
    const liveRoot = fs.mkdtempSync(path.join(base, 'pw-'))

    // The killed run's record: the pid and kernel start time of a runner that no longer exists.
    const runner = spawn('sleep', ['300'], { stdio: 'ignore' })
    const stat = fs.readFileSync(`/proc/${runner.pid}/stat`, 'utf8')
    const start = stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19]
    runner.kill('SIGKILL')
    await exited(runner)
    fs.writeFileSync(path.join(deadRoot, '.runner.json'), JSON.stringify({ pid: runner.pid, start }))
    recordRunOwner(liveRoot)

    const orphan = backendUnder(deadRoot)
    const concurrent = backendUnder(liveRoot)

    expect(reapAbandonedRuns()).toContain(deadRoot)
    expect(await exited(orphan)).toBe('SIGTERM')
    expect(fs.existsSync(deadRoot)).toBe(false)

    expect(concurrent.exitCode).toBeNull()
    expect(concurrent.signalCode).toBeNull()
    expect(() => process.kill(concurrent.pid!, 0)).not.toThrow()
    expect(fs.existsSync(liveRoot)).toBe(true)
  })
})
