// updater/relaunch-waiter.test.ts — staging, argv, and the ready/error
// handshake contract. No PowerShell is spawned: the spawn point is injected
// and the handshake file is written by the test (as the real script would),
// so the parent-side race contract is exercised without touching a package.

import { EventEmitter } from 'node:events'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import {
  buildRelaunchWaiterArgs,
  POWERSHELL_PATH,
  RELAUNCH_WAITER_READY_FILENAME,
  RELAUNCH_WAITER_SCRIPT,
  type SpawnWaiter,
  startRelaunchWaiter
} from './relaunch-waiter'

const OPTIONS = {
  processId: 4242,
  processStartTimeMs: 1_000_000,
  identityName: 'NousResearch.HermesBundled',
  scriptPath: '/payload/apps/desktop/scripts/update-relaunch-waiter.ps1'
}

const tempDirs: string[] = []

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

function fakeSpawn(handshake: 'ready' | 'error' | 'exit-nonzero' | 'none'): { spawn: SpawnWaiter; seen: any } {
  const seen: any = {}

  const spawn: SpawnWaiter = (_command, args, opts) => {
    seen.command = _command
    seen.args = args
    seen.opts = opts
    tempDirs.push(opts.cwd)

    const child: any = new EventEmitter()

    child.pid = handshake === 'error' ? undefined : 123

    child.unref = () => {}

    child.kill = () => {
      seen.killed = true
      child.emit('exit', null)
      child.emit('close', null)
    }

    const readyFile = args[args.indexOf('-ReadyFile') + 1]

    queueMicrotask(() => {
      if (handshake === 'ready') {
        fs.mkdirSync(path.dirname(readyFile), { recursive: true })
        fs.writeFileSync(readyFile, '0.18.1|Family|App')
      } else if (handshake === 'error') {
        child.emit('error', new Error('spawn ENOENT'))
        child.emit('close', -1)
      } else if (handshake === 'exit-nonzero') {
        child.emit('exit', 3)
        child.emit('close', 3)
      }
    })

    return child
  }

  return { spawn, seen }
}

async function stageScript(): Promise<string> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'relaunch-waiter-test-'))
  tempDirs.push(dir)
  const file = path.join(dir, RELAUNCH_WAITER_SCRIPT)
  fs.writeFileSync(file, '# stub\n')

  return file
}

describe('buildRelaunchWaiterArgs', () => {
  it('carries pid, start time, identity, ready file, and timeout as named parameters', () => {
    const args = buildRelaunchWaiterArgs(OPTIONS, 'C:\\temp\\ready.txt')

    expect(args).toEqual([
      '-NoProfile',
      '-NonInteractive',
      '-ExecutionPolicy',
      'Bypass',
      '-File',
      OPTIONS.scriptPath,
      '-ProcessId',
      '4242',
      '-ProcessStartTimeMs',
      '1000000',
      '-IdentityName',
      'NousResearch.HermesBundled',
      '-ReadyFile',
      'C:\\temp\\ready.txt',
      '-TimeoutSeconds',
      '900'
    ])
  })
})

describe('startRelaunchWaiter', () => {
  it('stages the script into a fresh temp dir and spawns the absolute PowerShell from there', async () => {
    const scriptPath = await stageScript()
    const { spawn, seen } = fakeSpawn('ready')

    const result = await startRelaunchWaiter({ ...OPTIONS, scriptPath }, { spawn, pollMs: 10 })

    expect(result).toHaveProperty('cancel')
    expect(seen.command).toBe(POWERSHELL_PATH)
    expect(seen.opts.detached).toBe(true)
    expect(seen.opts.stdio).toBe('ignore')
    expect(seen.opts.windowsHide).toBe(true)
    // The child runs from the staged dir, not the payload — nothing
    // inherited from the package may hold the swap open.
    expect(seen.opts.cwd).not.toBe(path.dirname(scriptPath))
    expect(seen.args).toContain('-File')
    const stagedScript = seen.args[seen.args.indexOf('-File') + 1]
    expect(stagedScript).not.toBe(scriptPath)
    expect(fs.existsSync(stagedScript)).toBe(true)
  })

  it('retains cancellation after the ready-file handshake', async () => {
    const scriptPath = await stageScript()
    const { spawn } = fakeSpawn('ready')

    const result = await startRelaunchWaiter({ ...OPTIONS, scriptPath }, { spawn, pollMs: 10 })

    expect(result).toHaveProperty('cancel')
    await result!.cancel()
  })

  it('returns no handle after a failed spawn closes', async () => {
    const scriptPath = await stageScript()
    const { spawn } = fakeSpawn('error')

    const result = await startRelaunchWaiter({ ...OPTIONS, scriptPath }, { spawn, pollMs: 10 })

    expect(result).toBeUndefined()
  })

  it('returns no handle on an early exit without readiness', async () => {
    const scriptPath = await stageScript()
    const { spawn } = fakeSpawn('exit-nonzero')

    const result = await startRelaunchWaiter({ ...OPTIONS, scriptPath }, { spawn, pollMs: 10 })

    expect(result).toBeUndefined()
  })

  it('stops the child before returning when readiness times out', async () => {
    const scriptPath = await stageScript()
    const { spawn, seen } = fakeSpawn('none')

    const result = await startRelaunchWaiter(
      { ...OPTIONS, scriptPath },
      { spawn, handshakeTimeoutMs: 80, pollMs: 20 }
    )

    expect(result).toBeUndefined()
    expect(seen.killed).toBe(true)
    expect(fs.existsSync(seen.opts.cwd)).toBe(false)
  }, 5_000)

  it('returns no handle when a missing script leaves no staging', async () => {
    const spawn = (() => {
      throw new Error('should not be reached')
    }) as unknown as SpawnWaiter

    const result = await startRelaunchWaiter(
      { ...OPTIONS, scriptPath: path.join(os.tmpdir(), 'definitely-missing.ps1') },
      { spawn }
    )

    expect(result).toBeUndefined()
  })

  it('the staged handshake file uses the reserved ready filename', async () => {
    const scriptPath = await stageScript()
    const { spawn, seen } = fakeSpawn('ready')

    await startRelaunchWaiter({ ...OPTIONS, scriptPath }, { spawn, pollMs: 10 })

    const readyFile = seen.args[seen.args.indexOf('-ReadyFile') + 1]

    expect(path.basename(readyFile)).toBe(RELAUNCH_WAITER_READY_FILENAME)
  })
})
