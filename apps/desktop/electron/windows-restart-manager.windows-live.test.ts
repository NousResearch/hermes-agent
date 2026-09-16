/**
 * windows-restart-manager.windows-live.test.ts
 *
 * LIVE Windows proof for the Restart Manager query and its ambiguous-holder
 * attribution: the generated script is compiled and executed by real Windows
 * PowerShell, against real processes, real module lists and a real junction.
 *
 * These replace source-contract assertions that could only restate the
 * generated text. Everything here is bounded, runs under a private temp
 * directory, and only spawns children it kills itself.
 */

import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { afterAll, describe, it } from 'vitest'

import { windowsPowerShellExecutable, windowsSystem32Dir, windowsSystem32Executable } from './windows-powershell-path'
import { queryWindowsProcessCreatedAt } from './windows-process-identity'
import {
  buildRestartManagerScript,
  listRestartManagerHoldersForResources,
  RESTART_MANAGER_ATTRIBUTION_FUNCTION,
  RESTART_MANAGER_NATIVE_SOURCE,
  writeRestartManagerResourceList
} from './windows-restart-manager'

const execFileAsync = promisify(execFile)
const isWindows = process.platform === 'win32'
const powershell = windowsPowerShellExecutable()

const cleanups: Array<() => void> = []

afterAll(() => {
  for (const cleanup of cleanups.splice(0)) {
    try {
      cleanup()
    } catch {
      /* best effort */
    }
  }
})

function tempDir(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix))
  cleanups.push(() => fs.rmSync(dir, { recursive: true, force: true }))

  return dir
}

/** A real, long-lived child whose module list we can enumerate. */
async function spawnProbeTarget(): Promise<{ pid: number; createdAt: number; kill: () => void }> {
  const child = spawn(powershell, ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', 'Start-Sleep -Seconds 90'], {
    stdio: 'ignore',
    windowsHide: true
  })

  if (!child.pid) {throw new Error('probe target failed to spawn')}

  const kill = () => {
    try {
      child.kill()
    } catch {
      /* already gone */
    }
  }

  cleanups.push(kill)

  const createdAt = await queryWindowsProcessCreatedAt(child.pid)

  if (createdAt === null) {
    kill()
    throw new Error('probe target create time unavailable')
  }

  return { pid: child.pid, createdAt, kill }
}

/**
 * Run the production attribution function, exactly as the generated script
 * embeds it, over one synthetic row.
 */
async function resolveAttribution(
  holderPid: number,
  expectedCreatedAt: number,
  root: string
): Promise<{ status: string; path: string }> {
  const script = [
    "$ErrorActionPreference = 'Stop'",
    `$rmSource = @"\n${RESTART_MANAGER_NATIVE_SOURCE}\n"@`,
    'Add-Type -TypeDefinition $rmSource',
    RESTART_MANAGER_ATTRIBUTION_FUNCTION,
    // The root arrives exactly as the desktop spelled it; canonicalization is
    // the function's job, not the caller's.
    `$rootClaim = '${root.replace(/'/g, "''")}'`,
    `Resolve-HolderAttribution ${Math.trunc(holderPid)} ([double]${expectedCreatedAt}) $rootClaim | ConvertTo-Json -Compress`
  ].join('\n')

  const { stdout } = await execFileAsync(
    powershell,
    ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
    { encoding: 'utf8', timeout: 60_000, windowsHide: true, maxBuffer: 4 * 1024 * 1024 }
  )

  return JSON.parse(String(stdout).trim())
}

describe.skipIf(!isWindows)('restart manager attribution against real processes', () => {
  it('attributes a holder through a junctioned root by canonical path', { timeout: 120_000 }, async () => {
    const target = await spawnProbeTarget()
    const dir = tempDir('hermes-rm-junction-')
    const link = path.join(dir, 'system32-link')

    // A junction spells the same directory differently. The raw module path
    // compare this replaced could never match it, so a real holder of our own
    // install was reported as somebody else's link and dropped.
    await execFileAsync(windowsSystem32Executable('cmd.exe'), ['/c', 'mklink', '/J', link, windowsSystem32Dir()], {
      windowsHide: true,
      timeout: 10_000
    })

    const resolved = await resolveAttribution(target.pid, target.createdAt, link)

    assert.equal(resolved.status, 'mapped')
    assert.match(resolved.path.toLowerCase(), /\\system32\\/)
    target.kill()
  })

  it('reports a holder whose module list cannot be read as unattributed, never as foreign', { timeout: 120_000 }, async () => {
    // PID 0x7FFFFFF0 cannot exist; GetProcessById throws. The old loop caught
    // that and `continue`d, so an unreadable holder silently vanished from the
    // holder list and the update proceeded over a file it still had mapped.
    const resolved = await resolveAttribution(0x7ffffff0, 1_700_000_000, windowsSystem32Dir())

    assert.equal(resolved.status, 'unattributed')
    assert.equal(resolved.path, '')
  })

  it('refuses to read a reused PID as the holder RM named', { timeout: 120_000 }, async () => {
    const target = await spawnProbeTarget()

    // Same live PID, a generation that is an hour older: this is what PID
    // reuse looks like. Its module list is a stranger's and must not decide
    // anything.
    const resolved = await resolveAttribution(target.pid, target.createdAt - 3_600, windowsSystem32Dir())

    assert.equal(resolved.status, 'unattributed')
    target.kill()
  })

  it('drops a holder that maps nothing under the root once its modules were read', { timeout: 120_000 }, async () => {
    const target = await spawnProbeTarget()
    const dir = tempDir('hermes-rm-foreign-')

    const resolved = await resolveAttribution(target.pid, target.createdAt, dir)

    assert.equal(resolved.status, 'foreign')
    assert.equal(resolved.path, '')
    target.kill()
  })
})

describe.skipIf(!isWindows)('restart manager query against real Windows PowerShell', () => {
  it('compiles, queries RM and names the process holding a locked file', { timeout: 120_000 }, async () => {
    const dir = tempDir('hermes-rm-query-')
    const locked = path.join(dir, 'held.bin')

    fs.writeFileSync(locked, 'x')

    // Hold the file open in a real child, then ask RM who has it.
    const holder = spawn(
      powershell,
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$f = [System.IO.File]::Open('${locked.replace(/'/g, "''")}', 'Open', 'ReadWrite', 'None'); Start-Sleep -Seconds 90; $f.Close()`
      ],
      { stdio: 'ignore', windowsHide: true }
    )

    cleanups.push(() => {
      try {
        holder.kill()
      } catch {
        /* already gone */
      }
    })

    assert.ok(holder.pid, 'holder spawned')

    // Give the child time to open the handle.
    for (let attempt = 0; attempt < 40; attempt++) {
      try {
        fs.closeSync(fs.openSync(locked, 'r+'))
        await new Promise(resolve => setTimeout(resolve, 250))
      } catch {
        break
      }
    }

    const holders = await listRestartManagerHoldersForResources([locked], { timeoutMs: 60_000 })

    holder.kill()

    assert.ok(
      holders.some(entry => entry.pid === holder.pid),
      `RM named the real holder; got ${JSON.stringify(holders)}`
    )
    // The row split is a literal pipe: an over-escaped pattern would have made
    // every row unparseable and the list empty.
    assert.ok(holders.every(entry => Number.isFinite(entry.createdAt) && entry.createdAt > 0))
  })

  it('emits a script that parses and runs even with quotes in the list path', { timeout: 120_000 }, async () => {
    const dir = tempDir("hermes-rm-quote-")
    const listPath = writeRestartManagerResourceList([path.join(dir, "it's.pyd")], [], dir)
    const script = buildRestartManagerScript(listPath, { attributionRoot: dir })

    const { stdout } = await execFileAsync(
      powershell,
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
      { encoding: 'utf8', timeout: 60_000, windowsHide: true, maxBuffer: 4 * 1024 * 1024 }
    )

    // Nothing holds a file that does not exist; an empty result still proves
    // the script compiled the shim and completed.
    assert.match(String(stdout).trim(), /^(\[\]|)$/)
  })
})
