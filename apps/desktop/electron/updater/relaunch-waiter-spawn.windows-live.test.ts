import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { buildSync } from 'esbuild'
import { expect, it } from 'vitest'

import { startRelaunchWaiter } from './relaunch-waiter'

it.runIf(process.platform === 'win32')(
  'a waiter that exits before its handshake reports its own exit code',
  async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waiter exit '))
    const script = path.join(root, 'waiter.ps1')
    const logs: string[] = []
    fs.writeFileSync(script, 'exit 2\n')

    try {
      const handle = await startRelaunchWaiter(
        {
          processId: process.pid,
          processStartTimeMs: Date.now() - process.uptime() * 1000,
          identityName: 'disposable-waiter-test',
          scriptPath: script
        },
        { log: message => logs.push(message) }
      )

      expect(handle).toBeUndefined()
      expect(logs.join('\n')).toMatch(/exited before its handshake after \d+ms \(exit code 2,/)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  },
  30_000
)

it.runIf(process.platform === 'win32').each([false, true])(
  'a real hidden PowerShell waiter survives parent exit unless cancelled and leaves no stage dir (cancel=%s)',
  async cancel => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waiter native '))
    const script = path.join(root, 'waiter.ps1')
    const result = path.join(root, 'result.json')
    const completed = path.join(root, 'completed.txt')
    const bundle = path.join(root, 'waiter.cjs')
    fs.writeFileSync(
      script,
      `
param([int]$ProcessId, [long]$ProcessStartTimeMs, [string]$IdentityName, [string]$ReadyFile, [int]$TimeoutSeconds)
Set-Content -LiteralPath ($IdentityName + '.pid') -Value $PID
Set-Content -LiteralPath ($IdentityName + '.stage') -Value $PSScriptRoot
Set-Content -LiteralPath $ReadyFile -Value 'ready'
$deadline = (Get-Date).AddSeconds(20)
while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
  if ((Get-Date) -gt $deadline) { exit 3 }
  Start-Sleep -Milliseconds 100
}
# The same exit cleanup as update-relaunch-waiter.ps1.
Set-Location $env:TEMP
Remove-Item -LiteralPath $PSScriptRoot -Recurse -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $IdentityName -Value 'parent exited'
`
    )
    buildSync({
      entryPoints: [path.resolve(import.meta.dirname, 'relaunch-waiter.ts')],
      outfile: bundle,
      bundle: true,
      platform: 'node',
      format: 'cjs'
    })

    try {
      const parent = spawnSync(
        process.execPath,
        [
          '-e',
          `
const fs = require('node:fs');
const { startRelaunchWaiter } = require(${JSON.stringify(bundle)});
(async () => {
  const logs = [];
  const handle = await startRelaunchWaiter({
    processId: process.pid, processStartTimeMs: Date.now() - process.uptime() * 1000,
    identityName: ${JSON.stringify(completed)}, scriptPath: ${JSON.stringify(script)}
  }, { log: message => logs.push(message) });
  if (${cancel} && handle) await handle.cancel();
  fs.writeFileSync(${JSON.stringify(result)}, JSON.stringify({ ready: !!handle, logs }));
})().catch(error => { console.error(error); process.exitCode = 1; });
`
        ],
        { encoding: 'utf8', windowsHide: true, timeout: 30_000 }
      )

      expect(parent.status, parent.stderr).toBe(0)
      const registration = JSON.parse(fs.readFileSync(result, 'utf8'))
      expect(registration.ready, registration.logs.join('\n')).toBe(true)

      if (cancel) {
        const workerPid = Number(fs.readFileSync(completed + '.pid', 'utf8').trim())
        expect(() => process.kill(workerPid, 0)).toThrow()
        expect(fs.existsSync(completed)).toBe(false)
      } else {
        await expect.poll(() => fs.existsSync(completed), { timeout: 10_000 }).toBe(true)
        expect(fs.readFileSync(completed, 'utf8').trim()).toBe('parent exited')
      }

      const stage = fs.readFileSync(completed + '.stage', 'utf8').trim()
      expect(path.basename(stage)).toMatch(/^hermes-relaunch-/)
      expect(fs.existsSync(stage), `${stage} is left behind`).toBe(false)
    } finally {
      if (fs.existsSync(completed + '.pid')) {
        const workerPid = Number(fs.readFileSync(completed + '.pid', 'utf8').trim())

        try {
          process.kill(workerPid)
        } catch {
          /* The fixture already exited. */
        }
      }

      fs.rmSync(root, { recursive: true, force: true })
    }
  },
  45_000
)
