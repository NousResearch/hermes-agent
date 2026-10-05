import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { describe, expect, it } from 'vitest'

import { POWERSHELL_PATH, RELAUNCH_WAITER_SCRIPT } from './relaunch-waiter'

const scriptPath = path.resolve(__dirname, '..', '..', 'scripts', RELAUNCH_WAITER_SCRIPT)
const quote = (value: string) => `'${value.replaceAll("'", "''")}'`

describe.skipIf(process.platform !== 'win32')('real PowerShell waiter control flow', () => {
  it.each([false, true])(
    'activates only after parent exit and package change (%s)',
    unavailable => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), 'relaunch-control-'))
      const ready = path.join(root, 'ready.txt')
      const activation = path.join(root, 'activation.txt')
      const wrapper = path.join(root, 'run.ps1')

      fs.writeFileSync(
        wrapper,
        `
$global:checks = 0
$global:birth = Get-Date
$global:clock = $global:birth
function Get-Date { return $global:clock }
function Start-Sleep { param($Milliseconds); $global:clock = $global:clock.AddSeconds(1) }
function Get-Process {
  param($Id, $ErrorAction)
  $global:checks++
  if ($global:checks -le 6) { return [pscustomobject]@{ StartTime = $global:birth } }
  return $null
}
function Get-AppxPackage {
  param($Name)
  ${unavailable ? 'return $null' : "return [pscustomobject]@{ Version = $(if ($global:checks -gt 6) { '2.0.0.0' } else { '1.0.0.0' }); PackageFamilyName = 'audit-only' }"}
}
function Get-AppxPackageManifest {
  return [pscustomobject]@{ Package = [pscustomobject]@{ Applications = [pscustomobject]@{ Application = [pscustomobject]@{ Id = 'Hermes' } } } }
}
function Start-Process {
  param($FilePath)
  if ($global:checks -le 6) { throw 'parent is still alive' }
  Set-Content -LiteralPath ${quote(activation)} -Value $FilePath
}
$birthMs = ([DateTimeOffset]$global:birth).ToUnixTimeMilliseconds()
& ${quote(scriptPath)} -ProcessId $PID -ProcessStartTimeMs $birthMs -IdentityName audit-only -ReadyFile ${quote(ready)} -TimeoutSeconds 10 -PollMillis 1
exit $LASTEXITCODE
`,
        'utf8'
      )

      try {
        const result = spawnSync(POWERSHELL_PATH, ['-NoProfile', '-NonInteractive', '-File', wrapper], {
          cwd: root,
          encoding: 'utf8',
          windowsHide: true,
          timeout: 30_000
        })

        expect(result.status, result.stdout + result.stderr).toBe(unavailable ? 2 : 0)
        expect(fs.existsSync(ready)).toBe(false)
        expect(fs.existsSync(activation)).toBe(!unavailable)

        if (!unavailable) {
          expect(fs.readFileSync(activation, 'utf8').trim()).toBe('shell:AppsFolder\\audit-only!Hermes')
        }
      } finally {
        fs.rmSync(root, { recursive: true, force: true })
      }
    },
    40_000
  )
})
