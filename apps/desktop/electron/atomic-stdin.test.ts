import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'

import { describe, test } from 'vitest'

import { atomicWindowsSpawnScript, probeWindowsRemote } from './windows-remote-lifecycle'

function buildSpawnCommand() {
  return atomicWindowsSpawnScript(
    {
      // SANDBOX: temp HERMES_HOME resolved by the helper itself
      hermesHome: 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\ssh-probe-home',
      python: 'C:\\Users\\TestUser\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\python.exe'
    },
    {
      ownershipId: '0123456789abcdef0123456789abcdef',
      spawnNonce: '0123456789abcdef',
      profile: 'default',
      hermesPath: 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\ssh-probe-home\\fake-hermes.exe',
      hermesHome: 'C:\\Users\\TestUser\\AppData\\Local\\Temp\\ssh-probe-home',
      tokenFingerprint: 'a'.repeat(32),
      startedAt: '2026-08-27T14:00:00.000Z'
    }
  )
}

// powershell.exe only exists on Windows; GitHub-hosted ubuntu runners ship
// pwsh instead, and macOS runners ship neither. Probe for a usable binary so
// the live parse check below runs wherever possible and skips cleanly
// everywhere else (repo convention: platform-gated live tests skip, never fail).
const POWERSHELL_BIN = process.platform === 'win32' ? 'powershell.exe' : 'pwsh'

function hasPowershell() {
  try {
    execFileSync(POWERSHELL_BIN, ['-NoProfile', '-NonInteractive', '-Command', '$null'], {
      stdio: 'ignore',
      timeout: 30000
    })

    return true
  } catch {
    return false
  }
}

test('atomic spawn script: write-lock via stdin pipe (static shape)', () => {
  const script = buildSpawnCommand()

  // The lock must be piped into the helper's stdin, not passed as argv
  assert.match(script, /\$lock\s*\|\s*& [^|]+ 'write-lock' '0123456789abcdef0123456789abcdef'\s*\|\s*Out-Null/)
  assert.doesNotMatch(script, /'write-lock' '[0-9a-f]{32}' \$lock/)

  // The PowerShell 5.1 pipe must arrive as UTF-8, not the console code page
  assert.match(script, /\$OutputEncoding=\[Text\.UTF8Encoding\]::new\(\$false\)/)

  // Write-Progress noise (CLIXML on stderr) must be silenced
  assert.match(script, /\$ProgressPreference="SilentlyContinue"/)
})

describe.skipIf(!hasPowershell())('atomic spawn script: live PowerShell parse', () => {
  test('generated script parses without syntax errors', () => {
    const script = buildSpawnCommand()

    const parseCheck = execFileSync(
      POWERSHELL_BIN,
      [
        '-NoProfile',
        '-NonInteractive',
        '-EncodedCommand',
        Buffer.from(
          '$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseInput([Console]::In.ReadToEnd(),[ref]$t,[ref]$e)>$null;if($e.Count -gt 0){$e|ForEach-Object {$_.Message};exit 1};Write-Output PARSE_OK',
          'utf16le'
        ).toString('base64')
      ],
      { encoding: 'utf8', input: script, timeout: 60000, stdio: 'pipe' }
    )

    assert.match(parseCheck, /PARSE_OK/)
  })
})

describe.skipIf(!hasPowershell())('buffered PowerShell transport: live execution', () => {
  test('executes the complete script from ASCII stdin with UTF-8 output and exit status', async () => {
    let command = ''
    await probeWindowsRemote({
      exec: async value => {
        command = value

        return '{"os":"Windows"}'
      }
    })
    assert.ok(command.length < 8191)
    const script = 'Write-Output "José"\n#' + 'long-script-padding'.repeat(500)

    const output = execFileSync(POWERSHELL_BIN, command.split(' ').slice(1), {
      input: Buffer.from(script, 'utf8').toString('base64'),
      encoding: 'utf8',
      timeout: 30000,
      stdio: 'pipe'
    })

    assert.equal(output.trim(), 'José')
    assert.throws(
      () =>
        execFileSync(POWERSHELL_BIN, command.split(' ').slice(1), {
          input: Buffer.from('exit 23', 'utf8').toString('base64'),
          timeout: 30000,
          stdio: 'pipe'
        }),
      (error: any) => error.status === 23
    )
  })
})
