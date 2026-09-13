import { execFileSync } from 'node:child_process'

import { assert, describe, test } from 'vitest'

import { atomicWindowsSpawnCommand } from './windows-remote-lifecycle'

function buildSpawnCommand() {
  return atomicWindowsSpawnCommand(
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

function decodeSpawnScript(command: string) {
  // Extract the EncodedCommand payload and decode it back to the script text
  const encoded = command.match(/-EncodedCommand\s+([A-Za-z0-9+/=]+)\s*$/)?.[1]
  assert.ok(encoded, 'command must carry an EncodedCommand payload')

  return Buffer.from(encoded, 'base64').toString('utf16le')
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
  const script = decodeSpawnScript(buildSpawnCommand())

  // The lock must be piped into the helper's stdin, not passed as argv
  assert.match(script, /\$lock\|& [^|]+ 'write-lock' '0123456789abcdef0123456789abcdef'\|Out-Null/)
  assert.notMatch(script, /'write-lock' '[0-9a-f]{32}' \$lock/)

  // The PowerShell 5.1 pipe must arrive as UTF-8, not the console code page
  assert.match(script, /\$OutputEncoding=\[Text\.UTF8Encoding\]::new\(\$false\)/)

  // Write-Progress noise (CLIXML on stderr) must be silenced
  assert.match(script, /\$ProgressPreference="SilentlyContinue"/)
})

describe.skipIf(!hasPowershell())('atomic spawn script: live PowerShell parse', () => {
  test('generated script parses without syntax errors', () => {
    const script = decodeSpawnScript(buildSpawnCommand())

    const parseCheck = execFileSync(
      POWERSHELL_BIN,
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseInput(@'\n${script}\n'@\n,[ref]$t,[ref]$e) > $null; if($e.Count -gt 0){$e | ForEach-Object {$_.Message}; exit 1}; 'PARSE_OK'`
      ],
      { encoding: 'utf8', timeout: 60000 }
    )

    assert.match(parseCheck, /PARSE_OK/)
  })
})
