import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  assertWindowsRemoteInstallUpdateClear,
  atomicWindowsSpawnScript,
  buildWindowsInteractiveCommand,
  helperCommandScript,
  powerShellExecPlan,
  probeWindowsRemote
} from './windows-remote-lifecycle'

// Windows OpenSSH's exec channel truncates long command payloads (empirically
// ~2,544 chars of base64); the tail is dropped silently, so PowerShell parses
// half a script and fails with "Missing closing '}'" (#118987).
const SSH_EXEC_PAYLOAD_CEILING = 2544

function encodedToken(command: string): string {
  const match = /-EncodedCommand\s+(\S+)\s*$/.exec(command)
  assert.ok(match, `expected an -EncodedCommand payload: ${command.slice(0, 120)}`)

  return match[1]
}

function sshWith(exec) {
  return { exec }
}

// Local twin of the helper in windows-remote-lifecycle.test.ts: the script may
// ride stdin once it outgrows the SSH exec transport, so a fake receiver has
// to look at whichever of the two carries the real script.
function decodeExec(command: string, stdinData?: unknown): string {
  const fromCommand = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
  const firstStdinLine = String(stdinData ?? '').split('\n', 1)[0]

  if (!firstStdinLine) {
    return fromCommand
  }

  const fromStdin = Buffer.from(firstStdinLine, 'base64').toString('utf16le')

  return fromStdin.length > fromCommand.length ? fromStdin : fromCommand
}

test('every Windows PowerShell exec payload stays under the SSH exec transport ceiling', async () => {
  // Capture what actually crosses the exec channel: the command line plus any
  // stdin payload. The builders below are only inputs — what matters is the
  // plan the shared decision point produces for each of them.
  const crossings: Array<{ label: string; commandChars: number; stdinChars: number }> = []

  const ssh = sshWith(async (command: string, options: any = {}) => {
    // The script may ride stdin once it outgrows the transport, so the fake
    // has to look at what the receiver would actually run, not just the
    // command line.
    const script = decodeExec(command, options.stdinData)

    crossings.push({
      label: script.includes('Get-Command hermes.exe') ? 'platform probe' : 'update marker probe',
      commandChars: encodedToken(command).length,
      stdinChars: String(options.stdinData || '').length
    })

    return script.includes('Get-Command hermes.exe')
      ? JSON.stringify({
          os: 'Windows',
          arch: 'AMD64',
          hermesHome: 'C:\\Users\\alice\\.hermes',
          hermesPath: 'C:\\Hermes\\hermes.exe',
          python: 'C:\\Hermes\\python.exe'
        })
      : 'CLEAR'
  })

  await assertWindowsRemoteInstallUpdateClear(ssh as any, 'C:\\Users\\alice\\.hermes\\profiles\\research')
  await probeWindowsRemote(ssh as any, 'C:\\Users\\alice\\.hermes\\hermes.exe')

  const oversized = crossings.filter(row => row.commandChars > SSH_EXEC_PAYLOAD_CEILING)

  assert.deepEqual(
    oversized,
    [],
    `SSH exec payloads over the ${SSH_EXEC_PAYLOAD_CEILING}-char ceiling get truncated by Windows OpenSSH: ${JSON.stringify(oversized)}`
  )

  // Every remaining Windows PowerShell builder must route through the same
  // decision point, so an over-long script there also lands on stdin instead
  // of silently losing its tail on the command line.
  const runtime = {
    hermesHome: 'C:\\Users\\alice\\.hermes',
    python: 'C:\\Users\\alice\\.hermes\\venv\\Scripts\\python.exe'
  }

  const builtScripts: Array<[string, string]> = [
    [
      'atomic spawn (reserved)',
      atomicWindowsSpawnScript(runtime, {
        ownershipId: '0123456789abcdef0123456789abcdef',
        spawnNonce: '0123456789abcdef',
        profile: 'default',
        hermesPath: 'C:\\Hermes\\hermes.exe',
        hermesHome: 'C:\\Users\\alice\\.hermes',
        tokenFingerprint: 'a'.repeat(32),
        startedAt: '2026-07-14T00:00:00.000Z'
      })
    ],
    ['atomic spawn (unreserved)', atomicWindowsSpawnScript(runtime)],
    ['remote helper', helperCommandScript(runtime, 'read-lock', ['0123456789abcdef0123456789abcdef'])],
    ['interactive terminal', buildWindowsInteractiveCommand('C:\\Users\\alice\\work')]
  ]

  for (const [label, script] of builtScripts) {
    const plan = powerShellExecPlan(script)

    assert.ok(
      encodedToken(plan.command).length <= SSH_EXEC_PAYLOAD_CEILING,
      `${label}: command-line payload must fit the transport, or Windows OpenSSH drops the tail`
    )
    // And the script itself must still be recoverable from what crosses.
    const firstStdinLine = String(plan.stdinData || '').split('\n', 1)[0]

    const decoded = firstStdinLine
      ? Buffer.from(firstStdinLine, 'base64').toString('utf16le')
      : Buffer.from(encodedToken(plan.command), 'base64').toString('utf16le')

    assert.ok(decoded.includes(script), `${label}: script must arrive whole`)
  }
})

test('an over-long PowerShell script is shipped through the exec channel without truncation', () => {
  // A payload far past the transport ceiling must survive the channel: either
  // it is chunked, or it is moved off the command line onto stdin.
  const script = '$ErrorActionPreference="Stop"' + ';Write-Output ("' + 'x'.repeat(6000) + '".Length)'
  const plan = powerShellExecPlan(script)

  // Whatever the transport, the command line itself must fit the ceiling.
  // When the script rides stdin the command line is only the small consumer,
  // so this holds either way — that is the invariant Windows OpenSSH needs.
  assert.ok(
    encodedToken(plan.command).length <= SSH_EXEC_PAYLOAD_CEILING,
    'the command-line payload must fit the transport, or Windows OpenSSH drops the tail'
  )

  // Whatever the transport, the script must be recoverable from what crossed.
  const firstStdinLine = String(plan.stdinData || '').split('\n', 1)[0]

  const decoded = firstStdinLine
    ? Buffer.from(firstStdinLine, 'base64').toString('utf16le')
    : Buffer.from(encodedToken(plan.command), 'base64').toString('utf16le')

  assert.ok(decoded.includes(script), 'the script must arrive whole; a truncated tail is the bug')
})
