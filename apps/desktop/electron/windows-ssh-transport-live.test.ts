import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

import { encodedPowerShellPayload, powerShellBufferedStdinCommand, psLiteral } from './windows-remote-lifecycle'

// Live SSH is opt-in: CI without a Windows LAN host still runs the local
// PowerShell transport regression in atomic-stdin.test.ts.
const target = process.env.TEST_WINDOWS_SSH_TARGET
const key = process.env.TEST_WINDOWS_SSH_KEY

const binary =
  process.env.TEST_WINDOWS_SSH_BINARY ||
  path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')

test.skipIf(process.platform !== 'win32' || !target || !key)(
  'native Windows SSH preserves a long compressed script and Unicode JSON byte-for-byte',
  () => {
    // A Git/MSYS SSH client has a separate argv rewrite bug (#134949).
    assert.ok(
      !fs.readFileSync(binary).toString('latin1').toLowerCase().includes('msys-2.0.dll'),
      'use a native SSH client'
    )
    const expected = { message: 'José 李', padding: 'long-script-padding'.repeat(1500) }
    const script = `Write-Output ${psLiteral(JSON.stringify(expected))}`
    assert.ok(Buffer.byteLength(script, 'utf8') > 8191)
    const command = powerShellBufferedStdinCommand()
    assert.ok(command.length < 8191)
    assert.match(command, /-EncodedCommand\s+[A-Za-z0-9+/=]+$/)

    const output = execFileSync(
      binary,
      [
        '-T',
        '-o',
        'BatchMode=yes',
        '-o',
        'ConnectTimeout=8',
        '-o',
        'StrictHostKeyChecking=yes',
        '-o',
        'UpdateHostKeys=no',
        '-i',
        key!,
        target!,
        command
      ],
      {
        input: encodedPowerShellPayload(script),
        encoding: 'utf8',
        timeout: 20000,
        stdio: 'pipe'
      }
    )

    assert.deepEqual(JSON.parse(output.trim()), expected)
  }
)
