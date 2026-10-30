import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { listWindowsRemoteProfiles, readWindowsRemoteInstallId } from './windows-remote-inventory'
import { atomicWindowsSpawn, atomicWindowsSpawnCommand, compressedPowerShellCommand, decodePowerShellCommand, encodedPowerShell, probeWindowsFiles, probeWindowsRemote } from './windows-remote-lifecycle'
import { WINDOWS_NO_REPARSE_PS } from './windows-runtime-command'

const python = process.env.HERMES_TEST_PYTHON || process.env.HERMES_PYTHON || 'python'
const runtime = { os: 'Windows', arch: 'AMD64', hermesHome: 'C:\\hermes', hermesPath: 'C:\\hermes\\bin\\hermes.exe', python: 'C:\\runtime\\python.exe' }
const bootstrap = "import os, sys, runpy; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"

function streamed(command: string) {
  return decodePowerShellCommand(command)
}

test('PATH bin launcher is resolved independently of its absent sibling Python, after the marker gate', async () => {
  const calls: string[] = []

  const ssh = { exec: async (command: string) => {
    const script = streamed(command)
    calls.push(script)

    if (script.includes('Get-Command hermes -CommandType Application')) {
      return JSON.stringify({ ...runtime, python: '', launcherPath: runtime.hermesPath })
    }

    if (script.includes('.hermes-update-in-progress')) {
      return 'CLEAR'
    }

    assert.match(script, /--print-runtime-command/)

    return JSON.stringify({ python: runtime.python, runtimeCommand: [runtime.python, '-I', '-c', bootstrap] })
  } }

  const result = await probeWindowsRemote(ssh)

  assert.equal(result.hermesPath, runtime.hermesPath)
  assert.equal(result.python, runtime.python)
  assert.ok(calls[0].includes('Test-Path -LiteralPath $candidatePython -PathType Leaf'))
  assert.ok(calls[1].includes('.hermes-update-in-progress'))
  assert.ok(calls[2].includes('Assert-NoReparse ([string]$command[0]) $false'))
})

test('held marker stops launcher imports and unrecognized runtime capabilities fail clearly', async () => {
  let imported = false

  const blocked = { exec: async (command: string) => {
    const script = streamed(command)

    if (script.includes('Get-Command hermes -CommandType Application')) {return JSON.stringify({ ...runtime, launcherPath: runtime.hermesPath })}

    if (script.includes('.hermes-update-in-progress')) {return 'HELD'}
    imported = true

    throw new Error('must not import')
  } }

  await assert.rejects(() => probeWindowsRemote(blocked), (e: any) => e.kind === 'update-in-progress')
  assert.equal(imported, false)

  const invalid = { exec: async (command: string) => {
    const script = streamed(command)

    if (script.includes('Get-Command hermes -CommandType Application')) {return JSON.stringify({ ...runtime, launcherPath: runtime.hermesPath })}

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}

    return '{}'
  } }

  await assert.rejects(() => probeWindowsRemote(invalid), (e: any) => e.kind === 'update-required')
})

test('atomic spawn streams its whole script and explicitly pipes Unicode caller JSON after draining stdin', async () => {
  const payload = JSON.stringify({ ownershipId: 'a'.repeat(32), spawnNonce: 'b'.repeat(16), profile: 'bob', hermesPath: 'C:\\用户\\hermes.exe' })
  let observed = ''

  const ssh = { exec: async (command: string, options: { stdinData?: string } = {}) => {
    assert.ok(command.length < 8191)
    assert.equal(options.stdinData, undefined)
    observed = streamed(command)
    assert.match(observed, /\$OutputEncoding=\[Text.UTF8Encoding\]::new\(\$false\)/)
    assert.ok(observed.includes(payload))
    assert.match(observed, /' \| & .*windows_ssh_runtime.*spawn/)
    assert.match(observed, /FileShare\]::None/)

    return JSON.stringify({ pid: 42, creationTimeNs: '1234567890123456789' })
  } }

  const result = await atomicWindowsSpawn(ssh, runtime, payload)

  assert.equal(result.pid, 42)
  assert.ok(observed.length > 1000)
})

test('Connect refuses an older helper before token upload or backend spawn on a boot-selected runtime', async () => {
  const { connectWindowsRemote } = await import('./windows-remote-lifecycle')
  let writes = 0

  const ssh = { exec: async (command: string) => {
    const script = streamed(command)

    if (script.includes('Get-Command hermes -CommandType Application')) {return JSON.stringify({ ...runtime, launcherPath: runtime.hermesPath })}

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}

    if (script.includes('--print-runtime-command')) {return JSON.stringify({ python: runtime.python, runtimeCommand: [runtime.python, '-I', '-c', bootstrap] })}

    if (script.includes("'inspect'")) {return JSON.stringify({ supported: true, path: runtime.hermesPath })}
    writes += 1

    throw new Error('write must not be reached')
  } }

  await assert.rejects(() => connectWindowsRemote({
    ssh, ownershipId: 'a'.repeat(32), pickLocalPort: async () => 4444, forward: async () => {},
    cancelForward: async () => {}, waitForHermes: async () => {}, probeReuseProof: async () => 'authenticated-ok'
  }), (error: any) => error.kind === 'update-required')
  assert.equal(writes, 0)
})

test('inventory reads profiles and install identity through the runtime helper after the marker gate', async () => {
  const calls: string[] = []

  const ssh = { exec: async (command: string, options: { stdinData?: string } = {}) => {
    assert.equal(options.stdinData, undefined)
    const script = streamed(command)
    calls.push(script)

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}
    assert.match(script, /'--run-module' 'hermes_cli\.windows_ssh_runtime' 'inventory'/)

    return JSON.stringify({ profiles: ['default', 'bob'], installId: 'A'.repeat(32) })
  } }

  assert.deepEqual(await listWindowsRemoteProfiles(ssh, runtime), ['default', 'bob'])
  assert.equal(await readWindowsRemoteInstallId(ssh, runtime), 'a'.repeat(32))
  assert.deepEqual(calls.map(script => script.includes('.hermes-update-in-progress')), [true, false, true, false])
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('real local PowerShell runs a long compressed script, preserving Unicode and piped JSON', () => {
  const pwsh = String(process.env.HERMES_TEST_PWSH)
  const payload = JSON.stringify({ marker: '用户/한글', length: 12000 })
  const script = `$OutputEncoding=[Text.UTF8Encoding]::new($false);${"# padding\r\n".repeat(1500)}'${payload}' | & '${python}' -c 'import json,sys; print(json.dumps(json.load(sys.stdin),ensure_ascii=False))'`
  const command = compressedPowerShellCommand(script)
  assert.ok(command.length < 8191)

  const result = spawnSync(pwsh, ['-NoProfile', '-NonInteractive', '-EncodedCommand', command.split(' ').at(-1)!], {
    encoding: 'utf8', timeout: 20000
  })

  assert.equal(result.status, 0, result.stderr)
  assert.deepEqual(JSON.parse(result.stdout.trim()), JSON.parse(payload))
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('real outer PowerShell rejects the old Invoke wrapper and accepts the compressed command', () => {
  const pwsh = String(process.env.HERMES_TEST_PWSH)
  const tools = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-outer-powershell-'))
  const marker = 'HERMES_LOCAL_OUTER_SHELL_OK_用户'
  const payload = encodedPowerShell(`Write-Output '${marker}'`)
  const old = 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command [ScriptBlock]::Create([Text.Encoding]::Unicode.GetString([Convert]::FromBase64String([Console]::In.ReadToEnd()))).Invoke()'
  fs.symlinkSync(pwsh, path.join(tools, 'powershell.exe'))

  try {
    const run = (command: string) => spawnSync(pwsh, ['-NoProfile', '-NonInteractive', '-Command', command], {
      input: payload, encoding: 'utf8', timeout: 20000, env: { ...process.env, PATH: `${tools}${path.delimiter}${process.env.PATH}` }
    })

    const red = run(old)
    const green = run(compressedPowerShellCommand(`Write-Output '${marker}'`))

    assert.notEqual(red.status, 0)
    assert.match(red.stderr, /Invoke/)
    assert.equal(green.status, 0, green.stderr)
    assert.equal(green.stdout.trim(), marker)
  } finally {
    fs.rmSync(tools, { recursive: true, force: true })
  }
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('real PowerShell parser accepts emitted probe and spawn programs', async () => {
  const pwsh = String(process.env.HERMES_TEST_PWSH)
  const scripts = [streamed(atomicWindowsSpawnCommand(runtime))]
  await probeWindowsFiles({ exec: async (command: string) => {
    scripts.push(streamed(command))

    return JSON.stringify(runtime)
  } })
  const list = scripts.map(script => `'${encodedPowerShell(script)}'`).join(',')
  const program = `$errorsFound=@();foreach($b in @(${list})){$text=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($b));$tokens=$null;$parseErrors=$null;[Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$parseErrors)|Out-Null;$errorsFound+=@($parseErrors)};if($errorsFound.Count){$errorsFound|ForEach-Object {$_.Message};exit 1};'HERMES_PS_PROGRAMS_PARSE_OK'`

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ps-parse-'))
  const file = path.join(dir, 'parse.ps1')
  fs.writeFileSync(file, `\uFEFF${program}`, 'utf8')

  try {
    const result = spawnSync(pwsh, ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', file], { encoding: 'utf8', timeout: 20000 })

    assert.equal(result.status, 0, result.stderr + result.stdout)
    assert.equal(result.stdout.trim(), 'HERMES_PS_PROGRAMS_PARSE_OK')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('no-reparse guard checks a leaf file and every ancestor, not FileInfo.Parent', () => {
  const pwsh = String(process.env.HERMES_TEST_PWSH)
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-no-reparse-'))
  fs.mkdirSync(path.join(root, 'real'))
  fs.writeFileSync(path.join(root, 'real', 'leaf'), 'fixture')
  fs.symlinkSync(path.join(root, 'real'), path.join(root, 'linked'))

  try {
    const file = path.join(root, 'linked', 'leaf').replace(/'/g, "''")
    const program = `${WINDOWS_NO_REPARSE_PS};Assert-NoReparse '${file}' $false`
    const result = spawnSync(pwsh, ['-NoProfile', '-NonInteractive', '-EncodedCommand', encodedPowerShell(program)], { encoding: 'utf8', timeout: 20000 })

    assert.notEqual(result.status, 0, 'a file beneath a reparse directory must be refused')
    assert.match(result.stderr, /reparse point/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
