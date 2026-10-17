import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { WINDOWS_PROFILE_LIST_PY, windowsInventoryScript } from './windows-remote-inventory'
import { atomicWindowsSpawn, atomicWindowsSpawnCommand, encodedPowerShell, powerShellStdinCommand, probeWindowsFiles, probeWindowsRemote } from './windows-remote-lifecycle'
import { WINDOWS_NO_REPARSE_PS, windowsHelperArgv, windowsRuntimeProgramArgv } from './windows-runtime-command'

const repo = path.resolve(import.meta.dirname, '../../..')
const python = process.env.HERMES_TEST_PYTHON || path.join(repo, '.venv/bin/python')
const runtime = { os: 'Windows', arch: 'AMD64', hermesHome: 'C:\\hermes', hermesPath: 'C:\\hermes\\bin\\hermes.exe', python: 'C:\\runtime\\python.exe' }
const bootstrap = "import os, sys, runpy; os.environ['HERMES_TEST_BOOT']='selected'; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"
const selected = { ...runtime, runtimeCommand: [python, '-I', '-c', bootstrap] }

function streamed(command: string, options: { stdinData?: string } = {}) {
  const token = command.match(/-EncodedCommand\s+(\S+)$/)?.[1]
  const wrapper = token ? Buffer.from(token, 'base64').toString('utf16le') : ''

  return wrapper.includes('ReadToEnd') ? Buffer.from(String(options.stdinData || '').trim(), 'base64').toString('utf16le') : wrapper
}

test('the canonical runtime prefix is actually executed, including bootstrap and Unicode arguments', () => {
  const program = "import json, os, sys; print(json.dumps([os.environ['HERMES_TEST_BOOT'], sys.argv[1], '$&'], ensure_ascii=False))"
  const argv = windowsRuntimeProgramArgv(selected, program)
  const result = spawnSync(argv[0], [...argv.slice(1), '用户/한글'], { encoding: 'utf8' })

  assert.equal(result.status, 0, result.stderr)
  assert.deepEqual(JSON.parse(result.stdout), ['selected', '用户/한글', '$&'])
  assert.ok(windowsHelperArgv(selected, 'inspect', [runtime.hermesPath]).includes('inspect'))
  assert.throws(() => windowsRuntimeProgramArgv({ ...runtime, runtimeCommand: [python, '-c', 'print(0)'] }, 'pass'), (e: any) => e.kind === 'update-required')
})

test('injected Python program survives the observed PowerShell 5.1 double-quote loss', () => {
  const program = `import json; print(json.dumps({'marker': '用户/quote"path'}, ensure_ascii=False))`
  const argv = windowsRuntimeProgramArgv(selected, program)
  const tail = argv.at(-1)!

  assert.match(tail, /exec\(bytes\.fromhex\('[0-9a-f]+'\)\.decode\('utf-8'\)\)$/)
  assert.equal(tail.startsWith(bootstrap.split('runpy.run_module')[0]), true)
  // The transport failure was observed on real Windows by Polaris; this is its loss-channel regression,
  // not a claim to execute PowerShell 5.1 on a Mac.
  const result = spawnSync(argv[0], [...argv.slice(1, -1), tail.replace(/"/g, '')], { encoding: 'utf8' })

  assert.equal(result.status, 0, result.stderr)
  assert.deepEqual(JSON.parse(result.stdout), { marker: '用户/quote"path' })
})

test('PATH bin launcher is resolved independently of its absent sibling Python, after the marker gate', async () => {
  const calls: string[] = []

  const ssh = { exec: async (command: string, options: { stdinData?: string }) => {
    const script = streamed(command, options)
    calls.push(script)

    if (script.includes('Get-Command hermes.exe')) {
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

  const blocked = { exec: async (command: string, options: { stdinData?: string }) => {
    const script = streamed(command, options)

    if (script.includes('Get-Command hermes.exe')) {return JSON.stringify({ ...runtime, launcherPath: runtime.hermesPath })}

    if (script.includes('.hermes-update-in-progress')) {return 'HELD'}
    imported = true

    throw new Error('must not import')
  } }

  await assert.rejects(() => probeWindowsRemote(blocked), (e: any) => e.kind === 'update-in-progress')
  assert.equal(imported, false)

  const invalid = { exec: async (command: string, options: { stdinData?: string }) => {
    const script = streamed(command, options)

    if (script.includes('Get-Command hermes.exe')) {return JSON.stringify({ ...runtime, launcherPath: runtime.hermesPath })}

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}

    return '{}'
  } }

  await assert.rejects(() => probeWindowsRemote(invalid), (e: any) => e.kind === 'update-required')
})

test('atomic spawn streams its whole script and explicitly pipes Unicode caller JSON after draining stdin', async () => {
  const payload = JSON.stringify({ ownershipId: 'a'.repeat(32), spawnNonce: 'b'.repeat(16), profile: 'bob', hermesPath: 'C:\\用户\\hermes.exe' })
  let observed = ''

  const ssh = { exec: async (command: string, options: { stdinData?: string }) => {
    assert.ok(command.length < 1024)
    observed = streamed(command, options)
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

  const ssh = { exec: async (command: string, options: { stdinData?: string }) => {
    const script = streamed(command, options)

    if (script.includes('Get-Command hermes.exe')) {return JSON.stringify({ ...runtime, launcherPath: runtime.hermesPath })}

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

test('canonical profile predicate really filters tombstones, markerless dirs and rollback junk under a pinned home', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-canonical-inventory-'))
  const root = path.join(home, 'state-root')
  const profiles = path.join(root, 'profiles')
  fs.mkdirSync(path.join(profiles, '.deleted'), { recursive: true })
  fs.writeFileSync(path.join(root, 'config.yaml'), '')

  for (const name of ['bob', 'win-research', 'ghost', 'tombstoned', '.rollback-1', 'bad name']) {
    fs.mkdirSync(path.join(profiles, name))

    if (name !== 'ghost') {fs.writeFileSync(path.join(profiles, name, 'config.yaml'), '')}
  }

  fs.writeFileSync(path.join(profiles, '.deleted', 'tombstoned'), '')

  try {
    for (const hermesHome of [root, path.join(profiles, 'bob')]) {
      const result = spawnSync(python, ['-c', WINDOWS_PROFILE_LIST_PY], {
        cwd: repo, encoding: 'utf8', env: { PATH: process.env.PATH || '', HOME: home, HERMES_HOME: hermesHome, PYTHONPATH: repo }
      })

      assert.equal(result.status, 0, result.stderr)
      assert.deepEqual(JSON.parse(result.stdout).profiles, ['default', 'bob', 'win-research'])
    }

    assert.equal(fs.existsSync(path.join(root, 'install_id')), false, 'inventory must not mint an identity')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('inventory programs guard ancestors, keep install-root anchoring and open identity no-follow', () => {
  const roster = windowsInventoryScript(runtime, 'profiles')
  const identity = windowsInventoryScript({ ...runtime, hermesHome: 'C:\\hermes\\profiles\\bob' }, 'install-id')

  assert.match(roster, /Assert-NoReparse \$profilesDir \$true/)
  assert.match(roster, /_iter_named_profile_dirs/)
  assert.match(identity, /Split-Path -Leaf \$parent\) -ieq "profiles"/)
  assert.match(identity, /GetFinalPathNameByHandle/)
  assert.match(identity, /0x00200000/)
  assert.doesNotMatch(identity, /read_or_create_install_id|WriteAllText|CreateDirectory/)
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('real local PowerShell bootstrap preserves Unicode and pipes JSON after script stdin is consumed', () => {
  const pwsh = String(process.env.HERMES_TEST_PWSH)
  const payload = JSON.stringify({ marker: '用户/한글', length: 12000 })
  const script = `$OutputEncoding=[Text.UTF8Encoding]::new($false);${"# padding\r\n".repeat(1500)}'${payload}' | & '${python}' -c 'import json,sys; print(json.dumps(json.load(sys.stdin),ensure_ascii=False))'`
  const token = powerShellStdinCommand().split(' ').at(-1)!

  const result = spawnSync(pwsh, ['-NoProfile', '-NonInteractive', '-EncodedCommand', token], {
    input: encodedPowerShell(script), encoding: 'utf8', timeout: 20000
  })

  assert.equal(result.status, 0, result.stderr)
  assert.deepEqual(JSON.parse(result.stdout.trim()), JSON.parse(payload))
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('real outer PowerShell rejects the old Invoke wrapper and accepts the encoded bootstrap', () => {
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
    const green = run(powerShellStdinCommand())

    assert.notEqual(red.status, 0)
    assert.match(red.stderr, /Invoke/)
    assert.equal(green.status, 0, green.stderr)
    assert.equal(green.stdout.trim(), marker)
  } finally {
    fs.rmSync(tools, { recursive: true, force: true })
  }
})

test.runIf(Boolean(process.env.HERMES_TEST_PWSH))('real PowerShell parser accepts emitted probe, inventory, spawn and no-follow C# programs', async () => {
  const pwsh = String(process.env.HERMES_TEST_PWSH)
  const scripts = [windowsInventoryScript(runtime, 'profiles'), windowsInventoryScript(runtime, 'install-id')]
  scripts.push(streamed(atomicWindowsSpawnCommand(runtime)))
  await probeWindowsFiles({ exec: async (command: string, options: { stdinData?: string }) => {
    scripts.push(streamed(command, options))

    return JSON.stringify(runtime)
  } })
  const list = scripts.map(script => `'${encodedPowerShell(script)}'`).join(',')
  const compile = windowsInventoryScript(runtime, 'install-id').match(/Add-Type -TypeDefinition @'[\s\S]*?'@/)?.[0]
  assert.ok(compile)
  const program = `$errorsFound=@();foreach($b in @(${list})){$text=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($b));$tokens=$null;$parseErrors=$null;[Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$parseErrors)|Out-Null;$errorsFound+=@($parseErrors)};${compile};if($errorsFound.Count){$errorsFound|ForEach-Object {$_.Message};exit 1};'HERMES_PS_PROGRAMS_PARSE_OK'`

  const result = spawnSync(pwsh, ['-NoProfile', '-NonInteractive', '-EncodedCommand', powerShellStdinCommand().split(' ').at(-1)!], {
    input: encodedPowerShell(program), encoding: 'utf8', timeout: 20000
  })

  assert.equal(result.status, 0, result.stderr + result.stdout)
  assert.equal(result.stdout.trim(), 'HERMES_PS_PROGRAMS_PARSE_OK')
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
