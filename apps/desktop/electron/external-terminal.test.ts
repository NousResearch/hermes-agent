import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  authInTerminalArgs,
  authPrepareArgs,
  buildTerminalScript,
  posixQuote,
  resolveTerminalLaunch,
  terminalScriptEnv,
  terminalScriptExtension,
  tuiResumeArgs,
  windowsQuote
} from './external-terminal'

const never = () => null
const always = (command: string) => `/usr/bin/${command}`

test('tuiResumeArgs resumes the session in the TUI', () => {
  assert.deepEqual(tuiResumeArgs('20260814_101010_abc123'), ['--tui', '--resume', '20260814_101010_abc123'])
})

test('tuiResumeArgs pins the profile ahead of the mode flag', () => {
  assert.deepEqual(tuiResumeArgs('sess', 'work'), ['--profile', 'work', '--tui', '--resume', 'sess'])
})

test('authInTerminalArgs maps an allowlisted provider to a fixed argv', () => {
  assert.deepEqual(authInTerminalArgs('anthropic'), ['auth', 'add', 'anthropic'])
  assert.deepEqual(authInTerminalArgs('anthropic', 'Work'), ['--profile', 'work', 'auth', 'add', 'anthropic'])
  assert.deepEqual(authInTerminalArgs('anthropic', 'Not A Slug!'), ['auth', 'add', 'anthropic'])
})

test('authInTerminalArgs rejects any provider id off the allowlist', () => {
  for (const id of ['claude-code', 'qwen-oauth', 'anthropic; rm -rf ~', 'constructor', '__proto__', '', 42, null]) {
    assert.equal(authInTerminalArgs(id, 'work'), null)
  }
})

test('posixQuote survives embedded single quotes', () => {
  assert.equal(posixQuote("/tmp/o'brien"), `'/tmp/o'\\''brien'`)
})

test('windowsQuote doubles embedded quotes', () => {
  assert.equal(windowsQuote('C:\\a "b"'), '"C:\\a ""b"""')
})

test('terminalScriptEnv drops PATH in any casing and keeps the rest', () => {
  const env = terminalScriptEnv(
    { Path: 'C:\\junk', PATH: '/junk', PYTHONPATH: '/repo', PYTHONUTF8: '1' },
    '/home/b/.hermes'
  )

  assert.deepEqual(env, { PYTHONPATH: '/repo', PYTHONUTF8: '1', HERMES_HOME: '/home/b/.hermes' })
})

test('terminalScriptEnv skips empty values and an absent home', () => {
  assert.deepEqual(terminalScriptEnv({ PYTHONPATH: '' }), {})
})

test('buildTerminalScript execs the resolved runtime with its env', () => {
  const script = buildTerminalScript({
    args: ['-m', 'hermes_cli.main', '--tui', '--resume', 'sess'],
    command: '/home/b/.hermes/hermes-agent/venv/bin/python',
    cwd: "/home/b/o'brien",
    env: { PYTHONPATH: '/home/b/.hermes/hermes-agent' },
    platform: 'darwin'
  })

  assert.equal(
    script,
    [
      '#!/bin/sh',
      `cd '/home/b/o'\\''brien' || exit 1`,
      `export PYTHONPATH='/home/b/.hermes/hermes-agent'`,
      `exec '/home/b/.hermes/hermes-agent/venv/bin/python' '-m' 'hermes_cli.main' '--tui' '--resume' 'sess'`,
      ''
    ].join('\n')
  )
})

test('buildTerminalScript emits a cmd script on Windows', () => {
  const script = buildTerminalScript({
    args: ['--tui', '--resume', 'sess'],
    command: 'C:\\hermes\\venv\\Scripts\\hermes.exe',
    cwd: 'C:\\Users\\b',
    env: { PYTHONUTF8: '1' },
    platform: 'win32'
  })

  assert.deepEqual(script.split('\r\n'), [
    '@echo off',
    'cd /d "C:\\Users\\b"',
    'set "PYTHONUTF8=1"',
    '"C:\\hermes\\venv\\Scripts\\hermes.exe" "--tui" "--resume" "sess"',
    ''
  ])
})

test('terminalScriptExtension matches what the platform binds to a terminal', () => {
  assert.equal(terminalScriptExtension('darwin'), '.command')
  assert.equal(terminalScriptExtension('win32'), '.cmd')
  assert.equal(terminalScriptExtension('linux'), '.sh')
})

test('macOS opens the script with no -a so LaunchServices picks the user handler', () => {
  assert.deepEqual(resolveTerminalLaunch({ findOnPath: never, platform: 'darwin', scriptPath: '/tmp/x.command' }), {
    command: 'open',
    args: ['/tmp/x.command']
  })
})

test('Windows prefers Windows Terminal and falls back to a cmd console', () => {
  assert.deepEqual(
    resolveTerminalLaunch({
      findOnPath: command => (command === 'wt.exe' ? 'C:\\wt.exe' : null),
      platform: 'win32',
      scriptPath: 'C:\\x.cmd'
    }),
    { command: 'C:\\wt.exe', args: ['cmd.exe', '/k', 'C:\\x.cmd'] }
  )

  assert.deepEqual(resolveTerminalLaunch({ findOnPath: never, platform: 'win32', scriptPath: 'C:\\x.cmd' }), {
    command: 'cmd.exe',
    args: ['/c', 'start', '', 'cmd.exe', '/k', 'C:\\x.cmd']
  })
})

test("Linux leads with the user's x-terminal-emulator alternative", () => {
  assert.deepEqual(resolveTerminalLaunch({ findOnPath: always, platform: 'linux', scriptPath: '/tmp/x.sh' }), {
    command: '/usr/bin/x-terminal-emulator',
    args: ['-e', '/bin/sh', '/tmp/x.sh']
  })
})

test('Linux falls down the emulator ladder and omits a flagless terminal', () => {
  const onlyKitty = (command: string) => (command === 'kitty' ? '/usr/bin/kitty' : null)

  assert.deepEqual(resolveTerminalLaunch({ findOnPath: onlyKitty, platform: 'linux', scriptPath: '/tmp/x.sh' }), {
    command: '/usr/bin/kitty',
    args: ['/bin/sh', '/tmp/x.sh']
  })
})

test('Linux with no emulator installed reports no launch', () => {
  assert.equal(resolveTerminalLaunch({ findOnPath: never, platform: 'linux', scriptPath: '/tmp/x.sh' }), null)
})

test('tuiResumeArgs drops a profile value that is not a valid profile id', () => {
  // A non-slug (numeric roster id, display label) must never cross into the
  // TUI launch argv — the CLI used to str()-coerce it into profiles/0 (#88842).
  assert.deepEqual(tuiResumeArgs('sess', 0 as unknown as string), ['--tui', '--resume', 'sess'])
  assert.deepEqual(tuiResumeArgs('sess', ''), ['--tui', '--resume', 'sess'])
  assert.deepEqual(tuiResumeArgs('sess', 'Not A Slug!'), ['--tui', '--resume', 'sess'])
})

test('tuiResumeArgs normalizes a valid profile id like the CLI', () => {
  assert.deepEqual(tuiResumeArgs('sess', 'Work'), ['--profile', 'work', '--tui', '--resume', 'sess'])
})

test.skipIf(process.platform === 'win32')(
  'a held sign-in window waits after a failure and closes after success',
  () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hold-on-failure-'))

    const run = (command: string) => {
      const scriptPath = path.join(dir, `${command}.sh`)
      fs.writeFileSync(scriptPath, buildTerminalScript({ args: [], command, cwd: dir, holdOnFailure: true }))

      return spawnSync('/bin/sh', [scriptPath], { encoding: 'utf8', input: '\n' })
    }

    const failed = run('false')
    assert.equal(failed.status, 1)
    assert.match(failed.stdout, /Press Return to close/)

    const succeeded = run('true')
    assert.equal(succeeded.status, 0)
    assert.doesNotMatch(succeeded.stdout, /Press Return/)
  }
)

test('authPrepareArgs installs only what an allowlisted sign-in needs', () => {
  assert.deepEqual(authPrepareArgs('anthropic'), ['pm', 'install', '--extra', 'anthropic'])

  for (const id of ['claude-code', 'anthropic; rm -rf ~', '__proto__', '', 7, null]) {
    assert.equal(authPrepareArgs(id), null)
  }
})

test.skipIf(process.platform === 'win32')('a failed prepare step still runs the sign-in after it', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'prepare-step-'))
  const scriptPath = path.join(dir, 'login.sh')

  fs.writeFileSync(
    scriptPath,
    buildTerminalScript({ args: ['signed-in'], before: { args: [], command: 'false' }, command: 'echo', cwd: dir })
  )

  const result = spawnSync('/bin/sh', [scriptPath], { encoding: 'utf8' })
  assert.equal(result.status, 0)
  assert.equal(result.stdout.trim(), 'signed-in')
})

test('on Windows the prepare step returns to the sign-in even through a .cmd launcher', () => {
  const lines = buildTerminalScript({
    args: ['auth', 'add', 'anthropic'],
    before: { args: ['pm', 'install', '--extra', 'anthropic'], command: 'C:\\hermes\\bin\\hermes.cmd' },
    command: 'C:\\hermes\\bin\\hermes.cmd',
    cwd: 'C:\\Users\\b',
    platform: 'win32'
  }).split('\r\n')

  const prepare = lines.findIndex(line => line.includes('"pm"'))
  const login = lines.findIndex(line => line.includes('"auth"'))

  assert.ok(lines[prepare].startsWith('call '))
  assert.ok(prepare < login)
})
