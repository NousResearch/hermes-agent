import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
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

test('posixQuote survives embedded single quotes', () => {
  assert.equal(posixQuote("/tmp/o'brien"), `'/tmp/o'\\''brien'`)
})

test('windowsQuote doubles embedded quotes', () => {
  assert.equal(windowsQuote('C:\\a "b"'), '"C:\\a ""b"""')
})

test('terminalScriptEnv drops PATH in any casing and keeps the rest', () => {
  const env = terminalScriptEnv({ Path: 'C:\\junk', PATH: '/junk', PYTHONPATH: '/repo', PYTHONUTF8: '1' }, '/home/b/.hermes')

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
