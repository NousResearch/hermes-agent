import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import {
  buildWindowsCliUpdateScript,
  resolveHermesCliBinary,
  shouldUseWindowsCliUpdateHandoff
} from './windows-cli-update-handoff'

test('resolveHermesCliBinary prefers Windows venv Scripts hermes.exe', () => {
  const root = 'C:\\Users\\apo\\AppData\\Local\\hermes\\hermes-agent'
  const expected = path.win32.join(root, 'venv', 'Scripts', 'hermes.exe')
  const seen: string[] = []

  const resolved = resolveHermesCliBinary(root, {
    isWindows: true,
    fileExists: candidate => {
      seen.push(candidate)

      return candidate === expected
    },
    findOnPath: () => 'C:\\tools\\hermes.cmd'
  })

  assert.equal(resolved, expected)
  assert.ok(seen[0].toLowerCase().includes('scripts'))
  assert.ok(!seen.some(s => s.includes(`${path.sep}bin${path.sep}`)))
})

test('resolveHermesCliBinary falls back to PATH when venv shim is missing', () => {
  const resolved = resolveHermesCliBinary('C:\\Hermes\\hermes-agent', {
    isWindows: true,
    fileExists: () => false,
    findOnPath: name => (name === 'hermes' ? 'C:\\Tools\\hermes.cmd' : null)
  })

  assert.equal(resolved, 'C:\\Tools\\hermes.cmd')
})

test('resolveHermesCliBinary uses POSIX venv/bin/hermes off Windows', () => {
  const root = '/home/apo/.hermes/hermes-agent'
  const expected = path.posix.join(root, 'venv', 'bin', 'hermes')

  const resolved = resolveHermesCliBinary(root, {
    isWindows: false,
    fileExists: candidate => candidate === expected,
    findOnPath: () => '/usr/local/bin/hermes'
  })

  assert.equal(resolved, expected)
})

test('shouldUseWindowsCliUpdateHandoff only when Windows + no staged + has CLI', () => {
  assert.equal(
    shouldUseWindowsCliUpdateHandoff({
      isWindows: true,
      stagedUpdater: null,
      hermesCli: 'C:\\venv\\Scripts\\hermes.exe'
    }),
    true
  )
  assert.equal(
    shouldUseWindowsCliUpdateHandoff({
      isWindows: true,
      stagedUpdater: 'C:\\hermes\\hermes-setup.exe',
      hermesCli: 'C:\\venv\\Scripts\\hermes.exe'
    }),
    false
  )
  assert.equal(
    shouldUseWindowsCliUpdateHandoff({
      isWindows: true,
      stagedUpdater: null,
      hermesCli: null
    }),
    false
  )
  assert.equal(
    shouldUseWindowsCliUpdateHandoff({
      isWindows: false,
      stagedUpdater: null,
      hermesCli: '/usr/bin/hermes'
    }),
    false
  )
})

test('buildWindowsCliUpdateScript waits for desktop PID then runs branch-pinned update', () => {
  const script = buildWindowsCliUpdateScript({
    desktopPid: 4242,
    hermesCmd: 'C:\\Hermes\\venv\\Scripts\\hermes.exe',
    updateRoot: 'C:\\Hermes\\hermes-agent',
    hermesHome: 'C:\\Hermes',
    branch: 'bb/gui',
    relaunchExe: 'C:\\Hermes\\Hermes\\Hermes.exe',
    relaunchArgs: ['--some-flag']
  })

  assert.match(script, /@echo off/)
  assert.match(script, /set "PID=4242"/)
  assert.match(script, /set "HERMES_HOME=C:\\Hermes"/)
  assert.match(script, /tasklist \/NH \/FI "PID eq %PID%"/)
  assert.match(script, /cd \/d "C:\\Hermes\\hermes-agent"/)
  assert.match(script, /"C:\\Hermes\\venv\\Scripts\\hermes\.exe" update --yes --branch bb\/gui/)
  assert.match(script, /start "" "C:\\Hermes\\Hermes\\Hermes\.exe" "--some-flag"/)
  assert.match(script, /del "%~f0"/)
})

test('buildWindowsCliUpdateScript omits --branch for main', () => {
  const script = buildWindowsCliUpdateScript({
    desktopPid: 1,
    hermesCmd: 'hermes.cmd',
    updateRoot: 'C:\\a',
    hermesHome: 'C:\\h',
    branch: 'main'
  })

  assert.match(script, /"hermes\.cmd" update --yes\r?$/m)
  assert.doesNotMatch(script, /--branch/)
})

test('buildWindowsCliUpdateScript strips quote characters from paths', () => {
  const script = buildWindowsCliUpdateScript({
    desktopPid: 9,
    hermesCmd: 'C:\\bad"quote\\hermes.exe',
    updateRoot: 'C:\\root',
    hermesHome: 'C:\\home'
  })

  assert.doesNotMatch(script, /bad"quote/)
  assert.match(script, /badquote/)
})
