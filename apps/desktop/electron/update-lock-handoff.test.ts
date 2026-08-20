import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  canHandOffLockedUpdateToInstaller,
  isSameInstallVenvHolder,
  listWindowsInstallVenvHolders,
  parseWindowsProcessList
} from './update-lock-handoff'

const UPDATE_ROOT = 'C:\\Users\\me\\.hermes\\hermes-agent'

test('parseWindowsProcessList parses LIST-formatted WMIC output', () => {
  const parsed = parseWindowsProcessList(
    [
      'CommandLine=C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
      'ExecutablePath=C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
      'ProcessId=4242',
      '',
      'CommandLine=C:\\Windows\\System32\\notepad.exe',
      'ExecutablePath=C:\\Windows\\System32\\notepad.exe',
      'ProcessId=99',
      ''
    ].join('\n')
  )

  assert.deepEqual(parsed, [
    {
      commandLine:
        'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
      executablePath: 'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
      pid: 4242
    },
    {
      commandLine: 'C:\\Windows\\System32\\notepad.exe',
      executablePath: 'C:\\Windows\\System32\\notepad.exe',
      pid: 99
    }
  ])
})

test('isSameInstallVenvHolder matches venv executables from the same install root', () => {
  assert.equal(
    isSameInstallVenvHolder(
      {
        commandLine: 'pythonw.exe -m hermes_cli.main gateway run --replace',
        executablePath: 'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
        pid: 4242
      },
      UPDATE_ROOT
    ),
    true
  )
})

test('isSameInstallVenvHolder matches base interpreters that still target this install', () => {
  assert.equal(
    isSameInstallVenvHolder(
      {
        commandLine:
          'C:\\Python313\\python.exe -m hermes_cli.main gateway run --replace --root C:\\Users\\me\\.hermes\\hermes-agent',
        executablePath: 'C:\\Python313\\python.exe',
        pid: 4242
      },
      UPDATE_ROOT
    ),
    true
  )
})

test('canHandOffLockedUpdateToInstaller allows same-install gateways', () => {
  assert.equal(
    canHandOffLockedUpdateToInstaller(UPDATE_ROOT, [
      {
        commandLine:
          'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
        executablePath: 'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
        pid: 4242
      }
    ]),
    true
  )
})

test('canHandOffLockedUpdateToInstaller rejects same-install non-gateway holders', () => {
  assert.equal(
    canHandOffLockedUpdateToInstaller(UPDATE_ROOT, [
      {
        commandLine:
          'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\python.exe -m hermes_cli.main serve --host 127.0.0.1',
        executablePath: 'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\python.exe',
        pid: 4242
      }
    ]),
    false
  )
})

test('listWindowsInstallVenvHolders filters to same-install venv holders and skips this process pid', () => {
  const holders = listWindowsInstallVenvHolders(UPDATE_ROOT, () =>
    [
      'CommandLine=C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
      'ExecutablePath=C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
      'ProcessId=4242',
      '',
      `CommandLine=${process.execPath}`,
      `ExecutablePath=${process.execPath}`,
      `ProcessId=${process.pid}`,
      '',
      'CommandLine=C:\\Windows\\System32\\notepad.exe',
      'ExecutablePath=C:\\Windows\\System32\\notepad.exe',
      'ProcessId=99',
      ''
    ].join('\n')
  )

  assert.deepEqual(holders, [
    {
      commandLine:
        'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
      executablePath: 'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
      pid: 4242
    }
  ])
})

test('listWindowsInstallVenvHolders falls back to PowerShell when wmic is unavailable', () => {
  const calls: string[] = []
  const holders = listWindowsInstallVenvHolders(UPDATE_ROOT, command => {
    calls.push(command)
    if (command === 'wmic') {
      throw new Error('wmic missing')
    }
    return [
      'CommandLine=C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
      'ExecutablePath=C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
      'ProcessId=4242',
      ''
    ].join('\n')
  })

  assert.deepEqual(calls, ['wmic', 'powershell'])
  assert.deepEqual(holders, [
    {
      commandLine:
        'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe -m hermes_cli.main gateway run --replace',
      executablePath: 'C:\\Users\\me\\.hermes\\hermes-agent\\venv\\Scripts\\pythonw.exe',
      pid: 4242
    }
  ])
})
