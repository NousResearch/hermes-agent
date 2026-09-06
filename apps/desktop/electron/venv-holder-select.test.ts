import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildVenvHolderListCommand,
  hasWindowsPathPrefix,
  isHermesOwnedUpdateHolder,
  isHermesOwnedVenvDaemon,
  isOperatorManagedServeCmdline,
  parseServePidFile,
  venvHolderProcessWmiFilter
} from './venv-holder-select'

const SCRIPTS = 'C:\\Hermes\\venv\\Scripts'
const RUNTIME = 'C:\\Hermes\\.hermes-runtime\\python'

const RUNTIME_EXE =
  'C:\\Hermes\\.hermes-runtime\\python\\generation-1\\cpython-3.11-windows-x86_64-none\\python.exe'

test('matches the hindsight daemon shim (exe under venv Scripts + hindsight cmdline)', () => {
  assert.equal(
    isHermesOwnedVenvDaemon(
      'C:\\Hermes\\venv\\Scripts\\pythonw.exe',
      'C:\\Hermes\\venv\\Scripts\\pythonw.exe -m hindsight_api.main --daemon --idle-timeout 300 --port 9177',
      SCRIPTS
    ),
    true
  )
})

test('Windows path prefix match is ordinal case-insensitive', () => {
  assert.equal(
    isHermesOwnedVenvDaemon(
      'c:\\hermes\\venv\\scripts\\python.exe',
      'python.exe -m hindsight_api.main --daemon',
      'C:\\Hermes\\venv\\Scripts'
    ),
    true
  )
})

test('excludes external venv holders that are not the hindsight daemon', () => {
  // a user terminal running the hermes CLI from the venv — must NOT be killed
  assert.equal(isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\hermes.exe', 'hermes chat -q "hi"', SCRIPTS), false)
  // an unrelated python script using the venv interpreter
  assert.equal(
    isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\python.exe', 'python C:\\tools\\import.py', SCRIPTS),
    false
  )
})

test('excludes exes outside the venv even when the cmdline mentions hindsight', () => {
  assert.equal(
    isHermesOwnedVenvDaemon('C:\\Other\\pythonw.exe', 'pythonw -m hindsight_api.main --daemon', SCRIPTS),
    false
  )
})

test('prefix boundary: sibling dirs (ScriptsX) do not match', () => {
  assert.equal(hasWindowsPathPrefix('C:\\Hermes\\venv\\ScriptsX\\python.exe', SCRIPTS), false)
  assert.equal(hasWindowsPathPrefix('C:\\Hermes\\venv\\Scripts\\python.exe', SCRIPTS), true)
})

test('null/undefined fields never match', () => {
  assert.equal(isHermesOwnedVenvDaemon(null, 'x', SCRIPTS), false)
  assert.equal(isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\pythonw.exe', null, SCRIPTS), false)
  assert.equal(isHermesOwnedVenvDaemon(undefined, undefined, SCRIPTS), false)
})

test('WMI listing is name-filtered to python.exe/pythonw.exe, not a full process table scan', () => {
  assert.equal(venvHolderProcessWmiFilter(), "Name='python.exe' OR Name='pythonw.exe'")
  assert.match(buildVenvHolderListCommand(), /Win32_Process -Filter "/)
  assert.match(buildVenvHolderListCommand(), /Name='python\.exe' OR Name='pythonw\.exe'/)
  assert.doesNotMatch(buildVenvHolderListCommand(), /Get-CimInstance Win32_Process \| Where-Object/)
})

test('parseServePidFile accepts a bare pid and ignores trailing noise', () => {
  assert.equal(parseServePidFile('44084\n'), 44084)
  assert.equal(parseServePidFile('  44084 extra'), 44084)
  assert.equal(parseServePidFile(''), null)
  assert.equal(parseServePidFile('nope'), null)
})

test('operator-managed serve cmdline matches a real bind and rejects desktop --port 0', () => {
  assert.equal(
    isOperatorManagedServeCmdline(
      'python.exe -m hermes_cli.main serve --host 100.108.244.44 --port 9119'
    ),
    true
  )
  assert.equal(
    isOperatorManagedServeCmdline('python.exe -m hermes_cli.main --profile default serve --host 127.0.0.1 --port 0'),
    false
  )
  assert.equal(isOperatorManagedServeCmdline('python.exe -m hermes_cli.main gateway run'), false)
  assert.equal(isOperatorManagedServeCmdline('hermes chat -q hi'), false)
})

test('update holder matches operator serve trampoline and runtime interpreter', () => {
  const cmd = 'python.exe -m hermes_cli.main serve --host 100.108.244.44 --port 9119'

  assert.equal(isHermesOwnedUpdateHolder('C:\\Hermes\\venv\\Scripts\\python.exe', cmd, SCRIPTS, RUNTIME), true)
  assert.equal(isHermesOwnedUpdateHolder(RUNTIME_EXE, cmd, SCRIPTS, RUNTIME), true)
  assert.equal(isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\python.exe', cmd, SCRIPTS), false)
})

test('update holder matches execute_code kernel runner mapping site-packages .pyd files', () => {
  const cmd = 'python.exe C:\\Users\\u\\AppData\\Local\\Temp\\hermes_kernel_abc\\hermes_kernel_runner.py'

  assert.equal(isHermesOwnedUpdateHolder('C:\\Hermes\\venv\\Scripts\\python.exe', cmd, SCRIPTS, RUNTIME), true)
  assert.equal(isHermesOwnedUpdateHolder(RUNTIME_EXE, cmd, SCRIPTS, RUNTIME), true)
})

test('update holder matches venv-launched Codex/Claude MCP bridges', () => {
  const cmd = 'python.exe -P -m agent.transports.hermes_tools_mcp_server'

  assert.equal(isHermesOwnedUpdateHolder('C:\\Hermes\\venv\\Scripts\\python.exe', cmd, SCRIPTS, RUNTIME), true)
  assert.equal(isHermesOwnedUpdateHolder(RUNTIME_EXE, cmd, SCRIPTS, RUNTIME), true)
})

test('update holder still matches hindsight and still excludes a user CLI', () => {
  assert.equal(
    isHermesOwnedUpdateHolder(
      'C:\\Hermes\\venv\\Scripts\\pythonw.exe',
      'pythonw.exe -m hindsight_api.main --daemon',
      SCRIPTS,
      RUNTIME
    ),
    true
  )
  assert.equal(
    isHermesOwnedUpdateHolder('C:\\Hermes\\venv\\Scripts\\hermes.exe', 'hermes chat -q "hi"', SCRIPTS, RUNTIME),
    false
  )
  assert.equal(
    isHermesOwnedUpdateHolder(
      'C:\\Hermes\\venv\\Scripts\\python.exe',
      'python.exe -m hermes_cli.main gateway run',
      SCRIPTS,
      RUNTIME
    ),
    false
  )
  assert.equal(isHermesOwnedUpdateHolder('C:\\Other\\python.exe', 'python -m hermes_cli.main serve --port 9119', SCRIPTS, RUNTIME), false)
})
