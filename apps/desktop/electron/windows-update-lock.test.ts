import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { listExternalVenvHolderPids, parseWindowsPidList } from './windows-update-lock.ts'

const ELECTRON_DIR = path.dirname(fileURLToPath(import.meta.url))

test('parseWindowsPidList keeps positive integer PIDs and de-dupes them', () => {
  assert.deepEqual(parseWindowsPidList('101, 202\r\n202\n0\nnope\n303'), [101, 202, 303])
  assert.deepEqual(parseWindowsPidList(''), [])
})

test('listExternalVenvHolderPids shells out through the configured hidden child options', () => {
  let call: any = null
  const result = listExternalVenvHolderPids({
    childOptions: options => ({ ...(options || {}), windowsHide: true }),
    currentPid: 42,
    exec: (cmd, args, options) => {
      call = { args, cmd, options }
      return '501, 777, 501'
    },
    isWindows: true,
    ownedPids: [9, 15],
    powerShellPath: 'powershell.exe',
    updateRoot: 'C:\\Users\\alice\\.hermes\\hermes-agent'
  })

  assert.deepEqual(result, [501, 777])
  assert.equal(call.cmd, 'powershell.exe')
  assert.deepEqual(call.args.slice(0, 4), ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command'])
  assert.match(call.args[4], /Get-CimInstance Win32_Process/)
  assert.equal(call.options.windowsHide, true)
  assert.equal(call.options.env.HERMES_OWNED_PIDS, '9,15,42')
  assert.equal(call.options.env.HERMES_UPDATE_ROOT, 'C:\\Users\\alice\\.hermes\\hermes-agent')
  assert.deepEqual(call.options.stdio, ['ignore', 'pipe', 'ignore'])
})

test('listExternalVenvHolderPids returns no PIDs when the scan is unavailable', () => {
  assert.deepEqual(
    listExternalVenvHolderPids({
      isWindows: false,
      powerShellPath: 'powershell.exe',
      updateRoot: 'C:\\Users\\alice\\.hermes\\hermes-agent'
    }),
    []
  )
  assert.deepEqual(
    listExternalVenvHolderPids({
      isWindows: true,
      powerShellPath: '',
      updateRoot: 'C:\\Users\\alice\\.hermes\\hermes-agent'
    }),
    []
  )
})

test('desktop update lock release re-scans and kills external venv holders', () => {
  const source = fs.readFileSync(path.join(ELECTRON_DIR, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

  assert.match(source, /childOptions:\s*hiddenWindowsChildOptions/)
  assert.match(source, /killExternalVenvHolderProcesses\(updateRoot, tag, pids\)/)
  assert.match(source, /killExternalVenvHolderProcesses\(updateRoot, tag, stragglers\)/)
})
