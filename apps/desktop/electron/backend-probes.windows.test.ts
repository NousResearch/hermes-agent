import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { execText } from './backend-claim'
import { verifyHermesCli } from './backend-probes'

// These exercise native Windows commands, not a mocked process.platform.
const windowsTest = test.skipIf(process.platform !== 'win32')

windowsTest(
  'runtime discovery reads native registry and Python launcher output',
  async () => {
    const registry = await execText(
      'reg.exe',
      ['query', 'HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion', '/v', 'ProductName'],
      { timeout: 15_000 }
    )

    assert.match(registry, /ProductName\s+REG_SZ\s+\S/)

    const python = await execText('py.exe', ['-3', '-c', 'import sys; print(sys.executable)'], { timeout: 15_000 })
    assert.ok(path.isAbsolute(python), 'the launcher must return a usable interpreter, not a command shim')
    assert.ok(fs.statSync(python).isFile())
  },
  40_000
)

windowsTest(
  'CLI probe preserves native cmd and bat success and failure results',
  async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-native-cli-'))

    try {
      for (const extension of ['cmd', 'bat']) {
        const command = path.join(root, `hermes.${extension}`)
        fs.writeFileSync(command, '@echo off\r\nif "%~1"=="--version" (exit /b 0) else (exit /b 1)\r\n')
        assert.equal(await verifyHermesCli(command, { shell: true }), true, extension)
        fs.writeFileSync(command, '@echo off\r\nexit /b 1\r\n')
        assert.equal(await verifyHermesCli(command, { shell: true }), false, extension)
      }
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  },
  40_000
)
