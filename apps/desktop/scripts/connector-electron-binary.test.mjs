import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { connectorElectronBinary } from './connector-electron-binary.mjs'

for (const location of ['root', 'workspace']) {
  test(`connector rehearsal resolves the ${location} Electron install`, () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'connector-electron-'))
    try {
      const desktop = path.join(root, 'apps/desktop')
      fs.mkdirSync(desktop, { recursive: true })
      fs.writeFileSync(path.join(desktop, 'package.json'), '{"name":"hermes"}')
      const electron = path.join(location === 'root' ? root : desktop, 'node_modules/electron')
      fs.mkdirSync(electron, { recursive: true })
      fs.writeFileSync(path.join(electron, 'package.json'), '{"name":"electron","main":"index.js"}')
      fs.writeFileSync(path.join(electron, 'index.js'), 'throw new Error("Electron entry point must not be loaded")\n')
      const binary = path.join(electron, 'dist', 'electron')
      fs.mkdirSync(path.dirname(binary), { recursive: true })
      fs.writeFileSync(path.join(electron, 'path.txt'), 'electron')
      fs.writeFileSync(binary, '')
      assert.equal(fs.realpathSync(connectorElectronBinary(desktop)), fs.realpathSync(binary))
      fs.rmSync(binary)
      assert.throws(() => connectorElectronBinary(desktop), /Electron binary missing/)
      fs.rmSync(path.join(electron, 'path.txt'))
      assert.throws(() => connectorElectronBinary(desktop), /Electron path.txt missing/)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
}
