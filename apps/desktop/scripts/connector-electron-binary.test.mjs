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
      fs.writeFileSync(path.join(electron, 'index.js'), 'module.exports = __filename\n')
      assert.equal(connectorElectronBinary(desktop), path.join(electron, 'index.js'))
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
}
