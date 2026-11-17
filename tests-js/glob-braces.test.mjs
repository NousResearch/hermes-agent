import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'

import { test } from 'vitest'

const desktop = createRequire(resolve(import.meta.dirname, '../apps/desktop/package.json'))

function through(...packages) {
  let from = desktop
  for (const name of packages) from = createRequire(from.resolve(name))
  return from
}

for (const [consumer, chain] of [
  ['electron-winstaller asar', ['electron-builder', 'app-builder-lib', 'electron-winstaller', '@electron/asar']],
  ['jake filelist', ['electron-builder', 'app-builder-lib', 'electron-winstaller', 'jake', 'filelist']],
  ['cacache glob', ['electron-builder', 'app-builder-lib', 'make-fetch-happen', 'cacache', 'glob']],
  ['desktop asar', ['@electron/asar']],
]) {
  test(`${consumer} matches brace patterns through its installed minimatch`, () => {
    const minimatch = through(...chain)('minimatch')
    const match = typeof minimatch === 'function' ? minimatch : minimatch.minimatch
    assert.equal(match('config/x.json', '{config,conf}/*.json'), true)
    assert.equal(match('other/x.json', '{config,conf}/*.json'), false)
  })
}
