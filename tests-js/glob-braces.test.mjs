import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

import { test } from 'vitest'

const fromWorkspace = createRequire(import.meta.url)

for (const consumer of ['glob', 'electron-winstaller', 'filelist', 'cacache', 'get-windows']) {
  test(`${consumer} matches brace patterns through its installed minimatch`, () => {
    const fromConsumer = createRequire(fromWorkspace.resolve(consumer))
    const minimatch = fromConsumer('minimatch')
    const match = typeof minimatch === 'function' ? minimatch : minimatch.minimatch

    assert.equal(match('config/x.json', '{config,conf}/*.json'), true)
    assert.equal(match('other/x.json', '{config,conf}/*.json'), false)
  })
}
