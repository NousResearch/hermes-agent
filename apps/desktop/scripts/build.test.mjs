import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { afterEach, test } from 'vitest'

import { buildSourceDesktop } from './build.mjs'

const roots = []

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function put(path, text) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, text)
}

test('the install stamp is written while the committed assets are still in place', () => {
  const root = mkdtempSync(join(tmpdir(), 'desktop-build-order-'))
  roots.push(root)
  const source = join(root, 'source')
  const icons = join(root, 'icons')
  const asset = join(source, 'apps/desktop/assets/icon.ico')
  put(asset, 'committed icon')
  put(join(icons, 'apps/desktop/assets/icon.ico'), 'flavored icon')

  // write-build-stamp asks git whether tracked files changed, so it must see
  // the committed bytes, not the flavored copy.
  let seenByStamp
  buildSourceDesktop({
    source,
    icons,
    run: (_command, [script]) => {
      if (basename(script) === 'write-build-stamp.mjs') seenByStamp = readFileSync(asset, 'utf8')
    }
  })

  assert.equal(seenByStamp, 'committed icon')
  assert.equal(readFileSync(asset, 'utf8'), 'flavored icon')
})
