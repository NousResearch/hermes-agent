import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { test } from 'vitest'

import { resolvePenAssetPath } from './assets'

// Build the key exactly as the editor does: the asset's absolute file URI path, minus the slash.
const keyFor = (absolute: string) => pathToFileURL(absolute).pathname.replace(/^\/+/, '')

test('an asset key lands beside the .pen, percent-escapes decoded', () => {
  const dir = path.resolve('/tmp/pens/pen.dev – An agentic canvas')
  const pen = path.join(dir, 'pen.dev – An agentic canvas.pen')
  const asset = path.join(dir, 'images', 'hero-wash.png')

  assert.equal(resolvePenAssetPath(pen, keyFor(asset)), asset)
})

test('a key that resolves outside the canvas folder is refused', () => {
  const dir = path.resolve('/tmp/pens/Ember')
  const pen = path.join(dir, 'Ember.pen')

  assert.equal(resolvePenAssetPath(pen, keyFor(path.resolve('/tmp/pens/Other/images/x.png'))), null)
  assert.equal(resolvePenAssetPath(pen, `${keyFor(dir)}/../Other/x.png`), null)
})
