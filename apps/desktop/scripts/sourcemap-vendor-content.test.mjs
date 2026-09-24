import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { stripVendorSourcesContent, stripVendorSourcesContentFile } from './sourcemap-vendor-content.mjs'

const map = {
  version: 3,
  sources: [
    '../src/app.tsx',
    '../../../node_modules/shiki/dist/langs.mjs',
    '..\\..\\node_modules\\mermaid\\dist\\mermaid.js',
    '../../shared/src/ansi.ts',
    '../src/node_modules_helper.ts'
  ],
  sourcesContent: ['app', 'shiki', 'mermaid', 'shared', 'helper'],
  mappings: 'AAAA;AACA',
  names: ['x']
}

test('drops embedded text only for dependency sources', () => {
  const out = stripVendorSourcesContent(map)

  assert.deepEqual(out.sourcesContent, ['app', null, null, 'shared', 'helper'])
})

test('keeps sources, mappings and names so vendor frames still resolve', () => {
  const out = stripVendorSourcesContent(map)

  assert.deepEqual(out.sources, map.sources)
  assert.equal(out.mappings, map.mappings)
  assert.deepEqual(out.names, map.names)
  assert.equal(out.sourcesContent.length, out.sources.length)
})

test('leaves a map without sourcesContent untouched', () => {
  const bare = { version: 3, sources: ['../../node_modules/a.js'], mappings: 'AAAA', names: [] }

  assert.equal(stripVendorSourcesContent(bare), bare)
})

test('rewrites a map file in place', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sourcemap-vendor-'))
  const file = path.join(dir, 'chunk.js.map')

  try {
    fs.writeFileSync(file, JSON.stringify(map))
    stripVendorSourcesContentFile(file)

    assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')).sourcesContent, ['app', null, null, 'shared', 'helper'])
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
