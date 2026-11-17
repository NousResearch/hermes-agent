import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { missingRendererAssets } from './renderer-bundle'

test('Vite manifest validates lazy assets without opening their JavaScript', async () => {
  const { build } = await import('vite')
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-renderer-manifest-'))

  try {
    fs.writeFileSync(path.join(root, 'index.html'), '<script type="module" src="/main.js"></script>')
    fs.writeFileSync(path.join(root, 'main.js'), 'globalThis.loadFeature = () => import("./lazy.js")')
    fs.writeFileSync(path.join(root, 'lazy.js'), 'export const feature = 42')
    await build({ root, configFile: false, logLevel: 'silent', build: { manifest: 'renderer-manifest.json' } })
    const dist = path.join(root, 'dist')
    const index = path.join(dist, 'index.html')
    const manifest = JSON.parse(fs.readFileSync(path.join(dist, 'renderer-manifest.json'), 'utf8'))
    const reads: string[] = []

    const deps = {
      readFileSync(file: string, encoding: 'utf8') {
        reads.push(file)

        return fs.readFileSync(file, encoding)
      }
    }

    assert.deepEqual(missingRendererAssets(index, deps), [])
    assert.equal(reads.filter(file => /\.m?js$/.test(file)).length, 0)
    fs.unlinkSync(path.join(dist, manifest['lazy.js'].file))
    assert.deepEqual(missingRendererAssets(index, deps), [manifest['lazy.js'].file])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a manifest from another generation falls back to the existing lazy graph check', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-renderer-stale-'))

  try {
    fs.mkdirSync(path.join(root, 'assets'))
    fs.writeFileSync(path.join(root, 'index.html'), '<script type="module" src="assets/new.js"></script>')
    fs.writeFileSync(path.join(root, 'assets/new.js'), 'const __vite__mapDeps = i => ["assets/missing.js"]')
    fs.writeFileSync(
      path.join(root, 'renderer-manifest.json'),
      JSON.stringify({
        'index.html': { file: 'assets/old.js', isEntry: true }
      })
    )
    assert.deepEqual(missingRendererAssets(path.join(root, 'index.html')), ['assets/missing.js'])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
