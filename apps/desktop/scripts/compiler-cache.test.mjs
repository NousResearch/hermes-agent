import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { withCompilerCache } from './compiler-cache.mjs'

function setup() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'compiler-cache-'))
  const lock = path.join(root, 'package-lock.json')
  fs.writeFileSync(lock, '{"v":1}')
  const calls = []
  const plugin = () => ({
    transform: {
      handler: async (code, id) => {
        calls.push(id)
        return { code: `compiled(${code})`, map: { version: 3, sources: [id], sourcesContent: [code], mappings: 'AAAA' } }
      }
    }
  })
  const cached = (options = {}) =>
    withCompilerCache(plugin(), {
      command: 'build',
      cacheRoot: path.join(root, 'cache'),
      base: root,
      toolchain: [lock],
      env: { NODE_ENV: 'production' },
      ...options
    })
  const entries = () =>
    fs.readdirSync(path.join(root, 'cache'), { recursive: true }).filter(name => String(name).endsWith('.json'))
  return { root, lock, calls, cached, entries, cleanup: () => fs.rmSync(root, { recursive: true, force: true }) }
}

test('a warm build serves the stored output, identical to a fresh compile, without recompiling', async () => {
  const t = setup()
  try {
    const id = path.join(t.root, 'src/App.tsx')
    const cold = await t.cached().transform.handler('<App/>', id)
    const warm = await t.cached().transform.handler('<App/>', id)
    assert.deepEqual(warm, cold)
    assert.deepEqual(warm.map.sourcesContent, ['<App/>']) // rebuilt from the key's own text
    assert.equal(t.calls.length, 1)
    await t.cached().transform.handler('<App changed/>', id)
    assert.equal(t.calls.length, 2) // new module text is a new key
  } finally { t.cleanup() }
})

test('a toolchain or NODE_ENV change starts a new generation; a corrupt entry is a miss', async () => {
  const t = setup()
  try {
    const id = path.join(t.root, 'src/App.tsx')
    await t.cached().transform.handler('<App/>', id)
    await t.cached({ env: { NODE_ENV: 'development' } }).transform.handler('<App/>', id)
    assert.equal(t.calls.length, 2)
    fs.writeFileSync(t.lock, '{"v":2}')
    await t.cached().transform.handler('<App/>', id)
    assert.equal(t.calls.length, 3)
    for (const name of t.entries()) fs.writeFileSync(path.join(t.root, 'cache', String(name)), '{torn')
    const result = await t.cached().transform.handler('<App/>', id)
    assert.equal(result.code, 'compiled(<App/>)')
    assert.equal(t.calls.length, 4)
  } finally { t.cleanup() }
})

test('closing a build keeps only the entries and generation it used; watch builds and dev servers never prune or cache', async () => {
  const t = setup()
  try {
    const a = path.join(t.root, 'src/A.tsx')
    const b = path.join(t.root, 'src/B.tsx')
    const first = t.cached()
    await first.transform.handler('<A/>', a)
    await first.transform.handler('<B/>', b)
    await t.cached({ env: { NODE_ENV: 'development' } }).transform.handler('<A/>', a) // a stale generation
    assert.equal(fs.readdirSync(path.join(t.root, 'cache')).length, 2)

    const watch = t.cached()
    await watch.transform.handler('<A/>', a)
    await watch.closeBundle.call({ meta: { watchMode: true } })
    assert.equal(t.entries().length, 3) // a partial watch graph prunes nothing

    const second = t.cached()
    await second.transform.handler('<A/>', a)
    await second.closeBundle.call({ meta: {} })
    assert.equal(fs.readdirSync(path.join(t.root, 'cache')).length, 1)
    assert.equal(t.entries().length, 1) // B (unused this build) and the dev generation are gone

    const plain = { transform: { handler: async () => ({ code: 'x' }) } }
    assert.equal(withCompilerCache(plain, { command: 'serve', cacheRoot: path.join(t.root, 'never'), base: t.root, toolchain: [t.lock] }), plain)
    assert.equal(plain.closeBundle, undefined)
    assert.equal(fs.existsSync(path.join(t.root, 'never')), false)
  } finally { t.cleanup() }
})
