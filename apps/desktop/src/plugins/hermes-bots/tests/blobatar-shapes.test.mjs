import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load({ blobatarSvg } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const sdkStub = new Proxy(
    { blobatarSvg },
    { get: (target, key) => (key in target ? target[key] : undefined) }
  )
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state: { profile: { listen: () => undefined }, gateway: { listen: () => undefined } } },
    sdk: sdkStub
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__blob = { isBlobShape, parseBlobShape, blobShapeString, blobMarkup, BLOB_KINDS, BLOB_KIND_TRAIT };
`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__blob
}

test('blob shape strings round-trip through parse/build', () => {
  const b = load()

  assert.equal(b.isBlobShape('blobatar'), true)
  assert.equal(b.isBlobShape('blobatar:seed123'), true)
  assert.equal(b.isBlobShape('blobatar::sun'), true)
  assert.equal(b.isBlobShape('circle'), false)
  assert.equal(b.isBlobShape(undefined), false)

  // Unlocked: seed follows the name.
  assert.equal(JSON.stringify(b.parseBlobShape('blobatar', 'inbox-triage')), JSON.stringify({ seed: 'inbox-triage', seedPart: '', kind: '' }))
  // Locked seed.
  assert.equal(JSON.stringify(b.parseBlobShape('blobatar:abc123', 'inbox-triage')), JSON.stringify({ seed: 'abc123', seedPart: 'abc123', kind: '' }))
  // Pinned silhouette, unlocked seed.
  assert.equal(JSON.stringify(b.parseBlobShape('blobatar::cloud', 'inbox-triage')), JSON.stringify({ seed: 'inbox-triage', seedPart: '', kind: 'cloud' }))
  // Unknown silhouette is ignored, never trusted.
  assert.equal(b.parseBlobShape('blobatar:abc:mystery', 'x').kind, '')

  assert.equal(b.blobShapeString('', ''), 'blobatar')
  assert.equal(b.blobShapeString('abc', ''), 'blobatar:abc')
  assert.equal(b.blobShapeString('abc', 'sun'), 'blobatar:abc:sun')
  assert.equal(b.blobShapeString('', 'sun'), 'blobatar::sun')
})

test('every silhouette has a trait position inside its frozen band', () => {
  const b = load()
  const bands = {
    round: [0, 0.22], organic: [0.22, 0.48], boxy: [0.48, 0.60], capsule: [0.60, 0.70],
    nub: [0.70, 0.79], cloud: [0.79, 0.86], droplet: [0.86, 0.915], hexagon: [0.915, 0.95],
    sun: [0.95, 0.98], triangle: [0.98, 1]
  }

  for (const kind of b.BLOB_KINDS) {
    const v = b.BLOB_KIND_TRAIT[kind]
    assert.equal(typeof v, 'number', kind)
    assert.ok(v >= bands[kind][0] && v < bands[kind][1], `${kind} trait ${v} outside band`)
  }
})

test('blobMarkup renders via the SDK export, tags data-bot-face, pins traits', () => {
  const calls = []
  const b = load({
    blobatarSvg: (seed, opts) => {
      calls.push({ seed, opts })
      return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"></svg>'
    }
  })

  const markup = b.blobMarkup('blobatar', 'inbox-triage', 56)
  assert.ok(markup.startsWith('<svg data-bot-face="inbox-triage" '), 'roster PNG backfill needs the data-bot-face tag')
  assert.equal(calls[0].seed, 'inbox-triage')
  assert.equal(calls[0].opts.size, 56)
  assert.equal('traits' in calls[0].opts, false)

  b.blobMarkup('blobatar:abc:sun', 'inbox-triage', 32)
  assert.equal(calls[1].seed, 'abc')
  assert.equal(calls[1].opts.traits.shape, b.BLOB_KIND_TRAIT.sun)
})

test('blobMarkup degrades to null when the SDK lacks the export or the renderer throws', () => {
  const withoutSdk = load()
  assert.equal(withoutSdk.blobMarkup('blobatar', 'x', 32), null)

  const throwing = load({
    blobatarSvg: () => {
      throw new Error('boom')
    }
  })
  assert.equal(throwing.blobMarkup('blobatar', 'x', 32), null)
})
