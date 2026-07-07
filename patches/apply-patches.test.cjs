'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { describe, it } = require('node:test')

const { PATCHES, applyPatchToSource } = require('./apply-patches.cjs')

const lookupPatch = PATCHES.find(p => p.file.includes('tapClientLookup'))

describe('applyPatchToSource', () => {
  it('rewrites the target string and stamps the marker', () => {
    const source = `before\n\t\t\t\t${lookupPatch.find}\nafter`
    const result = applyPatchToSource(source, lookupPatch)

    assert.equal(result.status, 'applied')
    assert.ok(result.source.includes(lookupPatch.marker))
    assert.ok(!result.source.includes('throw new Error(`tapClientLookup'))
  })

  it('is idempotent on already-patched source', () => {
    const once = applyPatchToSource(lookupPatch.find, lookupPatch)
    const twice = applyPatchToSource(once.source, lookupPatch)

    assert.equal(twice.status, 'already-applied')
    assert.equal(twice.source, once.source)
  })

  it('reports drift when neither marker nor target is present', () => {
    const result = applyPatchToSource('completely different module body', lookupPatch)

    assert.equal(result.status, 'target-not-found')
  })
})

describe('tapClientLookup containment behavior', () => {
  // Evaluate the patched bounds-check in isolation: out-of-bounds lookups must
  // clamp to the nearest live resource and return undefined for an empty list
  // instead of throwing (the production "Index N out of bounds (length: 0)"
  // crash from stale subscribers during reconnect churn).
  const patched = applyPatchToSource(lookupPatch.find, lookupPatch).source
  const get = new Function(
    'resources',
    'keys',
    'lookup',
    `${patched}\nreturn resources[lookup.index].methods;`
  )

  it('returns undefined when the list emptied out from under a subscriber', () => {
    assert.equal(get([], [], { index: 0 }), undefined)
    assert.equal(get([], [], { index: 4 }), undefined)
  })

  it('clamps one-past-end reads to the last live resource', () => {
    const resources = [{ methods: 'm0' }, { methods: 'm1' }]

    assert.equal(get(resources, ['a', 'b'], { index: 2 }), 'm1')
    assert.equal(get(resources, ['a', 'b'], { index: -1 }), 'm0')
  })

  it('leaves in-bounds lookups untouched', () => {
    const resources = [{ methods: 'm0' }, { methods: 'm1' }]

    assert.equal(get(resources, ['a', 'b'], { index: 1 }), 'm1')
  })
})

describe('installed package', () => {
  it('matches the patch target (or is already patched)', () => {
    const filePath = path.join(__dirname, '..', lookupPatch.file)

    if (!fs.existsSync(filePath)) {
      return
    }

    const source = fs.readFileSync(filePath, 'utf8')
    assert.ok(
      source.includes(lookupPatch.marker) || source.includes(lookupPatch.find),
      'installed @assistant-ui/store drifted from the pinned 0.2.13 shape'
    )
  })
})
