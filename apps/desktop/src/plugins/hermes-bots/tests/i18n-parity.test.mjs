// Guards the two things that silently rot in a hand-maintained bundle pair:
// a DE bundle that lags EN (users see raw dot-keys, because resolveMessage
// echoes the key), and a t('…') call whose key exists in neither bundle.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function sliceBundle(name) {
  const start = source.indexOf(`const ${name} = {`)
  assert.ok(start >= 0, `${name} not found in plugin.js`)
  const open = source.indexOf('{', start)
  let depth = 0

  for (let i = open; i < source.length; i++) {
    if (source[i] === '{') depth++
    else if (source[i] === '}') {
      depth--
      if (depth === 0) return source.slice(open, i + 1)
    }
  }

  throw new Error(`${name} literal is unbalanced`)
}

const EN = vm.runInNewContext(`(${sliceBundle('EN_MESSAGES')})`)
const DE = vm.runInNewContext(`(${sliceBundle('DE_MESSAGES')})`)

/** Flattens to `a.b.c` -> typeof leaf, so functions and strings both count. */
function leaves(node, prefix = '', out = new Map()) {
  for (const [key, value] of Object.entries(node)) {
    const path = prefix ? `${prefix}.${key}` : key

    if (value && typeof value === 'object' && !Array.isArray(value)) {
      leaves(value, path, out)
    } else {
      out.set(path, typeof value)
    }
  }

  return out
}

const enLeaves = leaves(EN)
const deLeaves = leaves(DE)

test('DE covers every EN leaf', () => {
  const missing = [...enLeaves.keys()].filter(k => !deLeaves.has(k))

  assert.deepEqual(missing, [], `DE bundle is missing ${missing.length} key(s)`)
})

test('DE carries no key that EN dropped', () => {
  const stale = [...deLeaves.keys()].filter(k => !enLeaves.has(k))

  assert.deepEqual(stale, [], `DE bundle has ${stale.length} key(s) EN no longer defines`)
})

test('string stays string, function stays function across bundles', () => {
  const drift = [...enLeaves.entries()]
    .filter(([k, kind]) => deLeaves.has(k) && deLeaves.get(k) !== kind)
    .map(([k, kind]) => `${k}: en=${kind} de=${deLeaves.get(k)}`)

  assert.deepEqual(drift, [], 'a callback-valued message must stay callable in both bundles')
})

test('callback messages take the same number of arguments in both bundles', () => {
  const arity = []

  for (const [path, kind] of enLeaves) {
    if (kind !== 'function') continue

    const read = bundle => path.split('.').reduce((n, p) => (n == null ? undefined : n[p]), bundle)
    const en = read(EN)
    const de = read(DE)

    if (typeof de === 'function' && en.length !== de.length) {
      arity.push(`${path}: en(${en.length}) de(${de.length})`)
    }
  }

  assert.deepEqual(arity, [], 'interpolation arguments must line up')
})

test('every t() key in plugin.js resolves in the EN bundle', () => {
  const keys = new Set()

  for (const m of source.matchAll(/\bt\(\s*'([a-zA-Z0-9_.]+)'/g)) {
    keys.add(m[1])
  }

  assert.ok(keys.size > 100, `expected the plugin to call t() widely, saw ${keys.size}`)

  const unresolved = [...keys].filter(k => !enLeaves.has(k))

  assert.deepEqual(unresolved, [], `${unresolved.length} t() call(s) point at a key no bundle defines`)
})

test('no message key encodes call-site order instead of meaning', () => {
  const numbered = [...enLeaves.keys()].filter(k => /[a-z]\d+$/.test(k.split('.').pop()))

  assert.deepEqual(numbered, [], 'rename these by context; a trailing digit says nothing about the surface')
})

/** The fallback path, sliced out of plugin.js so the real code is exercised. */
function loadFallback() {
  const start = source.indexOf('function resolveMessage(')
  const marker = 'let t = (key, ...args) =>'
  const end = source.indexOf(marker)
  assert.ok(start >= 0 && end > start, 'the resolver block must stay extractable')

  const warnings = []
  const context = {
    ID: 'hermes-bots',
    EN_MESSAGES: EN,
    console: { warn: m => warnings.push(m) }
  }
  vm.runInNewContext(
    `${source.slice(start, end)}\nglobalThis.__fallback = withEnglishFallback;\nglobalThis.__resolve = resolveMessage;`,
    context
  )

  return { fallback: context.__fallback, resolve: context.__resolve, warnings }
}

test('a key the active locale lacks renders English, not the dot-path', () => {
  const { fallback } = loadFallback()
  const key = 'cron.newCronjobTitle'

  // The host echoed the key back: that is the "no entry" signal.
  assert.equal(fallback(key, [], key), EN.cron.newCronjobTitle)
  assert.notEqual(EN.cron.newCronjobTitle, key)
})

test('a translated string passes through untouched', () => {
  const { fallback, warnings } = loadFallback()

  assert.equal(fallback('cron.newCronjobTitle', [], 'Neuer Cronjob'), 'Neuer Cronjob')
  assert.deepEqual(warnings, [], 'a resolved message must not warn')
})

test('a missing key warns once, not once per render', () => {
  const { fallback, warnings } = loadFallback()

  for (let i = 0; i < 5; i++) {
    fallback('nope.notAKey', [], 'nope.notAKey')
  }

  assert.equal(warnings.length, 1, 'repeat renders must stay silent after the first warning')
  assert.match(warnings[0], /no message for "nope\.notAKey"/)
})

test('callback messages keep their arguments through the fallback', () => {
  const { fallback } = loadFallback()
  const key = 'cron.everyDays'

  assert.equal(fallback(key, [3], key), EN.cron.everyDays(3))
  assert.match(fallback(key, [3], key), /3/)
})
