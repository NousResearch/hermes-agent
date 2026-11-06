import assert from 'node:assert'
import zlib from 'node:zlib'

import { test } from 'vitest'

import { __testing, extractThemes, readCentralDirectory } from './vscode-marketplace'

// Build a minimal zip with stored (uncompressed) entries so the test controls
// the bytes exactly — exercises the central-directory reader + theme extraction
// without a deflate dependency.
function makeZip(entries) {
  const locals = []
  const centrals = []
  let offset = 0

  for (const { name, data } of entries) {
    const nameBuf = Buffer.from(name, 'utf8')
    const body = Buffer.from(data, 'utf8')

    const local = Buffer.alloc(30 + nameBuf.length)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(0, 8) // method: stored
    local.writeUInt32LE(body.length, 18) // compressed size
    local.writeUInt32LE(body.length, 22) // uncompressed size
    local.writeUInt16LE(nameBuf.length, 26)
    nameBuf.copy(local, 30)

    locals.push(local, body)

    const central = Buffer.alloc(46 + nameBuf.length)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(0, 10) // method: stored
    central.writeUInt32LE(body.length, 20)
    central.writeUInt32LE(body.length, 24)
    central.writeUInt16LE(nameBuf.length, 28)
    central.writeUInt32LE(offset, 42) // local header offset
    nameBuf.copy(central, 46)

    centrals.push(central)
    offset += local.length + body.length
  }

  const centralStart = offset
  const centralBuf = Buffer.concat(centrals)

  const eocd = Buffer.alloc(22)
  eocd.writeUInt32LE(0x06054b50, 0)
  eocd.writeUInt16LE(entries.length, 8)
  eocd.writeUInt16LE(entries.length, 10)
  eocd.writeUInt32LE(centralBuf.length, 12)
  eocd.writeUInt32LE(centralStart, 16)

  return Buffer.concat([...locals, centralBuf, eocd])
}

// Like makeZip, but supports deflated (method 8) entries so a test can build a
// decompression bomb: a small compressed payload that inflates far larger.
function makeZipWithDeflated(entries) {
  const locals = []
  const centrals = []
  let offset = 0

  for (const { name, data, method = 0 } of entries) {
    const nameBuf = Buffer.from(name, 'utf8')
    const raw = Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf8')
    const body = method === 8 ? zlib.deflateRawSync(raw) : raw

    const local = Buffer.alloc(30 + nameBuf.length)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(method, 8)
    local.writeUInt32LE(body.length, 18) // compressed size
    local.writeUInt32LE(raw.length, 22) // uncompressed size
    local.writeUInt16LE(nameBuf.length, 26)
    nameBuf.copy(local, 30)

    locals.push(local, body)

    const central = Buffer.alloc(46 + nameBuf.length)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(method, 10)
    central.writeUInt32LE(body.length, 20)
    central.writeUInt32LE(raw.length, 24)
    central.writeUInt16LE(nameBuf.length, 28)
    central.writeUInt32LE(offset, 42) // local header offset
    nameBuf.copy(central, 46)

    centrals.push(central)
    offset += local.length + body.length
  }

  const centralStart = offset
  const centralBuf = Buffer.concat(centrals)

  const eocd = Buffer.alloc(22)
  eocd.writeUInt32LE(0x06054b50, 0)
  eocd.writeUInt16LE(entries.length, 8)
  eocd.writeUInt16LE(entries.length, 10)
  eocd.writeUInt32LE(centralBuf.length, 12)
  eocd.writeUInt32LE(centralStart, 16)

  return Buffer.concat([...locals, centralBuf, eocd])
}

test('readCentralDirectory finds every entry', () => {
  const zip = makeZip([
    { name: 'extension/package.json', data: '{}' },
    { name: 'extension/themes/x.json', data: '{}' }
  ])

  const records = readCentralDirectory(zip)
  assert.ok(records.has('extension/package.json'))
  assert.ok(records.has('extension/themes/x.json'))
})

test('extractThemes reads contributed color themes (resolving ./ paths)', () => {
  const pkg = JSON.stringify({
    name: 'theme-dracula',
    displayName: 'Dracula',
    contributes: {
      themes: [{ label: 'Dracula', uiTheme: 'vs-dark', path: './themes/dracula.json' }]
    }
  })

  const themeJson = JSON.stringify({ name: 'Dracula', type: 'dark', colors: { 'editor.background': '#282a36' } })

  const zip = makeZip([
    { name: 'extension/package.json', data: pkg },
    { name: 'extension/themes/dracula.json', data: themeJson }
  ])

  const themes = extractThemes(zip)
  assert.strictEqual(themes.length, 1)
  assert.strictEqual(themes[0].label, 'Dracula')
  assert.strictEqual(themes[0].uiTheme, 'vs-dark')
  assert.match(themes[0].contents, /editor\.background/)
})

test('extractThemes skips a decompression-bomb theme entry instead of inflating it', () => {
  // A tiny deflated payload that inflates to 17 MB — just over MAX_ENTRY_BYTES
  // (16 MB). Unbounded, inflateRawSync would materialize the full 17 MB (and a
  // real bomb would reach gigabytes). With the inflate cap, zlib throws
  // ERR_BUFFER_TOO_LARGE, the per-theme catch skips the entry, and the bomb
  // theme is absent from the result rather than exhausting main-process memory.
  const bomb = Buffer.alloc(17 * 1024 * 1024, 0x41)

  const pkg = JSON.stringify({
    name: 'bomb-theme',
    displayName: 'Bomb',
    contributes: { themes: [{ label: 'Bomb', uiTheme: 'vs-dark', path: './themes/bomb.json' }] }
  })

  const zip = makeZipWithDeflated([
    { name: 'extension/package.json', data: pkg, method: 0 },
    { name: 'extension/themes/bomb.json', data: bomb, method: 8 }
  ])

  const themes = extractThemes(zip)
  assert.strictEqual(themes.length, 0) // bomb entry skipped, not inflated
})

test('extractThemes skips an oversized stored theme entry instead of decoding it', () => {
  // A stored (uncompressed) theme entry larger than MAX_ENTRY_BYTES (16 MB)
  // must be rejected by the length check before it is decoded to a string —
  // the stored path has no zlib maxOutputLength to lean on.
  const oversized = Buffer.alloc(17 * 1024 * 1024, 0x41)

  const pkg = JSON.stringify({
    name: 'big-theme',
    displayName: 'Big',
    contributes: { themes: [{ label: 'Big', uiTheme: 'vs-dark', path: './themes/big.json' }] }
  })

  const zip = makeZipWithDeflated([
    { name: 'extension/package.json', data: pkg, method: 0 },
    { name: 'extension/themes/big.json', data: oversized, method: 0 }
  ])

  const themes = extractThemes(zip)
  assert.strictEqual(themes.length, 0) // oversized stored entry skipped, not decoded
})

test('extractThemes returns empty when the extension contributes no themes', () => {
  const zip = makeZip([{ name: 'extension/package.json', data: JSON.stringify({ name: 'x', contributes: {} }) }])
  assert.deepStrictEqual(extractThemes(zip), [])
})

test('extractThemes throws when the manifest is missing', () => {
  const zip = makeZip([{ name: 'extension/other.txt', data: 'hi' }])
  assert.throws(() => extractThemes(zip), /manifest missing/i)
})

test('looksLikeIconTheme filters icon/product-icon packs out of theme search', () => {
  const { looksLikeIconTheme } = __testing

  // Tagged contribution points are the strongest signal.
  assert.strictEqual(looksLikeIconTheme({ tags: ['theme', 'icon-theme'] }), true)
  assert.strictEqual(looksLikeIconTheme({ tags: ['product-icon-theme'] }), true)

  // Name/description fallback for packs that don't tag themselves.
  assert.strictEqual(looksLikeIconTheme({ displayName: 'Material Icon Theme' }), true)
  assert.strictEqual(looksLikeIconTheme({ shortDescription: 'A pack of file icons.' }), true)

  // Real color themes survive.
  assert.strictEqual(looksLikeIconTheme({ displayName: 'Dracula Official', tags: ['theme', 'color-theme'] }), false)
  assert.strictEqual(looksLikeIconTheme({ displayName: 'One Dark Pro' }), false)
})
