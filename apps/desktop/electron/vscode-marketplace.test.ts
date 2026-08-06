import assert from 'node:assert'
import test from 'node:test'

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
