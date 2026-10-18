import { cpSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { expect, test } from 'vitest'

// The freshness contract for OS file-manager metadata (#122632, #122803): writing
// .DS_Store / AppleDouble sidecars / .localized / Thumbs.db / Desktop.ini into any
// hashed tree must never flip a recorded product from current to stale. Exercises
// the real buildInputs/recordProduct/productCurrent path against a fixture tree
// instead of a full compiler run, so the regression shows up in seconds.
const fixture = (name) => {
  const dir = mkdtempSync(path.join(tmpdir(), name))
  mkdirSync(path.join(dir, 'web/src'), { recursive: true })
  writeFileSync(path.join(dir, 'package.json'), '{"private":true}')
  writeFileSync(path.join(dir, 'package-lock.json'), '{}')
  writeFileSync(path.join(dir, 'web/src/main.ts'), 'export const answer: string = "prepared web";')
  return dir
}

test('OS file-manager metadata landing in hashed trees never invalidates a recorded product', async () => {
  const { buildInputs, recordProduct, productCurrent } = await import('../scripts/build/freshness.mjs')
  const source = fixture('freshness-metadata-')
  const out = mkdtempSync(path.join(tmpdir(), 'freshness-out-'))
  try {
    mkdirSync(path.join(out, 'dist'), { recursive: true })
    const inputs = buildInputs(source, 'web')
    recordProduct({ source, product: 'web', out, inputs })
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    // A real source change must still invalidate — the guard stays fail-closed.
    writeFileSync(path.join(source, 'web/src/main.ts'), 'export const answer: string = "changed";')
    expect(productCurrent({ source, product: 'web', out })).toBe(false)
    writeFileSync(path.join(source, 'web/src/main.ts'), 'export const answer: string = "prepared web";')
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    // macOS Finder noise (#122632) and Windows file-manager noise (#122803) in
    // every hashed tree: source inputs, the output, prepared directories.
    for (const [name, content] of [
      ['.DS_Store', 'finder metadata'],
      ['web/src/._main.ts', 'apple double sidecar'],
      ['web/.localized', ''],
      ['web/Thumbs.db', 'thumb cache'],
      ['web/Desktop.ini', 'folder view settings'],
      ['._.DS_Store', 'double finder metadata'],
    ]) {
      writeFileSync(path.join(source, name), content)
      expect(productCurrent({ source, product: 'web', out }), name).toBe(true)
    }
    // Same noise inside the output tree and a prepared tree must not flip it either.
    writeFileSync(path.join(out, 'dist/.DS_Store'), 'finder metadata')
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    const prepared = mkdtempSync(path.join(tmpdir(), 'freshness-prepared-'))
    writeFileSync(path.join(prepared, 'icon.ico'), 'icon')
    writeFileSync(path.join(prepared, 'Thumbs.db'), 'thumb cache')
    const withPrepared = buildInputs(source, 'web', { icons: prepared })
    recordProduct({ source, product: 'web', out, inputs: withPrepared })
    writeFileSync(path.join(prepared, '.DS_Store'), 'finder metadata')
    expect(productCurrent({ source, product: 'web', out, prepared: { icons: prepared } })).toBe(true)
    // ...while a genuine prepared-input change still invalidates.
    writeFileSync(path.join(prepared, 'icon.ico'), 'changed icon')
    expect(productCurrent({ source, product: 'web', out, prepared: { icons: prepared } })).toBe(false)
  } finally {
    rmSync(source, { recursive: true, force: true })
    rmSync(out, { recursive: true, force: true })
  }
})

// Prebuilt products: a TUI/web receipt written in one checkout must certify the same bytes in
// another checkout of the same tree and on another platform/arch (the bundles are pure JS),
// while desktop stays bound to the platform/arch that packed its native addons.
test('a TUI/web receipt certifies the same tree in another checkout and on another host', async () => {
  const { buildInputs, recordProduct, productCurrent } = await import('../scripts/build/freshness.mjs')
  const root = mkdtempSync(path.join(tmpdir(), 'freshness-portable-'))
  const built = path.join(root, 'built-here')
  mkdirSync(built)
  const source = fixture('freshness-portable-src-')
  const icons = path.join(source, 'web/public')
  mkdirSync(icons, { recursive: true })
  writeFileSync(path.join(icons, 'favicon.ico'), 'icon')
  try {
    const out = path.join(source, 'hermes_cli/web_dist')
    mkdirSync(out, { recursive: true })
    writeFileSync(path.join(out, 'index.html'), '<html></html>')
    recordProduct({ source, product: 'web', out, inputs: buildInputs(source, 'web', { icons }) })
    const receipt = path.join(out, 'hermes-build.json')
    const saved = JSON.parse(readFileSync(receipt, 'utf8'))
    expect(saved.inputs.prepared[0].path).toBe('web/public')

    // The same tree in another directory (a moved checkout, a worktree, a CI runner).
    cpSync(source, built, { recursive: true })
    expect(productCurrent({ source: built, product: 'web', out: path.join(built, 'hermes_cli/web_dist') })).toBe(true)

    // Compiled on another platform/arch by the same Node.
    writeFileSync(receipt, JSON.stringify({ ...saved, platform: 'other-os', arch: 'other-arch' }) + '\n')
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    // ...but never by another Node, and a desktop receipt never crosses platforms.
    writeFileSync(receipt, JSON.stringify({ ...saved, node: '0.0.0' }) + '\n')
    expect(productCurrent({ source, product: 'web', out })).toBe(false)
    // A genuine desktop receipt for this tree: current here, never on another platform.
    const desktopInputs = buildInputs(source, 'desktop')
    recordProduct({ source, product: 'desktop', out, inputs: desktopInputs })
    expect(productCurrent({ source, product: 'desktop', out })).toBe(true)
    const desktop = JSON.parse(readFileSync(receipt, 'utf8'))
    writeFileSync(receipt, JSON.stringify({ ...desktop, platform: 'other-os', arch: 'other-arch' }) + '\n')
    expect(productCurrent({ source, product: 'desktop', out })).toBe(false)
  } finally {
    rmSync(source, { recursive: true, force: true })
    rmSync(root, { recursive: true, force: true })
  }
})
