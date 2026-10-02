import { execFileSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync, symlinkSync, statSync, utimesSync, openSync, closeSync, renameSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test, vi } from 'vitest'
import { buildTui } from '../scripts/build/tui.mjs'
import { buildWeb } from '../scripts/build/web.mjs'
import { productOutput, publishDirectory, renameWithRetry, withProduct } from '../scripts/build/frontend-common.mjs'

vi.mock('node:fs', async importOriginal => {
  const fs = await importOriginal()
  return { ...fs, renameSync: vi.fn(fs.renameSync) }
})

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(import.meta.url)
const temporary = []
afterEach(() => { for (const dir of temporary.splice(0)) rmSync(dir, { recursive: true, force: true }) })
function fixture() {
  const dir = mkdtempSync(path.join(tmpdir(), 'product builders '))
  temporary.push(dir)
  return dir
}
function put(root, name, content) {
  const file = path.join(root, name)
  mkdirSync(path.dirname(file), { recursive: true })
  writeFileSync(file, content)
}
function dependency(root, workspace, name) {
  const target = path.dirname(require.resolve(`${name}/package.json`))
  const link = path.join(root, workspace, 'node_modules', name)
  mkdirSync(path.dirname(link), { recursive: true })
  symlinkSync(target, link, 'junction')
}
function webSource(source) {
  put(source, 'package.json', '{"private":true,"type":"module"}')
  put(source, 'web/package.json', '{"type":"module"}')
  put(source, 'web/index.html', '<html><head><link rel="icon" href="/favicon.ico"></head><body><script type="module" src="/src/main.ts"></script></body></html>')
  put(source, 'web/src/main.ts', 'import { answer } from "./value"; document.body.textContent = answer;')
  put(source, 'web/src/value.ts', 'export const answer: string = "prepared web";')
  put(source, 'web/tsconfig.json', JSON.stringify({ files: [], references: [{ path: './tsconfig.app.json' }] }))
  put(source, 'web/tsconfig.app.json', JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'ESNext', types: [], noEmit: true, tsBuildInfoFile: './node_modules/.tmp/app.tsbuildinfo' }, include: ['src'] }))
  put(source, 'web/vite.config.ts', 'import { defineConfig } from "vite"; export default defineConfig({base: "/dashboard/", build: {outDir: "should-not-be-used"}});')
  put(source, 'web/public/retained.txt', 'retained public asset')
  put(source, 'web/public/favicon.ico', 'stale source icon')
}

test('product outputs exclude the source tree except supported generated destinations', async () => {
  const inputs = ['ui-tui', 'apps/shared', 'node_modules']
  // Only call the read-only validator on real source paths: never publish here.
  for (const name of ['pm', 'hermes_cli', 'scripts', 'pyproject.toml', '.git', 'tools', '.build', 'apps/desktop/build/products']) {
    expect(() => productOutput(repo, path.join(repo, name), inputs), name).toThrow(/output/i)
  }
  const base = fixture()
  const source = path.join(base, 'source')
  mkdirSync(source)
  for (const name of ['new-source-file', 'pm/generated', 'hermes_cli/web_dist-copy', 'apps/desktop/dist-copy', 'apps/desktop/build/native-deps-copy', 'hermes_cli/web_dist/assets']) {
    expect(() => productOutput(source, path.join(source, name), inputs), name).toThrow(/output/i)
  }
  for (const name of ['hermes_cli/web_dist', 'apps/desktop/dist', 'apps/desktop/build/native-deps', 'apps/desktop/build/products/tui', 'apps/desktop/build/products/web', '.build/web', '.build/termux/tui']) {
    const out = path.join(source, name)
    expect(productOutput(source, out, inputs).out).toBe(out)
    expect(existsSync(out)).toBe(false)
    await withProduct(out, product => put(product, 'built.txt', 'product'))
    expect(productOutput(source, out, inputs).out).toBe(out)
  }
  const adjacent = path.join(base, 'web-product')
  expect(productOutput(source, adjacent, inputs).out).toBe(adjacent)
  // Ownership never exempts a directory from source/input protection.
  const sourceDir = path.join(source, 'pm')
  await withProduct(sourceDir, product => put(product, 'keep.txt', 'source'))
  expect(() => productOutput(source, sourceDir, inputs)).toThrow(/output/i)
  const generatedInput = path.join(source, '.build/web')
  expect(() => productOutput(source, generatedInput, ['.build/web'])).toThrow(/overlap/)
  symlinkSync(sourceDir, path.join(source, '.build/redirect'), 'junction')
  symlinkSync(sourceDir, path.join(base, 'source-alias'), 'junction')
  for (const alias of [path.join(source, '.build/redirect'), path.join(base, 'source-alias')]) {
    expect(() => productOutput(source, path.join(alias, 'output'), inputs)).toThrow(/output/i)
  }
})

test('publication replaces only builder-owned directories and rechecks ownership after compilation', async () => {
  const base = fixture()
  const source = path.join(base, 'source')
  mkdirSync(source)
  const unowned = path.join(base, 'unowned')
  put(unowned, 'keep.txt', 'not a product')
  mkdirSync(path.join(base, 'empty'))
  put(base, 'file', 'not a directory')
  symlinkSync(unowned, path.join(base, 'link'), 'junction')
  symlinkSync(path.join(base, 'absent'), path.join(base, 'dangling'), 'junction')
  const inTree = path.join(source, '.build/unowned')
  put(inTree, 'keep.txt', 'not a product either')
  for (const out of [unowned, path.join(base, 'empty'), path.join(base, 'file'), path.join(base, 'link'), path.join(base, 'dangling'), inTree]) {
    expect(() => productOutput(source, out, []), out).toThrow(/output/i)
    let compiled = false
    await expect(withProduct(out, () => { compiled = true })).rejects.toThrow(/output/i)
    expect(compiled).toBe(false)
  }
  const staged = path.join(base, 'staged')
  put(staged, 'new.txt', 'new product')
  await expect(publishDirectory(staged, unowned)).rejects.toThrow(/output/i)
  expect(readFileSync(path.join(unowned, 'keep.txt'), 'utf8')).toBe('not a product')
  expect(readFileSync(path.join(staged, 'new.txt'), 'utf8')).toBe('new product')

  const out = path.join(base, 'product')
  await withProduct(out, product => put(product, 'old.txt', 'old product'))
  expect(productOutput(source, out, []).out).toBe(out)
  await withProduct(out, product => put(product, 'new.txt', 'new product'))
  expect(existsSync(path.join(out, 'old.txt'))).toBe(false)
  await expect(withProduct(out, () => { throw new Error('compile failed') })).rejects.toThrow('compile failed')
  expect(readFileSync(path.join(out, 'new.txt'), 'utf8')).toBe('new product')

  const occupied = path.join(base, 'occupied-during-build')
  await expect(withProduct(occupied, product => {
    put(product, 'new.txt', 'new product')
    put(occupied, 'keep.txt', 'created by someone else')
  })).rejects.toThrow(/output/i)
  expect(readFileSync(path.join(occupied, 'keep.txt'), 'utf8')).toBe('created by someone else')
})

test('existing npm build directories can be rebuilt without adopting arbitrary outputs', async () => {
  const source = fixture()
  for (const name of ['ui-tui/dist', 'hermes_cli/web_dist', 'apps/desktop/dist', 'apps/desktop/build/native-deps']) {
    const out = path.join(source, name)
    put(out, 'old-output', 'previous compiler output')
    await withProduct(out, product => put(product, 'new-output', 'rebuilt'), { source })
    expect(existsSync(path.join(out, 'old-output'))).toBe(false)
    expect(readFileSync(path.join(out, 'new-output'), 'utf8')).toBe('rebuilt')
  }
  const unrelated = path.join(source, 'pm')
  put(unrelated, 'keep', 'source')
  await expect(withProduct(unrelated, () => {}, { source })).rejects.toThrow(/output/i)
})

test('a transient hold on a published directory is retried instead of failing the build', async () => {
  // The update tail renames `ui-tui/.dist-<rand>` onto `ui-tui/dist`, which fails with EPERM
  // while another process still holds a handle inside either directory (an antivirus scan, the
  // previous build's watcher). That hold is transient, so publication must wait it out.
  const base = fixture()
  const out = path.join(base, 'dist')
  await withProduct(out, product => put(product, 'entry.js', 'previous output'), { source: base })
  const staged = path.join(base, '.dist-held')
  put(staged, 'entry.js', 'new output')
  const held = openSync(path.join(out, 'entry.js'), 'r')
  const released = new Promise(resolve => setTimeout(() => { closeSync(held); resolve() }, 120))
  await publishDirectory(staged, out, { source: base })
  await released
  expect(readFileSync(path.join(out, 'entry.js'), 'utf8')).toBe('new output')
  expect(existsSync(`${staged}.previous`)).toBe(false)
  // A hold inside the staged directory blocks the same rename and is retried the same way.
  const second = path.join(base, 'dist-two')
  await withProduct(second, product => put(product, 'entry.js', 'previous output'), { source: base })
  const stagedTwo = path.join(base, '.dist-held-two')
  put(stagedTwo, 'entry.js', 'newer output')
  const heldStaged = openSync(path.join(stagedTwo, 'entry.js'), 'r')
  const releasedStaged = new Promise(resolve => setTimeout(() => { closeSync(heldStaged); resolve() }, 120))
  await publishDirectory(stagedTwo, second, { source: base })
  await releasedStaged
  expect(readFileSync(path.join(second, 'entry.js'), 'utf8')).toBe('newer output')
}, 30_000)

test('publication stops retrying a hold that is not transient and keeps the previous product', async () => {
  const base = fixture()
  const out = path.join(base, 'dist')
  await withProduct(out, product => put(product, 'entry.js', 'previous output'), { source: base })
  const staged = path.join(base, '.dist-permanent')
  put(staged, 'entry.js', 'new output')
  // Held open for the whole run, so every attempt fails and the error escapes the retry loop.
  const held = openSync(path.join(out, 'entry.js'), 'r')
  await expect(publishDirectory(staged, out, { source: base })).rejects.toMatchObject({ code: 'EPERM' })
  // The compiler's product is unpublished and the last good product is still in place.
  expect(readFileSync(path.join(out, 'entry.js'), 'utf8')).toBe('previous output')
  closeSync(held)
}, 30_000)

test.each(['ENOENT', 'EEXIST', 'ENOTEMPTY'])('rename retry rethrows %s on the first attempt', async code => {
  // Windows also reports occupied destinations as EPERM, so real filesystem errors alone
  // cannot exercise every non-retryable code. Count calls instead of timing the backoff.
  const failure = Object.assign(new Error('rename failed'), { code })
  const rename = vi.mocked(renameSync).mockClear().mockImplementation(() => { throw failure })
  try {
    await expect(renameWithRetry('source', 'destination')).rejects.toBe(failure)
    expect(rename).toHaveBeenCalledExactlyOnceWith('source', 'destination')
  } finally {
    rename.mockReset()
  }
})

test('rename retry preserves filesystem errors for missing sources and occupied destinations', async () => {
  const base = fixture()
  await expect(renameWithRetry(path.join(base, 'missing'), path.join(base, 'elsewhere')))
    .rejects.toMatchObject({ code: 'ENOENT' })
  // The same for a rename onto an occupied directory (what a stale build left behind).
  const staged = path.join(base, 'staged')
  const occupied = path.join(base, 'occupied')
  put(staged, 'entry.js', 'new output')
  put(occupied, 'inner/kept.txt', 'stale build output')
  await expect(renameWithRetry(staged, occupied))
    .rejects.toMatchObject({ code: expect.stringMatching(/^(EPERM|EEXIST|ENOTEMPTY|EACCES)$/) })
}, 30_000)

test('web compiles with prepared icons and workspace-local tools without writing source or tsbuildinfo', async () => {
  const { readdirSync, lstatSync } = await import('node:fs')
  function snapshot(dir) {
    return Object.fromEntries(readdirSync(dir, { recursive: true }).sort().map(name => {
      const file = path.join(dir, name)
      const stat = lstatSync(file)
      return [name, stat.isFile() ? readFileSync(file).toString('base64') : 'directory or link']
    }))
  }
  const base = fixture()
  const source = path.join(base, 'source with spaces')
  const icons = path.join(base, 'prepared icons')
  webSource(source)
  dependency(source, 'web', 'typescript')
  dependency(source, 'web', 'vite')
  put(icons, 'web/public/favicon.ico', 'prepared icon bytes')
  const before = snapshot(source)
  const out = path.join(base, 'web output')
  execFileSync(process.execPath, [path.join(repo, 'scripts/build/web.mjs'), '--source', source, '--icons', icons, '--out', out], { cwd: base })
  expect(readFileSync(path.join(out, 'favicon.ico'), 'utf8')).toBe('prepared icon bytes')
  expect(readFileSync(path.join(out, 'retained.txt'), 'utf8')).toBe('retained public asset')
  expect(readFileSync(path.join(out, 'index.html'), 'utf8')).toContain('/dashboard/assets/')
  expect(snapshot(source)).toEqual(before)
  const second = path.join(base, 'second output')
  await buildWeb({ source, icons, out: second })
  expect(readFileSync(path.join(second, 'index.html'))).toEqual(readFileSync(path.join(out, 'index.html')))
}, 30_000)

test('web rejects missing icons and compiler failures without publishing stale success', async () => {
  const base = fixture()
  const source = path.join(base, 'source')
  const icons = path.join(base, 'icons')
  const out = path.join(base, 'web')
  webSource(source)
  dependency(source, '', 'typescript')
  dependency(source, '', 'vite')
  await expect(buildWeb({ source, icons, out })).rejects.toThrow(/icon/i)
  expect(existsSync(out)).toBe(false)
  put(icons, 'web/public/favicon.ico', 'prepared icon')
  await buildWeb({ source, icons, out })
  const previous = readFileSync(path.join(out, 'index.html'))
  put(source, 'web/src/value.ts', 'export const answer: string = 42;')
  await expect(buildWeb({ source, icons, out })).rejects.toThrow(/TypeScript/)
  expect(readFileSync(path.join(out, 'index.html'))).toEqual(previous)
  put(source, 'web/src/value.ts', 'export const answer: string = "valid";')
  put(source, 'web/index.html', '<script type="module" src="/missing-entry.js"></script>')
  await expect(buildWeb({ source, icons, out })).rejects.toThrow()
  expect(readFileSync(path.join(out, 'index.html'))).toEqual(previous)
}, 30_000)

function tuiSource(source) {
  put(source, 'package.json', '{"private":true}')
  put(source, 'ui-tui/package.json', '{"type":"module"}')
  put(source, 'ui-tui/src/entry.tsx', '#!/usr/bin/env node\nimport { answer } from "@hermes/ink"; console.log(answer);')
  put(source, 'ui-tui/packages/hermes-ink/src/entry-exports.ts', 'import devtools from "react-devtools-core"; devtools.initialize(); export const answer: string = "prepared source";')
}

test('TUI uses prepared workspace-local tools from an unrelated cwd and publishes a relocatable module product', () => {
  const base = fixture()
  const source = path.join(base, 'source with spaces')
  tuiSource(source)
  dependency(source, 'ui-tui', 'esbuild')
  const out = path.join(base, 'product one')
  execFileSync(process.execPath, [path.join(repo, 'scripts/build/tui.mjs'), '--source', source, '--out', out], { cwd: base })
  expect(JSON.parse(readFileSync(path.join(out, 'package.json'), 'utf8')).type).toBe('module')
  const entry = path.join(out, 'dist/entry.js')
  expect(readFileSync(entry, 'utf8').startsWith('#!')).toBe(false)
  rmSync(source, { recursive: true })
  expect(execFileSync(process.execPath, [entry], { cwd: base, encoding: 'utf8' }).trim()).toBe('prepared source')
})

test('TUI failure does not publish or disturb a previous product, and missing prepared tools fail clearly', async () => {
  const base = fixture()
  const source = path.join(base, 'source')
  tuiSource(source)
  const out = path.join(base, 'out')
  await expect(buildTui({ source, out })).rejects.toThrow(/esbuild.*prepared|prepared.*esbuild/i)
  expect(existsSync(out)).toBe(false)
  dependency(source, '', 'esbuild')
  await buildTui({ source, out })
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  expect(productCurrent({ source, product: 'tui', out: path.join(out, 'dist') })).toBe(true)
  const previous = readFileSync(path.join(out, 'dist/entry.js'))
  put(source, 'ui-tui/src/entry.tsx', 'const broken = ;')
  await expect(buildTui({ source, out })).rejects.toThrow()
  expect(productCurrent({ source, product: 'tui', out: path.join(out, 'dist') })).toBe(false)
  expect(readFileSync(path.join(out, 'dist/entry.js'))).toEqual(previous)
  await expect(buildTui({ source, out: source })).rejects.toThrow(/output/i)
  expect(existsSync(path.join(source, 'ui-tui/src/entry.tsx'))).toBe(true)
})

test('built web freshness follows shared sources and build inputs, not mtimes or generated trees', async () => {
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const base = fixture()
  const source = path.join(base, 'source')
  const icons = path.join(base, 'icons')
  const out = path.join(base, 'web')
  webSource(source)
  dependency(source, '', 'typescript')
  dependency(source, '', 'vite')
  put(icons, 'web/public/favicon.ico', 'icon')
  put(source, 'apps/shared/src/client.ts', 'export const version = 1')
  put(source, 'scripts/build/web.mjs', '// build input')
  put(source, 'assets/logo.svg', '<svg></svg>')
  await buildWeb({ source, icons, out })
  expect(productCurrent({ source, product: 'web', out })).toBe(true)
  put(source, 'web/node_modules/.tmp/tsbuildinfo', 'generated')
  expect(productCurrent({ source, product: 'web', out })).toBe(true)
  // macOS Finder metadata written into hashed input dirs must not invalidate the build (#122632).
  put(source, 'web/public/.DS_Store', 'finder metadata')
  put(source, 'apps/shared/src/._client.ts', 'apple double sidecar')
  put(source, 'assets/.localized', '')
  // Windows file-manager metadata is the same class of noise (#122803).
  put(source, 'web/public/Thumbs.db', 'thumb cache')
  put(source, 'assets/Desktop.ini', 'folder view settings')
  expect(productCurrent({ source, product: 'web', out })).toBe(true)
  // Install completion rewrites the runtime identity after building products.
  put(source, 'install-stamp.json', '{"builtAt": "later"}')
  expect(productCurrent({ source, product: 'web', out })).toBe(true)
  put(icons, 'web/public/favicon.ico', 'changed prepared icon')
  expect(productCurrent({ source, product: 'web', out })).toBe(false)
  await buildWeb({ source, icons, out })
  expect(readFileSync(path.join(out, 'favicon.ico'), 'utf8')).toBe('changed prepared icon')
  for (const input of ['apps/shared/src/client.ts', 'scripts/build/web.mjs', 'assets/icon.svg', 'package-lock.json']) {
    put(source, input, 'changed input')
    expect(productCurrent({ source, product: 'web', out }), input).toBe(false)
    await buildWeb({ source, icons, out })
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
  }
  expect(productCurrent({ source, product: 'desktop', out })).toBe(false)
  rmSync(path.join(out, 'favicon.ico'))
  expect(productCurrent({ source, product: 'web', out })).toBe(false)
}, 30_000)

test('built TUI stays current after documentation, test and unrelated recipe changes', async () => {
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const base = fixture()
  const source = path.join(base, 'source')
  const out = path.join(base, 'tui')
  tuiSource(source)
  dependency(source, '', 'esbuild')
  put(source, 'apps/shared/src/value.ts', 'export const value: string = "shared source";')
  await buildTui({ source, out })
  const entry = path.join(out, 'dist/entry.js')
  const before = readFileSync(entry)
  for (const name of [
    'ui-tui/README.md', 'ui-tui/src/__tests__/new.test.ts', 'ui-tui/src/value.spec.ts',
    'ui-tui/packages/hermes-ink/src/value.test.ts', 'apps/shared/src/value.test.ts',
    'apps/shared/README.md', 'scripts/build/web.mjs', 'scripts/build/desktop.mjs',
    'scripts/build/README.md', 'scripts/build/node-deps.mjs',
  ]) {
    put(source, name, 'not a compiler input')
    expect(productCurrent({ source, product: 'tui', out: path.join(out, 'dist') }), name).toBe(true)
  }
  await buildTui({ source, out })
  expect(readFileSync(entry)).toEqual(before)
  expect(execFileSync(process.execPath, [entry], { encoding: 'utf8' }).trim()).toBe('prepared source')
})

test('TUI freshness invalidates source, configuration and compiler inputs and damaged outputs', async () => {
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const base = fixture()
  const source = path.join(base, 'source')
  const out = path.join(base, 'tui')
  tuiSource(source)
  dependency(source, '', 'esbuild')
  // Include the shared workspace in the real compiler graph, not just the receipt.
  put(source, 'ui-tui/src/value.ts', 'export const value: string = "shared source";')
  put(source, 'apps/shared/src/value.ts', 'export { value } from "../../../ui-tui/src/value";')
  put(source, 'ui-tui/src/entry.tsx', 'import { value } from "../../apps/shared/src/value"; console.log(value);')
  put(source, 'package-lock.json', '{"lockfileVersion":3}')
  const preservedTimes = new Date('2020-01-01T00:00:00Z')
  const sameLengthChanges = [
    ['ui-tui/src/value.ts', 'shared', 'edited'],
    ['apps/shared/src/value.ts', 'value', 'VALUE'],
    ['package.json', 'true', 'null'],
    ['package-lock.json', ':3', ':2'],
  ]
  for (const [name] of sameLengthChanges) utimesSync(path.join(source, name), preservedTimes, preservedTimes)
  await buildTui({ source, out })
  const current = () => productCurrent({ source, product: 'tui', out: path.join(out, 'dist') })
  expect(current()).toBe(true)
  for (const [name, from, to] of sameLengthChanges) {
    const file = path.join(source, name), previous = readFileSync(file, 'utf8'), before = statSync(file)
    expect(previous).toContain(from)
    writeFileSync(file, previous.replace(from, to))
    utimesSync(file, preservedTimes, preservedTimes)
    expect([statSync(file).size, statSync(file).mtimeMs]).toEqual([before.size, before.mtimeMs])
    expect(current(), `same-size and same-mtime change in ${name}`).toBe(false)
    writeFileSync(file, previous)
    utimesSync(file, preservedTimes, preservedTimes)
    expect(current(), `restored ${name}`).toBe(true)
  }
  for (const name of [
    'ui-tui/src/value.ts', 'apps/shared/src/value.ts',
    'ui-tui/packages/hermes-ink/src/entry-exports.ts',
    'ui-tui/tsconfig.json', 'ui-tui/packages/hermes-ink/tsconfig.json', 'apps/shared/tsconfig.json',
    'ui-tui/package.json', 'ui-tui/packages/hermes-ink/package.json', 'apps/shared/package.json',
    'tsconfig.json', 'package.json', 'package-lock.json', '.npmrc', 'pm/lock.json',
    'scripts/build/tui.mjs', 'scripts/build/frontend-common.mjs', 'scripts/build/freshness.mjs',
  ]) {
    const file = path.join(source, name)
    const previous = existsSync(file) ? readFileSync(file) : null
    put(source, name, 'changed input')
    expect(current(), name).toBe(false)
    if (previous) writeFileSync(file, previous)
    else rmSync(file)
    expect(current(), `restored ${name}`).toBe(true)
  }
  put(source, 'ui-tui/src/value.ts', 'export const value: string = "rebuilt source";')
  await buildTui({ source, out })
  expect(current()).toBe(true)
  const entry = path.join(out, 'dist/entry.js')
  expect(execFileSync(process.execPath, [entry], { encoding: 'utf8' }).trim()).toBe('rebuilt source')
  writeFileSync(entry, 'damaged output')
  expect(current()).toBe(false)
})
