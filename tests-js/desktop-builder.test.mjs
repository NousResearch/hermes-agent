import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { stageGetWindows, stageNodePtyInto } from '../apps/desktop/scripts/stage-native-deps.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []

test('desktop development composition reuses prepared icon pixels instead of provisioning them again', async () => {
  const { buildSourceDesktop } = await import('../apps/desktop/scripts/build.mjs')
  const input = fixture()
  put(join(input.icons, 'apps/desktop/assets/icon.ico'), 'prepared packaging icon')
  const commands = []
  buildSourceDesktop({ source: input.source, icons: input.icons,
    run: (command, args) => commands.push([command, ...args]),
  })
  expect(readFileSync(join(input.source, 'apps/desktop/assets/icon.ico'), 'utf8')).toBe('prepared packaging icon')
  // path.join builds platform-native separators, so accept both `/` and `\`.
  const compile = commands.find(command => command.some(arg => /[/\\]scripts[/\\]build[/\\]desktop\.mjs$/.test(arg)))
  expect(compile[compile.indexOf('--icons') + 1]).toBe(input.icons)
})
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

test('desktop development composition restages native inputs only when their receipt is stale', async () => {
  const { buildSourceDesktop } = await import('../apps/desktop/scripts/build.mjs')
  const { recordNativeInputs } = await import('../apps/desktop/scripts/prepared-native-deps.mjs')
  const input = fixture()
  put(join(input.source, 'package-lock.json'), '{}')
  const nativeDeps = join(input.source, 'apps/desktop/build/native-deps')
  cpSync(input.nativeDeps, nativeDeps, { recursive: true })
  const staged = () => {
    const commands = []
    buildSourceDesktop({ source: input.source, run: (command, args) => commands.push([command, ...args]) })
    return commands.some(command => command.some(arg => /stage-native-deps\.mjs$/.test(arg)))
  }
  expect(staged()).toBe(true) // no receipt
  recordNativeInputs({ source: input.source, out: nativeDeps, platform: process.platform, arch: process.arch })
  expect(staged()).toBe(false)
  put(join(nativeDeps, 'native/helper-fixture'), 'tampered')
  expect(staged()).toBe(true) // the tree no longer matches its digest
  recordNativeInputs({ source: input.source, out: nativeDeps, platform: process.platform, arch: process.arch, degraded: true })
  expect(staged()).toBe(true) // a soft-failed component restages until the host can complete it
  recordNativeInputs({ source: input.source, out: nativeDeps, platform: process.platform, arch: process.arch })
  put(join(input.source, 'package-lock.json'), '{"lockfileVersion":3}')
  expect(staged()).toBe(true) // a dependency pin moved
})
function put(path, text) { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, text) }
function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'desktop build with spaces-'))
  roots.push(root)
  const source = join(root, 'source')
  const app = join(source, 'apps/desktop')
  put(join(app, 'package.json'), '{"type":"module"}')
  put(join(app, 'vite.config.mjs'), 'export default { base: "./", build: { minify: false } }')
  put(join(app, 'index.html'), '<html><div id="app"></div><script type="module" src="/src/index.js"></script></html>')
  put(join(app, 'src/index.js'), 'document.getElementById("app").textContent = "built renderer"')
  put(join(app, 'electron/entry.ts'), "await import('./main')")
  put(join(app, 'electron/main.ts'), 'console.log(JSON.stringify({ stamp: __HERMES_INSTALL_STAMP__, identity: __HERMES_PRODUCT_IDENTITY__ }))')
  put(join(app, 'electron/preload.ts'), 'globalThis.fixturePreload = "compiled preload"')
  put(join(app, 'electron/preview-guest-preload-entry.ts'), 'globalThis.fixtureGuestPreload = "compiled guest preload"')
  cpSync(join(repo, 'apps/desktop/product-identity.cjs'), join(app, 'product-identity.cjs'))
  cpSync(join(repo, 'apps/desktop/electron/native'), join(app, 'electron/native'), { recursive: true })
  // product-identity.cjs resolves the channel request through the in-tree
  // packaging helper (and its content-types table) by relative path.
  for (const helper of ['scripts/msix-shared.mjs', 'scripts/release-content-types.json']) {
    cpSync(join(repo, helper), join(source, helper))
  }
  symlinkSync(join(repo, 'node_modules'), join(app, 'node_modules'), 'junction')
  const icons = join(root, 'icons')
  put(join(icons, 'apps/desktop/public/apple-touch-icon.png'), 'fresh icon')
  put(join(app, 'public/apple-touch-icon.png'), 'stale icon')
  const nativeDeps = join(root, 'native')
  // The native input is the real host binding, not a fake compiler/dependency.
  stageNodePtyInto(join(repo, 'node_modules/node-pty'), join(nativeDeps, 'node-pty'))
  stageGetWindows({ source: repo, out: nativeDeps })
  put(join(nativeDeps, 'native/helper-fixture'), 'prepared executable resource')
  const stamp = join(root, 'install-stamp.json')
  put(stamp, JSON.stringify({ schemaVersion: 1, payload: 'light', updateMechanism: 'external', commit: 'a'.repeat(40), tag: 'v1.2.3', builtAt: '2026-01-01T00:00:00.000Z' }))
  return { source, out: join(root, 'result'), icons, nativeDeps, stamp }
}
function files(root, dir = root) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) return files(root, path)
    return entry.isFile() ? [[path.slice(root.length), readFileSync(path).toString('base64')]] : []
  })
}

test('desktop compiler consumes explicit immutable inputs, replaces variants, and preserves the last product on failure', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const input = fixture()
  const before = files(input.source)
  await buildDesktop(input)
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  for (const file of [join(input.icons, 'apps/desktop/public/apple-touch-icon.png'), input.stamp,
    join(input.nativeDeps, 'node-pty/package.json'), join(input.nativeDeps, 'native/helper-fixture')]) {
    const original = readFileSync(file)
    put(file, 'changed prepared input')
    expect(productCurrent({ ...input, product: 'desktop' }), file).toBe(false)
    writeFileSync(file, original)
    expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  }
  put(join(input.source, 'apps/shared/src/client.ts'), 'export const shared = true')
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(false)
  rmSync(join(input.source, 'apps/shared'), { recursive: true })
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  expect(files(input.source)).toEqual(before)
  expect(readFileSync(join(input.out, 'apple-touch-icon.png'), 'utf8')).toBe('fresh icon')
  expect(existsSync(join(input.out, 'assets'))).toBe(true)
  expect(existsSync(join(input.out, 'electron-preload.js'))).toBe(true)
  expect(readFileSync(join(input.out, 'native/helper-fixture'), 'utf8')).toBe('prepared executable resource')
  expect(existsSync(join(input.out, 'node_modules/native'))).toBe(false)
  put(join(input.out, 'native/helper-fixture'), 'signed resource')
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  rmSync(join(input.out, 'native/helper-fixture'))
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(false)
  expect(files(join(input.out, 'node_modules/node-pty')).some(([name]) => name.endsWith('.node'))).toBe(true)
  const run = () => JSON.parse(execFileSync(process.execPath, [join(input.out, 'electron-main.mjs')], { cwd: tmpdir(), encoding: 'utf8' }))
  expect(run().identity.light).toBe(true)
  put(input.stamp, JSON.stringify({ schemaVersion: 1, payload: 'bundled', updateMechanism: 'microsoft-store', commit: 'b'.repeat(40), tag: null }))
  // Ambient variant/tag must not override explicit stamp inputs or cached CJS identity.
  execFileSync(process.execPath, [join(repo, 'scripts/build/desktop.mjs'), ...Object.entries(input).flatMap(([key, value]) => [`--${key === 'nativeDeps' ? 'native-deps' : key}`, value])], {
    cwd: tmpdir(), env: { ...process.env, PATH: '', HERMES_DESKTOP_VARIANT: 'light', HERMES_PAYLOAD_TAG: 'v1.0.0+canary.20260911T000000Z' }, stdio: 'pipe',
  })
  expect(run().identity.store).toBe(true)
  expect(run().identity.light).toBe(false)
  expect(run().stamp.commit).toBe('b'.repeat(40))
  const built = files(input.out)
  put(join(input.source, 'apps/desktop/electron/main.ts'), 'THIS IS NOT TYPESCRIPT !!!')
  await expect(buildDesktop(input)).rejects.toThrow()
  expect(files(input.out)).toEqual(built)
  expect(files(input.source).some(([name]) => name.includes('.vite') || name.endsWith('tsbuildinfo'))).toBe(false)
}, 60000)

test('desktop output cannot overlap a source directory symlinked outside the checkout', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const root = mkdtempSync(join(tmpdir(), 'desktop symlinked source-'))
  roots.push(root)
  const source = join(root, 'source')
  const externalSource = join(root, 'external-source')
  mkdirSync(join(source, 'apps/desktop'), { recursive: true })
  put(join(externalSource, 'index.js'), 'source must not change')
  symlinkSync(externalSource, join(source, 'apps/desktop/src'), 'junction')
  const out = join(externalSource, 'product')
  const before = files(root)

  // Missing prepared inputs must not hide a failure to reject source overlap.
  await expect(buildDesktop({ source, out,
    icons: join(root, 'icons'), stamp: join(root, 'stamp.json'), nativeDeps: join(root, 'native'),
  })).rejects.toThrow(/Output must not overlap build inputs/)
  expect(files(root)).toEqual(before)
  expect(readdirSync(externalSource)).toEqual(['index.js'])
  expect(existsSync(out)).toBe(false)
})

test('in-tree desktop products rebuild after build exists without replacing prepared inputs', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const input = fixture()
  input.out = join(input.source, 'apps/desktop/build/products/desktop')
  await buildDesktop(input)
  put(join(input.source, 'apps/desktop/src/index.js'), 'document.getElementById("app").textContent = "warm rebuild"')
  await buildDesktop(input)
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  const assets = files(join(input.out, 'assets')).map(([, bytes]) => Buffer.from(bytes, 'base64').toString()).join('')
  expect(assets).toContain('warm rebuild')

  // The generated-path allowance never overrides an explicitly prepared input,
  // even when that input itself is an earlier builder-owned product.
  const built = files(input.out)
  for (const prepared of [
    { stamp: join(input.out, 'hermes-build.json') },
    { nativeDeps: join(input.out, 'node_modules') },
    { icons: input.out },
  ]) {
    await expect(buildDesktop({ ...input, ...prepared })).rejects.toThrow(/overlap/)
    expect(files(input.out)).toEqual(built)
  }
}, 30000)

test('a stamp-only change reuses the renderer bytes and rebakes only main/preload; a source change recompiles', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const input = fixture()
  expect((await buildDesktop(input)).reusedRenderer).toBe(false)
  const renderer = () => files(input.out).filter(([name]) => !/electron-|preload|hermes-build\.json/.test(name))
  const before = renderer()
  put(input.stamp, JSON.stringify({ ...JSON.parse(readFileSync(input.stamp, 'utf8')), commit: 'c'.repeat(40) }))
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(false)
  expect((await buildDesktop(input)).reusedRenderer).toBe(true)
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  expect(renderer()).toEqual(before)
  const run = () => JSON.parse(execFileSync(process.execPath, [join(input.out, 'electron-main.mjs')], { cwd: tmpdir(), encoding: 'utf8' }))
  expect(run().stamp.commit).toBe('c'.repeat(40))
  // A tampered output is never reused: the receipt's output hash no longer matches.
  put(join(input.out, 'index.html'), '<html>tampered</html>')
  put(input.stamp, JSON.stringify({ ...JSON.parse(readFileSync(input.stamp, 'utf8')), commit: 'd'.repeat(40) }))
  expect((await buildDesktop(input)).reusedRenderer).toBe(false)
  expect(readFileSync(join(input.out, 'index.html'), 'utf8')).not.toContain('tampered')
  put(join(input.source, 'apps/desktop/src/index.js'), 'document.getElementById("app").textContent = "changed source"')
  expect((await buildDesktop(input)).reusedRenderer).toBe(false)
  expect(files(join(input.out, 'assets')).map(([, b]) => Buffer.from(b, 'base64').toString()).join('')).toContain('changed source')
}, 60000)

test('a renderer built under different VITE_*/NODE_ENV settings is never reused or certified', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const input = fixture()
  const restamp = commit => put(input.stamp, JSON.stringify({ ...JSON.parse(readFileSync(input.stamp, 'utf8')), commit }))
  const saved = { probe: process.env.VITE_PERF_PROBE, node: process.env.NODE_ENV }
  try {
    process.env.NODE_ENV = 'production' // vitest itself runs with NODE_ENV=test
    process.env.VITE_PERF_PROBE = '1'
    expect((await buildDesktop(input)).reusedRenderer).toBe(false)
    delete process.env.VITE_PERF_PROBE
    // The probe build's receipt no longer describes a plain build: neither gate accepts it.
    expect(productCurrent({ ...input, product: 'desktop' })).toBe(false)
    restamp('e'.repeat(40))
    expect((await buildDesktop(input)).reusedRenderer).toBe(false)
    // Unset NODE_ENV means production (vite's own build default): the production receipt stays
    // current without it, and the next stamp-only build reuses it.
    delete process.env.NODE_ENV
    expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
    restamp('f'.repeat(40))
    expect((await buildDesktop(input)).reusedRenderer).toBe(true)
    process.env.NODE_ENV = 'development'
    restamp('a'.repeat(40))
    expect((await buildDesktop(input)).reusedRenderer).toBe(false)
  } finally {
    for (const [key, value] of [['VITE_PERF_PROBE', saved.probe], ['NODE_ENV', saved.node]]) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value
    }
  }
}, 90000)

test('the mid-compile guard ignores the build clock but still fails on a real provenance change', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const { buildInputs, recordProduct } = await import('../scripts/build/freshness.mjs')
  const input = fixture()
  await buildDesktop(input)
  const stamp = readFileSync(input.stamp, 'utf8')
  // write-build-stamp.mjs rewrites `builtAt` on EVERY build (build.mjs step 20), so a
  // second build racing the first must not be killed by its own input changing.
  // This is the mid-compile guard only; productCurrent is covered by the restamp
  // tests below.
  const clockMoved = { ...JSON.parse(stamp), builtAt: '2027-01-01T00:00:00.000Z' }
  expect(clockMoved).not.toEqual(JSON.parse(stamp))
  put(input.stamp, JSON.stringify(clockMoved))
  const paths = { icons: join(input.icons, 'apps/desktop/public'), stamp: input.stamp, nativeDeps: input.nativeDeps }
  // The mid-compile guard compares the prepared inputs, and the clock is not one:
  // re-recording with a restamped clock must NOT throw, or the racing build dies
  // again on a change the build machinery made.
  const inputs = buildInputs(input.source, 'desktop', paths)
  expect(() => recordProduct({ source: input.source, product: 'desktop', out: input.out, inputs })).not.toThrow()
  // A real provenance change, by contrast, is a genuine mid-compile change: the
  // compiler captured inputs BEFORE the stamp moved, so the guard must fire.
  for (const field of [{ commit: 'd'.repeat(40) }, { payload: 'store' }, { tag: 'v9.9.9' }]) {
    put(input.stamp, JSON.stringify({ ...JSON.parse(stamp), ...field }))
    expect(() => recordProduct({ source: input.source, product: 'desktop', out: input.out, inputs }))
      .toThrow(/inputs changed/)
  }
  put(input.stamp, stamp)
}, 60000)

test('a desktop build that another build restamps mid-compile still publishes, and the clock-only rewrite is not what kills it', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const input = fixture()
  const raced = { ...JSON.parse(readFileSync(input.stamp, 'utf8')), builtAt: '2027-03-03T00:00:00.000Z' }
  // The issue's actual shape: a second `hermes desktop` runs write-build-stamp.mjs
  // while this build is compiling, so recordProduct() re-hashes a stamp whose only
  // difference is the build clock. Provenance racing in must still fail the build.
  put(join(input.source, 'apps/desktop/vite.config.mjs'), `
    import { writeFileSync } from 'node:fs';
    export default { plugins: [{ name: 'restamp-during-build', buildStart() {
      writeFileSync(${JSON.stringify(input.stamp)}, ${JSON.stringify(JSON.stringify(raced))})
    }}] }
  `)
  await expect(buildDesktop(input)).resolves.toBeTruthy()
  expect(existsSync(join(input.out, 'hermes-build.json'))).toBe(true)
  // The guard itself: a PROVENANCE change during compilation must still throw, or
  // this test would pass simply because recordProduct stopped guarding anything.
  const swapped = { ...JSON.parse(readFileSync(input.stamp, 'utf8')), commit: 'e'.repeat(40) }
  put(join(input.source, 'apps/desktop/vite.config.mjs'), `
    import { writeFileSync } from 'node:fs';
    export default { plugins: [{ name: 'swap-commit-during-build', buildStart() {
      writeFileSync(${JSON.stringify(input.stamp)}, ${JSON.stringify(JSON.stringify(swapped))})
    }}] }
  `)
  await expect(buildDesktop(input)).rejects.toThrow(/inputs changed/)
}, 60000)

test('the pre-build gate still notices a restamp, so packaging cannot ship a bundle whose clock disagrees with the baked one', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const { productCurrent } = await import('../scripts/build/freshness.mjs')
  const input = fixture()
  await buildDesktop(input)
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  // What actually gets baked into the shipped main, read from the real bundle.
  const baked = readFileSync(join(input.out, 'electron-main.mjs'), 'utf8')
    .match(/builtAt:\s*"([^"]*)"/)?.[1] ?? null
  expect(baked).toBeTruthy()
  // A second build's write-build-stamp moves the LIVE stamp. electron-builder's
  // extraResources copies that live file into the bundle, so skipping the rebuild
  // here would ship Resources/install-stamp.json disagreeing with the baked main
  // and make detectBundleSwap offer a relaunch for a bundle never replaced.
  const restamped = { ...JSON.parse(readFileSync(input.stamp, 'utf8')), builtAt: '2027-04-04T00:00:00.000Z' }
  put(input.stamp, JSON.stringify(restamped))
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(false)
  // And rebuilding restores agreement, which is what the next `hermes desktop` does.
  await buildDesktop(input)
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(true)
  const rebaked = readFileSync(join(input.out, 'electron-main.mjs'), 'utf8')
    .match(/builtAt:\s*"([^"]*)"/)?.[1] ?? null
  expect(rebaked).toBe(restamped.builtAt)
}, 60000)

test('the receipt records the clock the output BAKED, so a restamp between the bake and the record still reads as not current', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const { productCurrent, buildInputs, recordProduct } = await import('../scripts/build/freshness.mjs')
  const input = fixture()
  await buildDesktop(input)
  const paths = { icons: join(input.icons, 'apps/desktop/public'), stamp: input.stamp, nativeDeps: input.nativeDeps }
  const inputs = buildInputs(input.source, 'desktop', paths)
  const baked = readFileSync(join(input.out, 'electron-main.mjs'), 'utf8').match(/builtAt:\s*"([^"]*)"/)?.[1]
  // The window that matters: electron-builder's extraResources copies the LIVE
  // stamp after the receipt exists, so a restamp landing between bundleElectronMain
  // and recordProduct would otherwise be recorded as if this output had baked it.
  put(input.stamp, JSON.stringify({ ...JSON.parse(readFileSync(input.stamp, 'utf8')), builtAt: '2028-08-08T00:00:00.000Z' }))
  expect(() => recordProduct({
    source: input.source, product: 'desktop', out: input.out, inputs, stampClock: baked,
  })).not.toThrow()
  const receipt = JSON.parse(readFileSync(join(input.out, 'hermes-build.json'), 'utf8'))
  expect(receipt.stampClock).toBe(baked)
  expect(productCurrent({ ...input, product: 'desktop' })).toBe(false)
}, 60000)

test('a missing or unparsable desktop stamp still hashes to a distinct value instead of failing the build', async () => {
  const { buildInputs } = await import('../scripts/build/freshness.mjs')
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const input = fixture()
  // prepared[] is sorted by name, so select the stamp entry by name, not index.
  const stampHash = (path) => buildInputs(input.source, 'desktop', {
    icons: join(input.icons, 'apps/desktop/public'), stamp: path, nativeDeps: input.nativeDeps,
  }).prepared.find(entry => entry.name === 'stamp').hash
  // A missing input must read as one stable hash, never throw: buildDesktop is
  // allowed to require a stamp, but the freshness probe itself never aborts.
  const missing = join(dirname(input.stamp), 'not-written-yet.json')
  expect(stampHash(missing)).toMatch(/^[0-9a-f]{64}$/)
  expect(stampHash(missing)).toBe(stampHash(missing))
  expect(stampHash(missing)).not.toBe(stampHash(input.stamp))
  // Unparsable bytes hash by content, so any edit still invalidates.
  put(input.stamp, 'not json at all')
  const junk = stampHash(input.stamp)
  put(input.stamp, 'still not json, but different')
  expect(junk).not.toBe(stampHash(input.stamp))
  await expect(buildDesktop(input)).rejects.toThrow()
}, 60000)

test('a prepared input changing during desktop compilation cannot publish a current receipt', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const input = fixture()
  await buildDesktop(input)
  const previous = files(input.out)
  const changedStamp = { ...JSON.parse(readFileSync(input.stamp, 'utf8')), commit: 'c'.repeat(40) }
  put(join(input.source, 'apps/desktop/vite.config.mjs'), `
    import { writeFileSync } from 'node:fs';
    export default { plugins: [{ name: 'change-prepared-input', buildStart() {
      writeFileSync(${JSON.stringify(input.stamp)}, ${JSON.stringify(JSON.stringify(changedStamp))})
    }}] }
  `)
  await expect(buildDesktop(input)).rejects.toThrow(/inputs changed/)
  expect(files(input.out)).toEqual(previous)
}, 30000)

test('native preparation stages the selected source into an explicit tree before compilation', async () => {
  const { prepareDesktopNativeDependencies } = await import('../apps/desktop/scripts/stage-native-deps.mjs')
  const input = fixture()
  cpSync(join(repo, 'package-lock.json'), join(input.source, 'package-lock.json'))
  const nativeOut = join(dirname(input.out), 'prepared-native')
  await prepareDesktopNativeDependencies({ source: input.source, out: nativeOut })
  expect(existsSync(join(nativeOut, 'node-pty/package.json'))).toBe(true)
  expect(files(join(nativeOut, 'node-pty')).some(([name]) => name.endsWith('.node'))).toBe(true)
  expect(existsSync(join(input.source, 'apps/desktop/dist'))).toBe(false)
})

test('typecheck uses scratch state and incomplete prepared inputs fail before publication', async () => {
  const { buildDesktop } = await import('../scripts/build/desktop.mjs')
  const input = fixture()
  put(join(input.source, 'apps/desktop/tsconfig.json'), JSON.stringify({ compilerOptions: { composite: true, skipLibCheck: true, types: [] }, include: ['src/*.ts'] }))
  put(join(input.source, 'apps/desktop/src/typed.ts'), 'export const value: string = "valid"')
  await buildDesktop({ ...input, typecheck: true })
  const built = files(input.out)
  expect(files(input.source).some(([name]) => name.endsWith('.tsbuildinfo') || name.endsWith('typed.js'))).toBe(false)
  put(join(input.source, 'apps/desktop/src/typed.ts'), 'export const value: string = 123')
  await expect(buildDesktop({ ...input, typecheck: true })).rejects.toThrow()
  expect(files(input.out)).toEqual(built)
  rmSync(join(input.icons, 'apps/desktop/public/apple-touch-icon.png'))
  await expect(buildDesktop(input)).rejects.toThrow(/icon/i)
  expect(files(input.out)).toEqual(built)
  await expect(buildDesktop({ ...input, out: join(input.source, 'apps/desktop/scripts') })).rejects.toThrow(/Output/)
  put(join(input.icons, 'apps/desktop/public/apple-touch-icon.png'), 'fresh icon')
  rmSync(join(input.nativeDeps, 'node-pty/build'), { recursive: true, force: true })
  rmSync(join(input.nativeDeps, 'node-pty/prebuilds'), { recursive: true, force: true })
  await expect(buildDesktop(input)).rejects.toThrow(/native binding/)
  expect(files(input.out)).toEqual(built)
}, 30000)
