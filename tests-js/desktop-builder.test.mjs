import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { stageGetWindows, stageNodePtyInto } from '../apps/desktop/scripts/stage-native-deps.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
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
  put(join(app, 'electron/main.ts'), 'console.log(JSON.stringify({ stamp: __HERMES_INSTALL_STAMP__, identity: __HERMES_PRODUCT_IDENTITY__ }))')
  put(join(app, 'electron/preload.ts'), 'globalThis.fixturePreload = "compiled preload"')
  cpSync(join(repo, 'apps/desktop/product-identity.cjs'), join(app, 'product-identity.cjs'))
  symlinkSync(join(repo, 'node_modules'), join(app, 'node_modules'), 'junction')
  const icons = join(root, 'icons')
  put(join(icons, 'apps/desktop/public/apple-touch-icon.png'), 'fresh icon')
  put(join(app, 'public/apple-touch-icon.png'), 'stale icon')
  const nativeDeps = join(root, 'native')
  // The native input is the real host binding, not a fake compiler/dependency.
  stageNodePtyInto(join(repo, 'node_modules/node-pty'), join(nativeDeps, 'node-pty'))
  stageGetWindows({ source: repo, out: nativeDeps })
  const stamp = join(root, 'install-stamp.json')
  put(stamp, JSON.stringify({ schemaVersion: 1, payload: 'light', updateMechanism: 'external', commit: 'a'.repeat(40), tag: 'v1.2.3' }))
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
  expect(files(input.source)).toEqual(before)
  expect(readFileSync(join(input.out, 'apple-touch-icon.png'), 'utf8')).toBe('fresh icon')
  expect(existsSync(join(input.out, 'assets'))).toBe(true)
  expect(existsSync(join(input.out, 'electron-preload.js'))).toBe(true)
  expect(files(join(input.out, 'node_modules/node-pty')).some(([name]) => name.endsWith('.node'))).toBe(true)
  const run = () => JSON.parse(execFileSync(process.execPath, [join(input.out, 'electron-main.mjs')], { cwd: tmpdir(), encoding: 'utf8' }))
  expect(run().identity.light).toBe(true)
  put(input.stamp, JSON.stringify({ schemaVersion: 1, payload: 'bundled', updateMechanism: 'microsoft-store', commit: 'b'.repeat(40), tag: null }))
  // Ambient variant/tag must not override explicit stamp inputs or cached CJS identity.
  execFileSync(process.execPath, [join(repo, 'scripts/build/desktop.mjs'), ...Object.entries(input).flatMap(([key, value]) => [`--${key === 'nativeDeps' ? 'native-deps' : key}`, value])], {
    cwd: tmpdir(), env: { ...process.env, PATH: '', HERMES_DESKTOP_VARIANT: 'light', HERMES_PAYLOAD_TAG: 'v1.0.0-canary.20260911' }, stdio: 'pipe',
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

test('native preparation stages the selected source into an explicit tree before compilation', async () => {
  const { prepareDesktopNativeDependencies } = await import('../apps/desktop/scripts/stage-native-deps.mjs')
  const input = fixture()
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
  await expect(buildDesktop({ ...input, out: join(input.source, 'apps/desktop/scripts') })).rejects.toThrow(/overlap/)
  put(join(input.icons, 'apps/desktop/public/apple-touch-icon.png'), 'fresh icon')
  rmSync(join(input.nativeDeps, 'node-pty/build'), { recursive: true, force: true })
  rmSync(join(input.nativeDeps, 'node-pty/prebuilds'), { recursive: true, force: true })
  await expect(buildDesktop(input)).rejects.toThrow(/native binding/)
  expect(files(input.out)).toEqual(built)
}, 30000)
