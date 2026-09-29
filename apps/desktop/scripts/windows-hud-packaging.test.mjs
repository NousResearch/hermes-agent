import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
import { test } from 'vitest'
import { recordNativeInputs, readNativeInputs, copyNativeInputs } from './prepared-native-deps.mjs'
import { buildHudModifierMonitor, hudModifierBinaryRelativePath } from './build-hud-modifier-monitor.mjs'
import { packageWithHudFallback } from './package-with-hud-fallback.mjs'

function fixture() {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'hud-packaging-')))
  const app = path.join(root, 'apps/desktop')
  const nativeDeps = path.join(app, 'build/native-deps')
  fs.mkdirSync(path.join(nativeDeps, 'node-pty'), { recursive: true })
  fs.writeFileSync(path.join(root, 'package-lock.json'), '{}')
  fs.writeFileSync(path.join(app, 'package.json'), '{}')
  fs.writeFileSync(path.join(nativeDeps, 'node-pty/package.json'), '{}')
  fs.writeFileSync(path.join(nativeDeps, 'node-pty/pty.node'), 'required binding')
  fs.cpSync(path.join(import.meta.dirname, '../electron/native'), path.join(app, 'electron/native'), { recursive: true })
  const selection = { source: root, nativeDeps, platform: 'win32', arch: process.arch }
  const relative = hudModifierBinaryRelativePath('win32', process.arch)
  const helper = path.join(nativeDeps, relative)
  fs.mkdirSync(path.dirname(helper), { recursive: true })
  fs.writeFileSync(helper, 'helper fixture')
  recordNativeInputs({ ...selection, out: nativeDeps })
  return { root, app, nativeDeps, selection, helper, relative }
}

test('only the missing optional Windows HUD helper may differ from admitted native inputs', () => {
  const f = fixture()
  try {
    const original = fs.readFileSync(f.helper)
    fs.writeFileSync(f.helper, 'changed helper')
    assert.throws(() => readNativeInputs(f.selection), /Stale or foreign/)
    fs.rmSync(path.join(f.nativeDeps, 'native'), { recursive: true })
    assert.equal(readNativeInputs(f.selection), f.nativeDeps)
    copyNativeInputs({ ...f.selection, out: path.join(f.app, 'dist/node_modules') })
    assert.equal(fs.existsSync(path.join(f.app, 'dist', f.relative)), false)
    assert.equal(fs.readFileSync(path.join(f.app, 'dist/node_modules/node-pty/pty.node'), 'utf8'), 'required binding')
    fs.writeFileSync(path.join(f.nativeDeps, 'node-pty/pty.node'), 'corrupt required binding')
    assert.throws(() => readNativeInputs(f.selection), /Stale or foreign/)
    fs.writeFileSync(path.join(f.nativeDeps, 'node-pty/pty.node'), 'required binding')
    fs.mkdirSync(path.dirname(f.helper), { recursive: true })
    fs.writeFileSync(f.helper, original)
    assert.equal(readNativeInputs(f.selection), f.nativeDeps)
    copyNativeInputs({ ...f.selection, out: path.join(f.app, 'dist/node_modules'), omitWindowsHud: true })
    assert.equal(fs.existsSync(path.join(f.app, 'dist', f.relative)), false)
    assert.deepEqual(fs.readFileSync(f.helper), original)
  } finally {
    fs.rmSync(f.root, { recursive: true, force: true })
  }
})

test('a real ASAR read race degrades only the HUD helper and retries at most once', async () => {
  const f = fixture()
  try {
    if (process.platform === 'win32') {
      assert.ok(buildHudModifierMonitor({ source: f.root, distDir: f.nativeDeps }))
      recordNativeInputs({ ...f.selection, out: f.nativeDeps })
    }
    const asarModule = pathToFileURL(createRequire(import.meta.url).resolve('@electron/asar')).href
    const child = path.join(f.root, 'pack.mjs')
    fs.writeFileSync(child, `
      import fs from 'node:fs'; import path from 'node:path';
      import { createPackageFromStreams } from ${JSON.stringify(asarModule)};
      import { copyNativeInputs } from ${JSON.stringify(new URL('./prepared-native-deps.mjs', import.meta.url).href)};
      const f = ${JSON.stringify(f)};
      const omit = process.env.HERMES_PACKAGING_OMIT_HUD_HELPER === '1';
      fs.appendFileSync(path.join(f.root, 'attempts'), omit ? 'omit\\n' : 'normal\\n');
      copyNativeInputs({ ...f.selection, out: path.join(f.app, 'dist/node_modules'), omitWindowsHud: omit });
      const dist = path.join(f.app, 'dist');
      const required = path.join(dist, 'node_modules/node-pty/pty.node');
      const helper = path.join(dist, f.relative);
      const files = omit ? [required] : [required, helper];
      const streams = files.map(file => ({path: path.relative(dist, file), type: 'file',
        stat: fs.statSync(file), unpacked: true, streamGenerator: () => fs.createReadStream(file)}));
      // Delete after enumeration, before ASAR opens the source stream: the
      // reported failure signature, without requiring an antivirus verdict.
      if (process.env.DAMAGE !== 'none') {
        if (!omit) fs.unlinkSync(process.env.DAMAGE === 'required' ? required : helper);
        else if (process.env.DAMAGE === 'second-attempt') fs.unlinkSync(required);
      }
      await createPackageFromStreams(path.join(f.root, process.env.DAMAGE + '.asar'), streams);
    `)
    const run = damage => packageWithHudFallback({ command: process.execPath,
      args: [child], options: { env: { ...process.env, DAMAGE: damage } },
      app: f.app, platform: 'win32', arch: process.arch })
    const { extractFile, listPackage } = await import('@electron/asar')
    assert.equal((await run('none')).status, 0)
    assert.deepEqual(extractFile(path.join(f.root, 'none.asar'), path.normalize(f.relative)), fs.readFileSync(f.helper))
    assert.equal(fs.readFileSync(path.join(f.root, 'attempts'), 'utf8'), 'normal\n')
    fs.rmSync(path.join(f.root, 'attempts'))
    assert.equal((await run('hud')).status, 0)
    assert.equal(fs.readFileSync(path.join(f.root, 'attempts'), 'utf8'), 'normal\nomit\n')
    assert.equal(extractFile(path.join(f.root, 'hud.asar'), path.normalize('node_modules/node-pty/pty.node')).toString(), 'required binding')
    assert.equal(listPackage(path.join(f.root, 'hud.asar')).some(name => name.includes('hud-modifier-monitor')), false)
    for (const damage of ['required', 'second-attempt']) {
      fs.rmSync(path.join(f.root, 'attempts'))
      assert.notEqual((await run(damage)).status, 0)
      assert.equal(fs.readFileSync(path.join(f.root, 'attempts'), 'utf8'),
        damage === 'required' ? 'normal\n' : 'normal\nomit\n')
    }
  } finally {
    fs.rmSync(f.root, { recursive: true, force: true })
  }
}, 130_000)
