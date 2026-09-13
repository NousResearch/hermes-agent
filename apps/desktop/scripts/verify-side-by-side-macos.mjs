#!/usr/bin/env node
// Native fixture proof, not signed Hermes artifact E2E.
// Run: node apps/desktop/scripts/verify-side-by-side-macos.mjs <icon-input-root>
// Inputs: cases.json [{label, env}], plus <label>/icon{,-dark}.icns from
// scripts/generate_icons.py. Use stable, canary, commit-a, commit-b, in order.
// Existing Electron/build dependencies are read-only. All writes stay in /tmp.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath, pathToFileURL } from 'node:url'

assert.equal(process.platform, 'darwin', 'Run this proof on native macOS')
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const desktop = path.join(repo, 'apps/desktop')
const require = createRequire(path.join(desktop, 'package.json'))
const builder = path.dirname(require.resolve('app-builder-lib'))
const { AppInfo } = await import(pathToFileURL(path.join(builder, 'appInfo.js')))
const { MacPackager } = await import(pathToFileURL(path.join(builder, 'macPackager.js')))
const { createMacApp } = await import(pathToFileURL(path.join(builder, 'electron/mac/electronMac.js')))
const { build } = require('esbuild')
const electron = path.dirname(require.resolve('electron'))
const electronApp = path.join(electron, 'dist/Electron.app')
assert.ok(process.argv[2], 'Pass the icon input root with cases.json and generated ICNS files')
const inputs = path.resolve(process.argv[2])
const cases = JSON.parse(fs.readFileSync(path.join(inputs, 'cases.json'), 'utf8'))
assert.deepEqual(
  cases.map(row => row.label),
  ['stable', 'canary', 'commit-a', 'commit-b']
)
const root = fs.mkdtempSync('/tmp/hermes-sxs-macos-')
const home = path.join(root, 'home')
fs.mkdirSync(home)
const env = {
  ...process.env,
  HOME: home,
  CFFIXED_USER_HOME: home,
  TMPDIR: root,
  HERMES_HOME: path.join(home, '.hermes'),
  HERMES_RUNTIME_DIR: path.join(root, 'runtime')
}
for (const key of Object.keys(env)) {
  if (
    /^(HERMES_(BUILD_COMMIT|PAYLOAD_TAG|PAYLOAD_VERSION|DESKTOP_)|ELECTRON_RUN_AS_NODE|NODE_OPTIONS|BUILD_NUMBER)/.test(
      key
    )
  )
    delete env[key]
}
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex')
const rows = []
const children = new Set()
console.log(`Native macOS fixture proof: ${root}`)
function checked(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', env, timeout: 120_000, ...options })
  fs.appendFileSync(
    path.join(root, 'commands.log'),
    JSON.stringify({ command, args, status: result.status, stdout: result.stdout, stderr: result.stderr }) + '\n'
  )
  if (result.error) throw result.error
  assert.equal(result.status, 0, `${command} ${args.join(' ')}\n${result.stdout}\n${result.stderr}`)
  return result.stdout.trim()
}
function plist(file) {
  checked('/usr/bin/plutil', ['-lint', file])
  return JSON.parse(checked('/usr/bin/plutil', ['-convert', 'json', '-o', '-', file]))
}
async function waitFor(predicate, detail) {
  const deadline = Date.now() + 20_000
  while (Date.now() < deadline) {
    if (predicate()) return
    await delay(50)
  }
  throw new Error(`Timed out: ${detail}`)
}
async function launch(row, label, hold) {
  const receipt = path.join(root, `${label}.json`)
  const log = fs.openSync(path.join(root, `${label}.log`), 'w')
  const child = spawn(row.binary, ['--no-first-run', '--disable-gpu', '--disable-crash-reporter'], {
    cwd: root,
    env: { ...env, PROBE_RECEIPT: receipt, PROBE_HOLD: hold ? '1' : '0' },
    stdio: ['ignore', log, log]
  })
  fs.closeSync(log)
  children.add(child)
  child.once('exit', () => children.delete(child))
  await waitFor(() => fs.existsSync(receipt) || child.exitCode !== null || child.signalCode !== null, label)
  assert.ok(fs.existsSync(receipt), `${label} exited before receipt; see ${label}.log`)
  const result = JSON.parse(fs.readFileSync(receipt, 'utf8'))
  if (!hold || !result.lock) await waitFor(() => child.exitCode !== null || child.signalCode !== null, `${label} exit`)
  return { child, result, receipt }
}
async function stop(child) {
  if (child.exitCode === null && child.signalCode === null) {
    child.kill('SIGTERM')
    await waitFor(() => child.exitCode !== null || child.signalCode !== null, 'fixture exit')
  }
}
try {
  for (const { label, env: flavor } of cases) {
    const facts = JSON.parse(
      checked(
        process.execPath,
        [
          '--input-type=module',
          '-e',
          `
      import identity from ${JSON.stringify(pathToFileURL(path.join(desktop, 'product-identity.cjs')).href)};
      import config from ${JSON.stringify(pathToFileURL(path.join(desktop, 'electron-builder.config.cjs')).href)};
      console.log(JSON.stringify({identity,config}));
    `
        ],
        { env: { ...env, HERMES_DESKTOP_VARIANT: 'bundled', ...flavor } }
      )
    )
    const { identity, config } = facts
    const metadata = {
      ...JSON.parse(fs.readFileSync(path.join(desktop, 'package.json'), 'utf8')),
      ...config.extraMetadata
    }
    const appInfo = new AppInfo({ config, metadata }, null, config.mac)
    const out = path.join(root, label)
    fs.mkdirSync(out)
    checked('/bin/cp', ['-R', electronApp, path.join(out, 'Electron.app')])
    // Exercise the actual builder transformations, not a handwritten plist.
    const adapter = {
      config,
      appInfo,
      platformOptions: config.mac,
      platformSpecificBuildOptions: config.mac,
      framework: { distMacOsAppName: 'Electron.app' },
      fileAssociations: [],
      getPlatformConfig: () => ({ config: config.mac }),
      getIconPath: async () => path.join(inputs, label, 'icon.icns'),
      applyCommonInfo: MacPackager.prototype.applyCommonInfo
    }
    await createMacApp(adapter, out, null, 'mac')
    const bundle = path.join(out, `${appInfo.productFilename}.app`)
    const contents = path.join(bundle, 'Contents')
    const info = plist(path.join(contents, 'Info.plist'))
    assert.equal(info.CFBundleIdentifier, identity.appId)
    for (const key of ['CFBundleName', 'CFBundleDisplayName', 'CFBundleExecutable'])
      assert.equal(info[key], identity.displayName)
    assert.ok(fs.existsSync(path.join(contents, 'MacOS', info.CFBundleExecutable)))
    const helpers = fs
      .readdirSync(path.join(contents, 'Frameworks'))
      .filter(name => name.endsWith('.app'))
      .map(name => {
        const dir = path.join(contents, 'Frameworks', name, 'Contents')
        const helper = plist(path.join(dir, 'Info.plist'))
        assert.ok(helper.CFBundleIdentifier.startsWith(`${identity.appId}.helper`))
        assert.ok(fs.existsSync(path.join(dir, 'MacOS', helper.CFBundleExecutable)))
        return { name, id: helper.CFBundleIdentifier, executable: helper.CFBundleExecutable }
      })
    assert.ok(helpers.length > 0)
    assert.equal(
      hash(path.join(contents, 'Resources', info.CFBundleIconFile)),
      hash(path.join(inputs, label, 'icon.icns'))
    )
    const icons = []
    for (const filename of ['icon.icns', 'icon-dark.icns']) {
      const icon = path.join(inputs, label, filename)
      const decoded = path.join(out, `${filename}.iconset`)
      checked('/usr/bin/iconutil', ['-c', 'iconset', '-o', decoded, icon])
      const frames = fs
        .readdirSync(decoded)
        .filter(name => name.endsWith('.png'))
        .map(frame => {
          const dimensions = checked('/usr/bin/sips', [
            '-g',
            'pixelWidth',
            '-g',
            'pixelHeight',
            path.join(decoded, frame)
          ])
          const size = /^icon_(\d+)x\d+(@2x)?\.png$/.exec(frame)
          assert.ok(size, `Unexpected native ICNS frame: ${frame}`)
          const expected = Number(size[1]) * (size[2] ? 2 : 1)
          const width = Number(/pixelWidth:\s*(\d+)/.exec(dimensions)?.[1])
          const height = Number(/pixelHeight:\s*(\d+)/.exec(dimensions)?.[1])
          assert.deepEqual([width, height], [expected, expected])
          return { frame, width, height }
        })
      assert.ok(frames.length >= 7, `${label} ${filename}: incomplete native ICNS decode`)
      assert.equal(Math.max(...frames.map(frame => frame.width)), 1024)
      icons.push({ filename, sha256: hash(icon), frames })
    }
    const resources = path.join(contents, 'Resources')
    fs.rmSync(path.join(resources, 'default_app.asar'), { force: true })
    const appDir = path.join(resources, 'app')
    fs.mkdirSync(appDir)
    fs.writeFileSync(path.join(appDir, 'package.json'), JSON.stringify({ ...metadata, main: 'main.mjs' }))
    await build({
      stdin: {
        contents: `
      import assert from 'node:assert/strict';
      import fs from 'node:fs';
      import path from 'node:path';
      import { app } from 'electron';
      import { applyDesktopIdentity, PRODUCT_IDENTITY } from ${JSON.stringify(path.join(desktop, 'electron/product-identity.ts'))};
      const initial = { name: app.getName(), userData: app.getPath('userData'), appData: app.getPath('appData') };
      // HOME and CFFIXED_USER_HOME redirect native path discovery, not identity.
      assert.ok(initial.userData.startsWith(process.env.HOME + '/'), JSON.stringify(initial));
      const selected = applyDesktopIdentity(app);
      const final = { name: app.getName(), userData: app.getPath('userData'), appData: app.getPath('appData') };
      assert.ok(final.userData.startsWith(process.env.HOME + '/'));
      fs.mkdirSync(final.userData, {recursive:true});
      app.setPath('sessionData', final.userData);
      const lock = app.requestSingleInstanceLock();
      const result = { initial, final, selected, identity: PRODUCT_IDENTITY, lock, pid: process.pid, versions: process.versions, packaged: app.isPackaged, secondInstances: 0 };
      const save = () => {
        fs.writeFileSync(process.env.PROBE_RECEIPT + '.tmp', JSON.stringify(result, null, 2));
        fs.renameSync(process.env.PROBE_RECEIPT + '.tmp', process.env.PROBE_RECEIPT);
      };
      app.on('second-instance', () => { result.secondInstances++; save(); });
      if (!lock) { save(); app.exit(0); }
      else app.whenReady().then(() => { app.dock?.hide(); save(); if (process.env.PROBE_HOLD !== '1') app.exit(0); });
      setTimeout(() => app.exit(3), 90000).unref();
    `,
        resolveDir: desktop,
        loader: 'ts'
      },
      outfile: path.join(appDir, 'main.mjs'),
      bundle: true,
      platform: 'node',
      format: 'esm',
      external: ['electron'],
      define: { __HERMES_PRODUCT_IDENTITY__: JSON.stringify(identity) }
    })
    const row = {
      label,
      identity,
      bundle,
      binary: path.join(contents, 'MacOS', info.CFBundleExecutable),
      info,
      helpers,
      icons,
      updaterCache: appInfo.updaterCacheDirName,
      packageName: metadata.name,
      packageProductName: metadata.productName
    }
    rows.push(row)
    console.log(`Bundle identity and native ICNS decode PASS: ${label}`)
  }
  for (const field of ['bundle', 'updaterCache']) assert.equal(new Set(rows.map(row => row[field])).size, 4)
  assert.equal(new Set(rows.map(row => row.info.CFBundleIdentifier)).size, 4)
  assert.equal(
    new Set(rows.flatMap(row => row.helpers.map(helper => helper.id))).size,
    rows.reduce((n, row) => n + row.helpers.length, 0)
  )
  for (const index of [0, 1]) assert.equal(new Set(rows.map(row => row.icons[index].sha256)).size, 4)
  const running = []
  for (const row of rows) {
    const instance = await launch(row, `${row.label}-primary`, true)
    assert.equal(instance.result.lock, true)
    assert.equal(instance.result.packaged, true)
    assert.equal(instance.result.initial.name, row.packageProductName)
    assert.equal(path.basename(instance.result.initial.userData), row.packageProductName)
    assert.equal(
      path.basename(instance.result.final.userData),
      row.label === 'stable' ? row.packageProductName : row.identity.appNamePascal
    )
    running.push(instance)
    row.runtime = instance.result
  }
  assert.equal(new Set(running.map(row => row.result.final.userData)).size, 4)
  const duplicate = await launch(rows[1], 'canary-duplicate', false)
  assert.equal(duplicate.result.lock, false)
  await waitFor(
    () => JSON.parse(fs.readFileSync(running[1].receipt, 'utf8')).secondInstances === 1,
    'canary second-instance event'
  )
  for (const instance of running) assert.equal(instance.child.exitCode, null)
  await stop(running[1].child)
  const restarted = await launch(rows[1], 'canary-restarted', false)
  assert.equal(restarted.result.lock, true)
  assert.equal(restarted.result.final.userData, running[1].result.final.userData)
  for (const index of [0, 2, 3]) assert.equal(running[index].child.exitCode, null)
  fs.writeFileSync(
    path.join(root, 'receipt.json'),
    JSON.stringify(
      {
        scope:
          'Native builder transformation and Electron fixture only; no release build, signing, registration, update or Hermes backend E2E.',
        platform: os.release(),
        arch: process.arch,
        node: process.version,
        electron: JSON.parse(fs.readFileSync(path.join(electron, 'package.json'), 'utf8')).version,
        builder: JSON.parse(fs.readFileSync(path.join(builder, '../package.json'), 'utf8')).version,
        testSha256: hash(fileURLToPath(import.meta.url)),
        sourceHashes: Object.fromEntries(
          ['product-identity.cjs', 'electron-builder.config.cjs', 'electron/product-identity.ts'].map(file => [
            file,
            hash(path.join(desktop, file))
          ])
        ),
        rows,
        duplicate: duplicate.result,
        restarted: restarted.result
      },
      null,
      2
    )
  )
  console.log(
    `PASS: four concurrent identities; duplicate rejected; canary restart isolated. Receipt: ${root}/receipt.json`
  )
} finally {
  for (const child of children) await stop(child)
}
