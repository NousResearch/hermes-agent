#!/usr/bin/env node
// bundle-electron-main.mjs — bundles electron/main.cjs into a single
// self-contained file so the nix build doesn't need to ship node_modules/.
//
// `electron` is provided by the runtime; `node-pty` is staged separately
// via stage-native-deps.cjs.  `preload.cjs` is NOT require()'d by main —
// Electron loads it via path.join(__dirname, 'preload.cjs') — so it stays
// as a separate file and doesn't need bundling.
import { build } from 'esbuild'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { renameSync, existsSync } from 'node:fs'
import { createRequire } from 'node:module'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const entry = resolve(root, 'electron/main.cjs')
const tmp = resolve(root, 'electron/main.bundled.cjs')

const options = {
  entryPoints: [entry],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node20',
  outfile: tmp,
  external: ['electron', 'node-pty'],
  logLevel: 'info',
}

// esbuild's native binary ships as a platform-specific OPTIONAL dependency
// (`@esbuild/<os>-<arch>`). esbuild is only a transitive dependency here, and
// on some lockfile / `npm ci` combinations npm never lays the optional package
// down — leaving `node_modules/@esbuild` empty, so esbuild throws
//   Error: The package "@esbuild/win32-x64" could not be found ...
// at build time and the desktop build fails (observed on Windows). Self-heal:
// install the matching platform binary at esbuild's own resolved version, then
// retry. A no-op once the binary is present.
async function bundleMain() {
  try {
    await build(options)
  } catch (err) {
    const message = String((err && err.message) || err)
    const missing = message.match(/The package "(@esbuild\/[^"]+)" could not be found/)
    if (!missing) throw err

    const require = createRequire(import.meta.url)
    const esbuildPkgJson = require.resolve('esbuild/package.json')
    const version = require(esbuildPkgJson).version
    const installRoot = resolve(dirname(esbuildPkgJson), '..', '..')
    const pkg = `${missing[1]}@${version}`
    console.warn(`[bundle-electron-main] ${missing[1]} is missing — installing ${pkg} and retrying…`)

    // Invoke npm via the current Node + npm-cli.js: no shell, array args
    // (nothing is interpolated/escaped), and it sidesteps Node 20+'s refusal to
    // execFile a .cmd directly (EINVAL on Windows). npm_execpath is set when
    // this runs under `npm run`; otherwise resolve npm-cli.js next to Node.
    const { execFileSync } = await import('node:child_process')
    const nodeDir = dirname(process.execPath)
    const npmCli = [
      process.env.npm_execpath,
      resolve(nodeDir, 'node_modules', 'npm', 'bin', 'npm-cli.js'),
      resolve(nodeDir, '..', 'lib', 'node_modules', 'npm', 'bin', 'npm-cli.js'),
    ].find((p) => p && /\.[cm]?js$/i.test(p) && existsSync(p))
    if (!npmCli) {
      throw new Error(`could not locate npm-cli.js to install ${pkg}; run \`npm install --no-save ${pkg}\` and rebuild`)
    }
    execFileSync(
      process.execPath,
      [npmCli, 'install', '--no-save', '--no-audit', '--no-fund', '--prefix', installRoot, pkg],
      { stdio: 'inherit' },
    )
    await build(options)
  }
}

await bundleMain()

// Overwrite the original with the bundled version.
renameSync(tmp, entry)

console.log(`bundled ${entry}`)
