// electron-bundle-options.mjs — the esbuild options for the Electron main and
// preload bundles, shared by bundle-electron-main.mjs (which writes them) and
// assert-root-install.mjs (which resolves them in memory before a build), so the
// pre-build check sees exactly the entrypoints, externals and settings the real
// bundle uses.
import { resolve } from 'node:path'

// `electron` and `node-pty` are external (provided by the runtime / staged
// separately via stage-native-deps).
export const ELECTRON_EXTERNALS = ['electron', 'node-pty', 'get-windows', 'fs']

export function electronBundleOptions(appDir, { isDev = false } = {}) {
  const distDir = resolve(appDir, 'dist')
  // Production bundles bake packaged=true so unpackaged `electron .` still
  // behaves like a packaged build. Dev bundles (`--dev`) leave the env alone
  // so HERMES_DESKTOP_DEV_SERVER / source-tree resolution keep working.
  const define = isDev
    ? {}
    : { 'process.env.HERMES_DESKTOP_IS_PACKAGED': JSON.stringify(true) }

  return [
    // main.ts → dist/electron-main.mjs
    {
      entryPoints: [resolve(appDir, 'electron/main.ts')],
      bundle: true,
      platform: 'node',
      format: 'esm',
      target: 'node20',
      outfile: resolve(distDir, 'electron-main.mjs'),
      external: ELECTRON_EXTERNALS,
      banner: {
        js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);",
      },
      define,
    },
    // preload.ts → dist/electron-preload.js
    {
      entryPoints: [resolve(appDir, 'electron/preload.ts')],
      bundle: true,
      platform: 'node',
      format: 'cjs',
      target: 'node20',
      outfile: resolve(distDir, 'electron-preload.js'),
      external: ELECTRON_EXTERNALS,
      define,
    },
  ]
}
