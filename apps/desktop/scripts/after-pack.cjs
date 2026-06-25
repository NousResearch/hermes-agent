/**
 * after-pack.cjs — electron-builder afterPack hook.
 *
 * Runs for EVERY packed build — first install, `hermes desktop`, the
 * installer's --update rebuild, and a dev's manual `npm run pack` — and does
 * two things:
 *
 *   1. assertBundledJsDeps (all platforms): fail the build if the JS runtime
 *      deps that scripts/stage-native-deps.cjs stages into the asar (simple-git
 *      and its dependency closure, require()d by electron/git-review-ops.cjs)
 *      didn't actually land in the packaged app.asar. Those deps are hoisted to
 *      the workspace-root node_modules and only reach the bundle via the
 *      package.json `build.files` entry; a silent miss there ships an app that
 *      crashes on launch with "Cannot find module 'simple-git'" (#52735). A
 *      packed-output assertion is the only reliable guard — the on-disk staging
 *      can succeed while electron-builder drops the files (e.g. extraResources
 *      strips node_modules dirs).
 *
 *   2. stampExeIdentity (Windows only): stamp the Hermes icon + identity onto
 *      the packed Hermes.exe via rcedit (delegated to set-exe-identity.cjs), so
 *      the branded exe can never silently revert to the stock "Electron"
 *      icon/name (the bug when the stamp lived only in install.ps1, which the
 *      update path doesn't use). rcedit edits PE resources, irrelevant on
 *      macOS/Linux where identity comes from Info.plist / desktop entry.
 *      Best-effort: a stamp failure must never fail an otherwise-good build
 *      (worst case is the stock icon, not a broken app), so we log and resolve.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the product basename (e.g. 'Hermes')
 */

const fs = require('node:fs')
const path = require('node:path')
const asar = require('@electron/asar')

const { stampExeIdentity } = require('./set-exe-identity.cjs')

// Path to the packed app.asar for this target, or null when asar packaging is
// disabled (loose app dir — nothing to assert against here).
function resolveAsarPath(context) {
  const product = context.packager?.appInfo?.productFilename || 'Hermes'
  const asarPath =
    context.electronPlatformName === 'darwin'
      ? path.join(context.appOutDir, `${product}.app`, 'Contents', 'Resources', 'app.asar')
      : path.join(context.appOutDir, 'resources', 'app.asar')
  return fs.existsSync(asarPath) ? asarPath : null
}

// Fail the build if simple-git's full dependency closure isn't present in the
// packaged asar. Walks the closure from simple-git's own bundled package.json
// so the check tracks the real dependency graph rather than a hand-kept list.
function assertBundledJsDeps(context) {
  const asarPath = resolveAsarPath(context)
  if (!asarPath) return

  const readManifest = name =>
    JSON.parse(asar.extractFile(asarPath, `node_modules/${name}/package.json`).toString())

  const seen = new Set()
  const missing = []
  const stack = ['simple-git']
  while (stack.length) {
    const name = stack.pop()
    if (seen.has(name)) continue
    seen.add(name)
    let manifest
    try {
      manifest = readManifest(name)
    } catch {
      missing.push(name)
      continue
    }
    for (const dep of Object.keys(manifest.dependencies || {})) {
      if (!seen.has(dep)) stack.push(dep)
    }
  }

  if (missing.length) {
    throw new Error(
      `[after-pack] packaged app.asar is missing staged JS deps: ${missing.join(', ')}. ` +
        `The simple-git closure that scripts/stage-native-deps.cjs stages into ` +
        `build/native-deps/node_modules did not fully reach the bundle — check the ` +
        `package.json build.files entry (from: build/native-deps/node_modules, ` +
        `to: node_modules). Shipping this would crash the app on launch with ` +
        `"Cannot find module" (#52735).`
    )
  }

  console.log(`[after-pack] verified simple-git + ${seen.size - 1} closure deps bundled in app.asar`)
}

exports.default = async function afterPack(context) {
  assertBundledJsDeps(context)

  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(__dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
