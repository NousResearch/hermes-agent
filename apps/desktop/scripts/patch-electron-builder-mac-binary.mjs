import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'

if (process.platform !== 'darwin') {
  process.exit(0)
}

const require = createRequire(import.meta.url)

const desktopRoot = path.resolve(import.meta.dirname, '..')
const repoRoot = path.resolve(desktopRoot, '..', '..')
const electronMacPath = path.join(repoRoot, 'node_modules', 'app-builder-lib', 'out', 'electron', 'electronMac.js')

const marker = 'hermes-macos-electron-binary-fallback'

function patchPlistParser() {
  let plistParserPath
  try {
    plistParserPath = require.resolve('plist/lib/parse.js')
  } catch {
    console.warn('[patch-electron-builder] plist parser not found')
    return
  }

  const plistSource = fs.readFileSync(plistParserPath, 'utf8')
  const plistMarker = 'hermes-plist-mime-type'
  if (plistSource.includes(plistMarker)) {
    console.log('[patch-electron-builder] plist MIME-type fallback already applied')
    return
  }

  const plistNeedle = 'parseFromString(xml);'
  if (!plistSource.includes(plistNeedle)) {
    console.warn('[patch-electron-builder] skipped: expected plist parser shape not found')
    return
  }

  fs.writeFileSync(
    plistParserPath,
    plistSource.replace(plistNeedle, `parseFromString(xml, 'text/xml'); // ${plistMarker}`)
  )
  console.log('[patch-electron-builder] applied plist MIME-type fallback')
}

patchPlistParser()

const needle = `    await Promise.all([
        doRename(path.join(contentsPath, "MacOS"), electronBranding.productName, appPlist.CFBundleExecutable),
        (0, builder_util_1.unlinkIfExists)(path.join(appOutDir, "LICENSE")),
        (0, builder_util_1.unlinkIfExists)(path.join(appOutDir, "LICENSES.chromium.html")),
    ]);`
const replacement = `    // ${marker}: electron-builder 26.8.x can sometimes copy
    // Electron.app without its main MacOS/Electron binary before this rename.
    // Restore it from the installed Electron runtime so local desktop installs
    // do not fail with ENOENT during macOS arm64 packaging.
    const macosDir = path.join(contentsPath, "MacOS");
    const bundledElectronBinary = path.join(macosDir, electronBranding.productName);
    if (!fs.existsSync(bundledElectronBinary)) {
        const candidates = [
            path.join(packager.info.framework.distMacOsAppName, "Contents", "MacOS", electronBranding.productName),
            // npm may nest the workspace-only electron devDep under
            // apps/desktop/node_modules (process.cwd() during pack), or hoist
            // it to the repo root. Try the workspace-local install first, then
            // the root hoist, so the fallback works under either layout.
            path.join(process.cwd(), "node_modules", "electron", "dist", "Electron.app", "Contents", "MacOS", electronBranding.productName),
            path.join(process.cwd(), "..", "..", "node_modules", "electron", "dist", "Electron.app", "Contents", "MacOS", electronBranding.productName),
        ];
        const sourceBinary = candidates.find(candidate => fs.existsSync(candidate));
        if (sourceBinary == null) {
            throw new Error("Electron binary missing from packaged app and Electron runtime: " + bundledElectronBinary);
        }
        await (0, promises_1.copyFile)(sourceBinary, bundledElectronBinary);
        await (0, promises_1.chmod)(bundledElectronBinary, 0o755);
    }
    await Promise.all([
        doRename(macosDir, electronBranding.productName, appPlist.CFBundleExecutable),
        (0, builder_util_1.unlinkIfExists)(path.join(appOutDir, "LICENSE")),
        (0, builder_util_1.unlinkIfExists)(path.join(appOutDir, "LICENSES.chromium.html")),
    ]);`

if (!fs.existsSync(electronMacPath)) {
  console.warn(`[patch-electron-builder] skipped: ${electronMacPath} not found`)
  process.exit(0)
}

const source = fs.readFileSync(electronMacPath, 'utf8')
if (source.includes(marker)) {
  console.log('[patch-electron-builder] macOS Electron binary fallback already applied')
  process.exit(0)
}

if (!source.includes(needle)) {
  console.warn('[patch-electron-builder] skipped: expected electronMac.js shape not found')
  process.exit(0)
}

fs.writeFileSync(electronMacPath, source.replace(needle, replacement))
console.log('[patch-electron-builder] applied macOS Electron binary fallback')
