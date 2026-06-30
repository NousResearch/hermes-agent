/**
 * after-pack.cjs — electron-builder afterPack hook.
 *
 * Stamps the Reuben icon + identity onto the packed Windows Reuben.exe via
 * rcedit (delegated to set-exe-identity.cjs). This runs for EVERY packed build
 * so the branded exe can never silently revert to the stock "Electron"
 * icon/name.
 *
 * Windows-only: rcedit edits PE resources, irrelevant on macOS/Linux where the
 * app identity comes from the bundle Info.plist / desktop entry. Best-effort:
 * a stamp failure must never fail an otherwise-good build (worst case is the
 * stock icon, not a broken app), so we log and resolve rather than throw.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Reuben')
 */

const path = require('node:path')

const { stampExeIdentity } = require('./set-exe-identity.cjs')

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Reuben'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(__dirname, '..')

  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); ${productName}.exe keeps the stock Electron icon`)
  }
}
