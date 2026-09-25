/**
 * after-pack.mjs — electron-builder afterPack hook.
 *
 * Per-platform post-pack work on the unpacked app: payload relocation, nested
 * Chromium + wheel signing on macOS, PE signature sanitizing and batch signing
 * on Windows. The exe identity stamp lives in after-extract.mjs (#105629).
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import path from 'node:path'
import fs from 'node:fs'
import { mkdir, readdir } from 'node:fs/promises'
import { runPython } from '../../../scripts/build/python.mjs'

import { batchSignAppTree } from './batch-sign-binaries.mjs'
import { rehashPayloadDigests } from './payload-digests.mjs'
import { resolveSigningIdentity, signNestedChromium } from './sign-nested-chromium.mjs'
import { signWheelZipMembers } from './sign-wheel-zips.mjs'
import { sanitizeTree } from './sanitize-pe-signatures.mjs'

/**
 * Restore the empty app-level localizations dropped during Electron extraction.
 * Runs after language filtering and before signing; the markers come from the
 * packaged framework, not the host's Electron (which may be another version).
 * Non-blocking: a failed restore leaves a usable package and says so.
 */
export async function restoreMacLocaleMarkers({ appOutDir, packager }) {
  try {
    const resources = packager.getResourcesDir(appOutDir)
    const framework = packager.getMacOsElectronFrameworkResourcesDir(appOutDir)
    const entries = await readdir(framework, { withFileTypes: true })
    // Chromium also ships grammatical-gender packs; these are not macOS locales.
    const locales = entries.filter(
      entry =>
        entry.isDirectory() && entry.name.endsWith('.lproj') && !/_(FEMININE|MASCULINE|NEUTER)\.lproj$/.test(entry.name)
    )
    await Promise.all(locales.map(entry => mkdir(path.join(resources, entry.name), { recursive: true })))
  } catch (error) {
    console.warn(
      `[after-pack] macOS locale markers were not restored: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

export default async function afterPack(context) {
  const platform = context.electronPlatformName
  const resources = platform === 'darwin'
    ? path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`, 'Contents', 'Resources')
    : path.join(context.appOutDir, 'resources')
  const payload = path.join(resources, 'agent-payload')
  if (platform !== 'win32' && fs.existsSync(path.join(payload, 'manifest.json'))) {
    runPython([
      path.resolve(import.meta.dirname, '../../../scripts/bundles/payload.py'), 'relocate', payload], { stdio: 'inherit' })
  }
  if (platform === 'darwin') {
    await restoreMacLocaleMarkers(context)
    if (fs.existsSync(payload)) {
      const entitlements = path.join(import.meta.dirname, '..', 'electron', 'entitlements.mac.inherit.plist')
      const { identity, keychain } = await resolveSigningIdentity(context.packager)
      const nested = signNestedChromium(payload, { entitlements, identity, keychain })
      console.log(
        `[after-pack] repaired ${nested.repaired} framework links; signed ${nested.signed} nested chromium targets` +
          (identity ? ` as ${identity}` : ' (no Developer ID in the builder keychain)')
      )
      // uv-cache wheel zips carry Mach-O members the notary validates but
      // electron-osx-sign cannot reach; sign them in place (see module doc).
      const wheels = signWheelZipMembers(payload, { identity, keychain })
      if (wheels.signed > 0) {
        console.log(
          `[after-pack] signed ${wheels.signed} Mach-O members across ${wheels.wheels} payload wheel zips` +
            (identity ? ` as ${identity}` : ' (no Developer ID in the builder keychain)'))
      }
      // The macOS signer refreshes this again before sealing the outer app.
      // Unsigned builds end here and still need final-byte facts.
      rehashPayloadDigests(payload)
    }
    return
  }
  if (platform === 'linux') {
    return
  }
  if (platform !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)

  // Repair dangling PE certificate tables BEFORE electron-builder signs the
  // tree. A stripped-but-still-declared signature makes signtool reject the
  // file with 0x800700C1, and AppxSIP inspects every PE inside the MSIX, so
  // one bad payload DLL fails the whole package. Unlike the stamp below this
  // is NOT best-effort: shipping past it means shipping an unsignable bundle.
  // this is a hack until https://github.com/astral-sh/python-build-standalone/pull/1217 is merged.
  const { scanned, repaired } = sanitizeTree(context.appOutDir)
  console.log(`[after-pack] ${scanned} PEs scanned, ${repaired.length} dangling certificate tables cleared`)
  for (const file of repaired) {
    console.log(`  ${file}`)
  }

  // The identity stamp already ran from afterExtract on the pristine exe
  // (scripts/after-extract.mjs, #105629); rcedit cannot commit to the
  // ASAR-integrity-rewritten PE we hold here.

  // Batch-sign every payload binary AFTER sanitize (above) and the rcedit
  // stamp: a dangling certificate table or a subsequent resource edit would
  // invalidate the signature. The product exe is excluded here and signed
  // per-file by the customSign hook (scripts/batch-sign-binaries.mjs) after
  // electron-builder's own rcedit + fuses pass. No-op with a loud warning when
  // the AZURE_SIGN_* environment is absent (unsigned/fork/canary lanes).
  await batchSignAppTree(context.appOutDir, exe, {
    config: context.packager.config,
    resourcesDir: context.packager.buildResourcesDir,
  })
  rehashPayloadDigests(payload)
}
