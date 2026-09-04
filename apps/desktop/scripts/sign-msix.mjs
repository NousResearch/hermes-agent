// electron-builder custom win.sign hook: Azure Trusted Signing for the MSIX
// package only. Windows install validation checks the package signature
// (AppxSignature.p7x over AppxBlockMap.xml); inner files are covered by the
// block-map hashes, NOT per-file Authenticode, so Hermes.exe and every
// payload binary stay unsigned and this hook signs exactly one artifact per
// build.
//
// Wired from electron-builder.config.cjs as
//   win.sign = { type: 'signtool', sign: './scripts/sign-msix.mjs', ... }
// The Azure endpoint/account/profile do NOT ride through the config: this
// module reads the AZURE_SIGN_* environment variables directly
// (azureConfigFromEnv) and hands them to app-builder-lib's own
// WindowsSignAzureManager, so the actual signing path is byte-for-byte the
// one a plain { type: 'azure' } config would run.
//
// The Store-submission variant (artifactName "Store-" prefixed) is the one
// exception: its manifest Publisher is the Partner Center publisher ID
// (CN=EE6D86E4-...), which no signable cert subject can match — CA/B CSBR
// requires the legal entity's validated name and Artifact Signing cannot
// customize CN — so SignerSign rejects the package with ERROR_BAD_FORMAT
// (0x8007000B). Partner Center re-signs the package with the Microsoft
// Store certificate on ingestion, so the Store- msix ships unsigned.

import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)

// Artifacts that carry the only signature Windows validates for a packaged
// app. A .msixbundle envelope is included for the single-invocation bundle
// case (the envelope signature covers the inner .msix packages).
const SIGNED_ARTIFACT_EXTENSIONS = ['.msix', '.msixbundle']

// Store-submission artifacts are not ATS-signable (publisher mismatch, see
// the header) and don't need a signature (Partner Center signs on
// ingestion). Anything with this basename prefix is left unsigned.
const STORE_ARTIFACT_PREFIX = 'Store-'

/**
 * Whether `file` is a signing target. Everything except the MSIX package is
 * covered by the package's block-map hashes and must stay untouched —
 * signing it after packaging would break the hash. The Store-submission
 * variant is excluded too: it cannot carry an ATS signature and does not
 * need one.
 */
export function shouldSignFile(file) {
  if (path.basename(file).startsWith(STORE_ARTIFACT_PREFIX)) return false
  const lower = file.toLowerCase()
  return SIGNED_ARTIFACT_EXTENSIONS.some(ext => lower.endsWith(ext))
}

/**
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {{ type: 'azure' } & Record<string, string | undefined>} the
 *   win.sign azure configuration, composed from the environment.
 */
export function azureConfigFromEnv(env = process.env) {
  return {
    type: 'azure',
    endpoint: env.AZURE_SIGN_ENDPOINT,
    codeSigningAccountName: env.AZURE_SIGN_ACCOUNT,
    certificateProfileName: env.AZURE_SIGN_PROFILE,
    publisherName: env.AZURE_SIGN_PUBLISHER
  }
}

// app-builder-lib's exports map exposes only "." and "./internal";
// import('app-builder-lib/dist/...') throws ERR_PACKAGE_PATH_NOT_EXPORTED.
// Resolve the class the way run-electron-builder.mjs finds the CLI: resolve
// the entry module, walk up to the package root, then direct-file import
// (file URLs bypass the exports map). The constructibility of this class is
// pinned by the tripwire test in sign-msix.test.mjs, so a builder bump
// that moves it fails js-tests instead of a release build.
async function loadAzureManagerClass() {
  const entry = require.resolve('app-builder-lib')
  let root = path.dirname(entry)
  while (!fs.existsSync(path.join(root, 'package.json'))) {
    const parent = path.dirname(root)
    if (parent === root) throw new Error('app-builder-lib package root not found')
    root = parent
  }
  const mod = await import(
    pathToFileURL(path.join(root, 'dist', 'codeSign', 'win', 'windowsSignAzureManager.js')).href
  )
  return mod.WindowsSignAzureManager
}

// One manager per process: electron-builder calls the hook once per file,
// and the manager memoizes toolset downloads / dlib metadata behind it.
let managerPromise = null

function azureManager(packager) {
  if (managerPromise == null) {
    managerPromise = (async () => {
      const WindowsSignAzureManager = await loadAzureManagerClass()
      // The manager's constructor re-derives the signing config from
      // packager.platformOptions.sign and throws unless type === 'azure' —
      // but our config's win.sign is the { type: 'signtool' } hook wiring.
      // Shim the packager with the azure config composed from the
      // environment; everything else (config.toolsets, buildResourcesDir,
      // getTempFile) delegates to the real packager via the prototype.
      const shim = Object.create(packager)
      Object.defineProperty(shim, 'platformOptions', {
        value: { ...packager.platformOptions, sign: azureConfigFromEnv() }
      })
      const manager = new WindowsSignAzureManager(shim)
      // No-op on the modern signtool /dlib path (winCodeSign >= 1.3.0);
      // installs the legacy PowerShell module otherwise.
      await manager.initialize()
      return manager
    })()
  }
  return managerPromise
}

/**
 * Sign one file on app-builder-lib's own WindowsSignAzureManager — the exact
 * path a plain { type: 'azure' } config runs. Used by the sign-msix hook for
 * the package itself and by batch-sign-binaries.mjs's customSign hook for the
 * product exe (which must be signed after rcedit, not in the afterPack batch).
 *
 * @param {string} file
 * @param {any} packager WinPackager
 */
export async function azureSignFile(file, packager) {
  const mgr = await azureManager(packager)
  // signFileWithDlib reads only options.path (plus the manager's own
  // signing config), so platformOptions is sufficient here.
  await mgr.signFile({ path: file, options: packager.platformOptions })
}

/**
 * The electron-builder custom sign hook.
 *
 * @param {{ path: string }} configuration CustomWindowsSignTaskConfiguration
 * @param {any} packager WinPackager
 */
export default async function sign(configuration, packager) {
  if (path.basename(configuration.path).startsWith(STORE_ARTIFACT_PREFIX)) {
    console.log(`[sign-msix] skip Store- submission package (Partner Center signs on ingestion): ${configuration.path}`)
    return
  }
  if (!shouldSignFile(configuration.path)) {
    console.log(`[sign-msix] skip non-MSIX (covered by package block map): ${configuration.path}`)
    return
  }

  await azureSignFile(configuration.path, packager)
}
