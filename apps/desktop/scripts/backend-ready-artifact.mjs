/**
 * Guard the packaged backend-readiness parser against artifact skew (#60772).
 *
 * The packaged app's `dist/electron-main.mjs` bundle must accept BOTH ready
 * tokens (`HERMES_BACKEND_READY` from a current backend, and the legacy
 * `HERMES_DASHBOARD_READY` from older ones). A stale artifact that matches
 * only one token boots a perfectly healthy backend and then kills it after
 * "Timed out waiting for Hermes backend port announcement" — invisible to
 * every source-level test, because the packaged bundle is the only thing that
 * ships. afterPack runs for every packed build, so this turns that class of
 * skew into a build failure instead of a user-side boot loop.
 */

import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

const PACKAGED_MAIN_MODULE = 'dist/electron-main.mjs'
const READY_TOKENS = ['HERMES_BACKEND_READY', 'HERMES_DASHBOARD_READY']
// The esbuild-bundled regex source for the dual-token readiness matcher. The
// bundle keeps the source literal verbatim, so the packaged text contains the
// (unexecuted) source form `HERMES_(?:BACKEND|DASHBOARD)_READY`; the parens
// and `?` are escaped here to match that substring as text.
const READY_MATCHER_SOURCE = /HERMES_\(\?:BACKEND\|DASHBOARD\)_READY/

function resolvePackagedAsarPath(context) {
  const appOutDir = context?.appOutDir
  if (!appOutDir || typeof appOutDir !== 'string') {
    throw new Error('electron-builder afterPack context is missing appOutDir')
  }

  if (context.electronPlatformName === 'darwin') {
    if (appOutDir.endsWith('.app')) {
      return path.join(appOutDir, 'Contents', 'Resources', 'app.asar')
    }
    const productName = context.packager?.appInfo?.productFilename || 'Hermes'
    return path.join(appOutDir, `${productName}.app`, 'Contents', 'Resources', 'app.asar')
  }

  return path.join(appOutDir, 'resources', 'app.asar')
}

function loadAsarModule() {
  try {
    return require('@electron/asar')
  } catch (err) {
    throw new Error(
      `Cannot inspect packaged app.asar because @electron/asar is unavailable: ${err.message}`
    )
  }
}

function unpackedPathForAsar(asarPath) {
  return path.join(`${asarPath}.unpacked`, PACKAGED_MAIN_MODULE)
}

function extractPackagedMainSource(asarPath, options = {}) {
  if (!fs.existsSync(asarPath)) {
    throw new Error(`Missing packaged app.asar: ${asarPath}`)
  }

  const unpackedPath = unpackedPathForAsar(asarPath)
  if (fs.existsSync(unpackedPath)) {
    return fs.readFileSync(unpackedPath, 'utf8')
  }

  const asarModule = options.asarModule ?? loadAsarModule()
  if (!asarModule || typeof asarModule.extractFile !== 'function') {
    throw new Error('@electron/asar module does not expose extractFile')
  }

  let source
  try {
    source = asarModule.extractFile(asarPath, PACKAGED_MAIN_MODULE)
  } catch (err) {
    throw new Error(
      `Could not extract ${PACKAGED_MAIN_MODULE} from ${asarPath}: ${err.message}`
    )
  }

  return Buffer.isBuffer(source) ? source.toString('utf8') : String(source)
}

function assertBackendReadyArtifactSourceAcceptsBothTokens(
  source,
  label = PACKAGED_MAIN_MODULE
) {
  if (!READY_MATCHER_SOURCE.test(source)) {
    throw new Error(
      `${label} does not contain a packaged readiness matcher accepting ` +
        READY_TOKENS.join(' and ')
    )
  }
}

function assertPackagedBackendReadyArtifact(asarPath, options = {}) {
  const source = extractPackagedMainSource(asarPath, options)
  assertBackendReadyArtifactSourceAcceptsBothTokens(source, PACKAGED_MAIN_MODULE)
  return {
    asarPath,
    module: PACKAGED_MAIN_MODULE,
    tokens: READY_TOKENS.slice()
  }
}

export {
  PACKAGED_MAIN_MODULE,
  READY_TOKENS,
  assertBackendReadyArtifactSourceAcceptsBothTokens,
  assertPackagedBackendReadyArtifact,
  extractPackagedMainSource,
  resolvePackagedAsarPath,
  unpackedPathForAsar
}
