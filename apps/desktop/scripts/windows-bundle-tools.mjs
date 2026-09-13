// Standalone bundle jobs need the same tools as electron-builder, even when
// GitHub evicts the build cache before the publishing job starts.
import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)

async function loadBuilderTools() {
  // app-builder-lib exports only its entry and ./internal. Resolve the
  // installed, lock-pinned package before loading its toolset implementation.
  const entry = pathToFileURL(require.resolve('app-builder-lib'))
  return import(new URL('./toolsets/winCodeSign.js', entry).href)
}

export async function ensureWindowsBundleTools({
  signing = false,
  config = require('../electron-builder.config.cjs'),
  resourcesDir = path.resolve(import.meta.dirname, '..', config.directories?.buildResources || 'build'),
  load = loadBuilderTools,
} = {}) {
  const builder = await load()
  const configured = config.toolsets?.winCodeSign
  const { kit } = await builder.getWindowsKitsBundle({ winCodeSign: configured, resourcesDir })
  const result = {
    makeappx: path.join(kit, 'makeappx.exe'),
    signtool: path.join(kit, 'signtool.exe'),
    dlib: null,
    dotnetRoot: null,
  }
  if (signing) {
    if (configured != null && typeof configured === 'object') {
      // This is electron-builder's custom-toolset contract: the owner
      // provides the dlib alongside the kit and manages its runtime.
      result.dlib = path.join(kit, 'Azure.CodeSigning.Dlib.dll')
    } else {
      const version = configured == null || configured === 'latest' ? builder.WIN_CODESIGN_LATEST : configured
      const ats = await builder.getAtsBundleDir(version)
      result.dlib = path.join(ats, path.basename(kit), 'Azure.CodeSigning.Dlib.dll')
      result.dotnetRoot = await builder.getDotnetRuntimeDir(version)
    }
  }
  const files = [result.makeappx, result.signtool]
  if (signing) files.push(result.dlib)
  if (result.dotnetRoot) files.push(path.join(result.dotnetRoot, 'dotnet.exe'))
  for (const file of files) {
    if (!fs.statSync(file).isFile()) throw new Error(`Windows bundle tool is not a file: ${file}`)
  }
  return result
}
