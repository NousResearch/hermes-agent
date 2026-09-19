import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { fileDigest, treeDigest, preparationRequired } from './prepared-packaging.mjs'

/** @typedef {{ source: string, nativeDeps: string, platform?: string, arch?: string, nativeToolchain?: string }} NativeSelection */
/** @param {string} source @returns {string} */
function nativeIdentity(source) {
  return createHash('sha256').update(JSON.stringify([
    fileDigest(path.join(source, 'package-lock.json')),
    fileDigest(path.join(source, 'apps/desktop/package.json')),
    fileDigest(path.join(import.meta.dirname, 'stage-native-deps.mjs')),
    fileDigest(path.join(import.meta.dirname, 'prepared-native-deps.mjs')),
  ])).digest('hex')
}

/**
 * The sidecar stays outside node_modules so it never ships in the application.
 * @param {{ source: string, out: string, platform: string, arch: string, nativeToolchain?: string }} inputs
 * @returns {void}
 */
export function recordNativeInputs({ source, out, platform, arch, nativeToolchain }) {
  fs.writeFileSync(`${out}.prepared.json`, JSON.stringify({
    schema: 1, source: fs.realpathSync(source), out: fs.realpathSync(out),
    platform, arch, nativeToolchain, identity: nativeIdentity(source), digest: treeDigest(out),
  }) + '\n')
}

/** @param {NativeSelection} inputs @returns {string} */
export function readNativeInputs({ source, nativeDeps, platform = process.platform, arch = process.arch, nativeToolchain }) {
  try {
    const record = JSON.parse(fs.readFileSync(`${nativeDeps}.prepared.json`, 'utf8'))
    if (record.schema !== 1 || record.source !== fs.realpathSync(source) || record.out !== fs.realpathSync(nativeDeps) ||
        record.platform !== platform || record.arch !== arch ||
        (nativeToolchain !== undefined && record.nativeToolchain !== nativeToolchain) ||
        record.identity !== nativeIdentity(source) || record.digest !== treeDigest(nativeDeps)) {
      throw preparationRequired('Stale or foreign native inputs')
    }
    if (!fs.statSync(path.join(nativeDeps, 'node-pty/package.json')).isFile()) throw preparationRequired('Missing prepared node-pty')
    return record.out
  } catch (error) {
    throw preparationRequired(`Cannot consume native inputs: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/** @param {NativeSelection & { out: string }} inputs @returns {void} */
export function copyNativeInputs({ out, ...inputs }) {
  const nativeDeps = readNativeInputs(inputs)
  const destination = path.resolve(out)
  if (destination === nativeDeps || destination.startsWith(nativeDeps + path.sep) || nativeDeps.startsWith(destination + path.sep)) {
    throw preparationRequired('Native input and product directories overlap')
  }
  fs.rmSync(destination, { recursive: true, force: true })
  fs.cpSync(nativeDeps, destination, { recursive: true, dereference: true })
}
