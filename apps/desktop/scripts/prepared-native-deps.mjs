import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { fileDigest, treeDigest, preparationRequired } from './prepared-packaging.mjs'
import { hudModifierBinaryRelativePath, warnWindowsHudUnavailable } from './build-hud-modifier-monitor.mjs'

/** @typedef {{ source: string, nativeDeps: string, platform?: string, arch?: string, nativeToolchain?: string }} NativeSelection */
/** @param {string} source @returns {string} */
function nativeIdentity(source) {
  return createHash('sha256').update(JSON.stringify([
    fileDigest(path.join(source, 'package-lock.json')),
    fileDigest(path.join(source, 'apps/desktop/package.json')),
    fileDigest(path.join(import.meta.dirname, 'stage-native-deps.mjs')),
    fileDigest(path.join(import.meta.dirname, 'prepared-native-deps.mjs')),
    ...['build-command-screenshot-monitor.mjs', 'build-hud-modifier-monitor.mjs']
      .map(name => fileDigest(path.join(import.meta.dirname, name))),
    ...['command-screenshot-monitor.m', 'hud-modifier-gesture.h', 'hud-modifier-gesture.cs',
      'hud-modifier-monitor.m', 'hud-modifier-monitor-win.cs', 'hud-modifier-monitor-x11.c']
      .map(name => fileDigest(path.join(source, 'apps/desktop/electron/native', name))),
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
    // Only this optional executable may disappear. All other files, modes and
    // symlinks retain their integrity checks, including other native helpers.
    ...(platform === 'win32' ? { withoutHudDigest: treeDigest(out, [hudModifierBinaryRelativePath(platform, arch)]) } : {}),
  }) + '\n')
}

/** @param {NativeSelection} inputs @returns {string} */
export function readNativeInputs({ source, nativeDeps, platform = process.platform, arch = process.arch, nativeToolchain }) {
  try {
    const record = JSON.parse(fs.readFileSync(`${nativeDeps}.prepared.json`, 'utf8'))
    if (record.schema !== 1 || record.source !== fs.realpathSync(source) || record.out !== fs.realpathSync(nativeDeps) ||
        record.platform !== platform || record.arch !== arch ||
        (nativeToolchain !== undefined && record.nativeToolchain !== nativeToolchain) ||
        record.identity !== nativeIdentity(source)) {
      throw preparationRequired('Stale or foreign native inputs')
    }
    if (record.digest !== treeDigest(nativeDeps)) {
      const optional = hudModifierBinaryRelativePath(platform, arch)
      const missingHudOnly = platform === 'win32' &&
        !fs.lstatSync(path.join(nativeDeps, optional), { throwIfNoEntry: false }) &&
        typeof record.withoutHudDigest === 'string' &&
        record.withoutHudDigest === treeDigest(nativeDeps, [optional])
      if (!missingHudOnly) throw preparationRequired('Stale or foreign native inputs')
    }
    if (!fs.statSync(path.join(nativeDeps, 'node-pty/package.json')).isFile()) throw preparationRequired('Missing prepared node-pty')
    return record.out
  } catch (error) {
    throw preparationRequired(`Cannot consume native inputs: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/** @param {NativeSelection & { out: string, omitWindowsHud?: boolean }} inputs @returns {void} */
export function copyNativeInputs({ out, ...inputs }) {
  copyNativeTree({ ...inputs, nativeDeps: readNativeInputs(inputs), out })
}

/** Copy admitted modules and executable resources without rebuilding either.
 * @param {{ nativeDeps: string, out: string, platform?: string, arch?: string, omitWindowsHud?: boolean }} inputs out is the product's node_modules.
 * @returns {void}
 */
export function copyNativeTree({ nativeDeps, out, platform = process.platform, arch = process.arch, omitWindowsHud = false }) {
  nativeDeps = fs.realpathSync(nativeDeps)
  const destination = path.resolve(out)
  const helpers = path.join(path.dirname(destination), 'native')
  for (const target of [destination, helpers]) {
    if (target === nativeDeps || target.startsWith(nativeDeps + path.sep) || nativeDeps.startsWith(target + path.sep)) {
      throw preparationRequired('Native input and product directories overlap')
    }
  }
  fs.rmSync(destination, { recursive: true, force: true })
  fs.rmSync(helpers, { recursive: true, force: true })
  const preparedHelpers = path.join(nativeDeps, 'native')
  fs.cpSync(nativeDeps, destination, { recursive: true, dereference: true,
    filter: file => file !== preparedHelpers })
  const hudRelative = hudModifierBinaryRelativePath(platform, arch)
  const skipHud = platform === 'win32' && omitWindowsHud
  const sourceHud = path.join(nativeDeps, hudRelative)
  const copyHelpers = omit => fs.cpSync(preparedHelpers, helpers, { recursive: true, dereference: true,
    filter: file => !omit || file !== sourceHud })
  if (fs.existsSync(preparedHelpers)) {
    try {
      copyHelpers(skipHud)
    } catch (error) {
      // A quarantine may race cpSync too. Never swallow another file's error.
      if (platform !== 'win32' || error.code !== 'ENOENT' || error.path !== sourceHud || fs.existsSync(sourceHud)) throw error
      fs.rmSync(path.join(helpers, path.relative(preparedHelpers, sourceHud)), { force: true })
      copyHelpers(true)
    }
  }
  const stagedHud = path.join(path.dirname(destination), hudRelative)
  if (platform === 'win32' && !fs.existsSync(stagedHud)) warnWindowsHudUnavailable(stagedHud)
}
