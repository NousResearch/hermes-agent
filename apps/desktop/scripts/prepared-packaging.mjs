import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'

/** @typedef {{ path: string, digest: string }} PreparedFile */
/** @typedef {{ sevenZip: string, icons: string, winCodeSign?: string, appimage?: string, fpm?: string }} PackagingToolsets */
/** @typedef {{ schema: number, source: string, out: string, identity: string, target: string, formats: string[], electron: string, toolsets: PackagingToolsets, windows: import('./windows-bundle-tools.mjs').WindowsBundleTools | null, dmgbuild: string | null, files: PreparedFile[] }} PreparedPackaging */

/** @param {string} message @returns {Error} */
export function preparationRequired(message) {
  return new Error(`${message}; run preparation again`)
}

/** @param {string} file @returns {string} */
export function fileDigest(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex')
}

/**
 * Include symlink destinations and modes, not timestamps. A copied supplier
 * may use internal symlinks; external links would make its receipt incomplete.
 * @param {string} root
 * @returns {string}
 */
export function treeDigest(root) {
  const hash = createHash('sha256')
  const canonicalRoot = fs.realpathSync(root)
  /** @param {string} entry @returns {void} */
  function visit(entry) {
    const relative = path.relative(root, entry)
    const stat = fs.lstatSync(entry)
    hash.update(JSON.stringify([relative, stat.mode & 0o777]))
    if (stat.isSymbolicLink()) {
      const target = fs.realpathSync(entry)
      if (target !== canonicalRoot && !target.startsWith(canonicalRoot + path.sep)) {
        throw preparationRequired(`Prepared input contains an external link: ${entry}`)
      }
      hash.update(fs.readlinkSync(entry))
    } else if (stat.isDirectory()) {
      for (const name of fs.readdirSync(entry).sort()) visit(path.join(entry, name))
    } else if (stat.isFile()) {
      hash.update(fileDigest(entry))
    } else {
      throw preparationRequired(`Unsupported prepared input: ${entry}`)
    }
  }
  visit(root)
  return hash.digest('hex')
}

/** @param {string} source @returns {string} */
export function packagingIdentity(source) {
  const files = ['package-lock.json', 'apps/desktop/package.json', 'apps/desktop/electron-builder.config.cjs']
  const recipe = ['prepare-packaging-tools.mjs', 'prepared-packaging.mjs', 'prepare-dmgbuild.mjs', 'prepare_dmgbuild.py', 'windows-bundle-tools.mjs', 'run-electron-builder.mjs']
  return createHash('sha256').update(JSON.stringify([
    ...files.map(file => fileDigest(path.join(source, file))),
    ...recipe.map(file => fileDigest(path.join(import.meta.dirname, file))),
    fileDigest(path.join(import.meta.dirname, '../../../pm/lock.json')),
  ])).digest('hex')
}

/** @param {string} root @param {string} file @returns {void} */
function assertOwned(root, file) {
  const canonicalRoot = fs.realpathSync(root)
  const canonicalFile = fs.realpathSync(file)
  if (!path.isAbsolute(file) || !canonicalFile.startsWith(canonicalRoot + path.sep)) {
    throw preparationRequired(`Prepared input is outside its work directory: ${file}`)
  }
}

/**
 * Publish only after every selected supplier completed. The receipt is job-local,
 * not a cache attestation; trusted cache writers remain a prerequisite.
 * @param {{ source: string, out: string, target: string, formats: string[], electron: string, toolsets: PackagingToolsets, windows?: import('./windows-bundle-tools.mjs').WindowsBundleTools | null, dmgbuild?: string | null }} inputs
 * @returns {Promise<string>}
 */
export async function publishPackagingInputs(inputs) {
  const out = fs.realpathSync(inputs.out)
  const paths = [inputs.electron, ...Object.values(inputs.toolsets)]
  if (inputs.windows?.dotnetRoot) paths.push(inputs.windows.dotnetRoot)
  if (inputs.dmgbuild) paths.push(path.dirname(inputs.dmgbuild))
  const files = paths.map(file => {
    assertOwned(out, file)
    return { path: file, digest: treeDigest(file) }
  })
  /** @type {PreparedPackaging} */
  const result = {
    schema: 1, source: fs.realpathSync(inputs.source), out,
    identity: packagingIdentity(inputs.source), target: inputs.target,
    formats: inputs.formats, electron: inputs.electron, toolsets: inputs.toolsets,
    windows: inputs.windows ?? null, dmgbuild: inputs.dmgbuild ?? null, files,
  }
  const manifest = path.join(out, 'prepared.json')
  fs.writeFileSync(`${manifest}.tmp`, JSON.stringify(result, null, 2) + '\n')
  fs.renameSync(`${manifest}.tmp`, manifest)
  return manifest
}

/**
 * Read-only admission. No supplier or installer may be imported on this path.
 * @param {string} manifest
 * @param {string} source
 * @param {string} [target]
 * @returns {PreparedPackaging}
 */
export function readPackagingInputs(manifest, source, target = `${process.platform}-${process.arch}`) {
  try {
    /** @type {PreparedPackaging} */
    const result = JSON.parse(fs.readFileSync(manifest, 'utf8'))
    if (result.schema !== 1 || result.source !== fs.realpathSync(source) || result.out !== fs.realpathSync(path.dirname(manifest)) ||
        result.target !== target || result.identity !== packagingIdentity(source)) {
      throw preparationRequired('Stale or foreign packaging inputs')
    }
    const required = [result.electron, result.toolsets.sevenZip, result.toolsets.icons, ...Object.values(result.toolsets)]
    if (target.startsWith('win32-')) {
      if (!result.windows || !result.toolsets.winCodeSign || !result.windows.dotnetRoot) throw preparationRequired('Missing Windows tool selection')
      required.push(result.windows.dotnetRoot)
    }
    if (result.formats.includes('dmg')) {
      if (!result.dmgbuild) throw preparationRequired('Missing prepared dmgbuild')
      required.push(path.dirname(result.dmgbuild))
    }
    for (const file of new Set(required)) {
      assertOwned(result.out, file)
      const record = result.files.find(entry => entry.path === file)
      if (!record || record.digest !== treeDigest(file)) throw preparationRequired(`Missing or changed packaging input: ${file}`)
    }
    return result
  } catch (error) {
    throw preparationRequired(`Cannot consume packaging inputs ${manifest}: ${error instanceof Error ? error.message : String(error)}`)
  }
}
