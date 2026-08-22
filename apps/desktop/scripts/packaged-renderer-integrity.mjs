import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { listPackage, statFile } from '@electron/asar'

const RENDERER_PREFIX = 'dist/'
const POST_PACK_MUTABLE_PREFIX = 'dist/node_modules/'

function normalizeAsarPath(value) {
  return value.replace(/\\/g, '/').replace(/^\/+/, '')
}

function isRendererFile(value) {
  return value.startsWith(RENDERER_PREFIX)
}

function isPostPackMutable(value) {
  // macOS code signing rewrites native node-pty/get-windows payloads after
  // electron-builder creates the ASAR metadata. Their paths must still match,
  // but their post-sign sizes and hashes legitimately differ from the index.
  return value.startsWith(POST_PACK_MUTABLE_PREFIX)
}

function collectDiskFiles(root, relative = '') {
  const files = []
  for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
    const child = relative ? path.join(relative, entry.name) : entry.name
    if (entry.isDirectory()) {
      files.push(...collectDiskFiles(root, child))
    } else if (entry.isFile()) {
      files.push(normalizeAsarPath(path.posix.join('dist', child)))
    }
  }
  return files.sort()
}

function sha256File(filePath) {
  const hash = createHash('sha256')
  hash.update(fs.readFileSync(filePath))
  return hash.digest('hex')
}

function diskPathFor(unpackedDist, archivePath) {
  return path.join(unpackedDist, ...archivePath.slice('dist/'.length).split('/'))
}

export function packagedResourcesDir(appOutDir, platform, productFilename = 'Hermes') {
  return platform === 'darwin'
    ? path.join(appOutDir, `${productFilename}.app`, 'Contents', 'Resources')
    : path.join(appOutDir, 'resources')
}

export function verifyPackagedRenderer(resourcesDir) {
  const asarPath = path.join(resourcesDir, 'app.asar')
  const unpackedDist = path.join(resourcesDir, 'app.asar.unpacked', 'dist')
  const errors = []

  if (!fs.existsSync(asarPath)) {
    return { ok: false, errors: [`missing ASAR archive: ${asarPath}`] }
  }
  if (!fs.existsSync(path.join(unpackedDist, 'index.html'))) {
    return {
      ok: false,
      errors: [`missing unpacked renderer entry point: ${path.join(unpackedDist, 'index.html')}`]
    }
  }

  let archiveEntries
  try {
    archiveEntries = listPackage(asarPath)
  } catch (err) {
    return { ok: false, errors: [`cannot read ASAR index: ${err.message}`] }
  }

  const indexedFiles = new Map()
  for (const rawEntry of archiveEntries) {
    const entry = normalizeAsarPath(rawEntry)
    if (!isRendererFile(entry)) continue
    try {
      const metadata = statFile(asarPath, entry)
      if (!('files' in metadata)) indexedFiles.set(entry, metadata)
    } catch (err) {
      errors.push(`cannot read ASAR metadata for ${entry}: ${err.message}`)
    }
  }

  const diskFiles = new Set(collectDiskFiles(unpackedDist).filter(isRendererFile))

  for (const entry of diskFiles) {
    if (!indexedFiles.has(entry)) {
      errors.push(`unpacked renderer file is absent from ASAR index: ${entry}`)
    }
  }
  for (const [entry, metadata] of indexedFiles) {
    const filePath = diskPathFor(unpackedDist, entry)
    if (!diskFiles.has(entry)) {
      errors.push(`ASAR index points to a missing unpacked renderer file: ${entry}`)
      continue
    }
    if (metadata.unpacked !== true) {
      errors.push(`renderer file is not marked unpacked in ASAR index: ${entry}`)
      continue
    }
    if (isPostPackMutable(entry)) continue
    const diskSize = fs.statSync(filePath).size
    if (typeof metadata.size === 'number' && metadata.size !== diskSize) {
      errors.push(`renderer size mismatch for ${entry}: ASAR=${metadata.size}, disk=${diskSize}`)
      continue
    }
    const expectedHash = metadata.integrity?.hash?.toLowerCase()
    if (!expectedHash) {
      errors.push(`renderer file has no ASAR integrity hash: ${entry}`)
    } else if (sha256File(filePath) !== expectedHash) {
      errors.push(`renderer hash mismatch between ASAR index and disk: ${entry}`)
    }
  }

  return {
    ok: errors.length === 0,
    errors,
    indexedFileCount: indexedFiles.size,
    unpackedFileCount: diskFiles.size
  }
}

export function formatPackagedRendererErrors(errors, limit = 8) {
  const visible = errors.slice(0, limit)
  const omitted = errors.length - visible.length
  return [...visible, ...(omitted > 0 ? [`... and ${omitted} more renderer integrity error(s)`] : [])].join('\n')
}

export function assertPackagedRendererIntegrity(resourcesDir) {
  const result = verifyPackagedRenderer(resourcesDir)
  if (!result.ok) {
    throw new Error(formatPackagedRendererErrors(result.errors))
  }
  return result
}

const invokedDirectly = process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url

if (invokedDirectly) {
  const resourcesDir = process.argv[2]
  if (!resourcesDir) {
    console.error('usage: node packaged-renderer-integrity.mjs <resources-dir>')
    process.exitCode = 2
  } else {
    const result = verifyPackagedRenderer(path.resolve(resourcesDir))
    if (!result.ok) {
      console.error(formatPackagedRendererErrors(result.errors))
      process.exitCode = 1
    }
  }
}
