import fs from 'node:fs'
import path from 'node:path'

/**
 * @param {{ config: { extraResources?: Array<string | { from?: string, to?: string }> | null }, projectDir: string }} packager
 * @returns {string | null} null for a variant that ships no payload (Hermes Light)
 */
export function preparedPayloadSource({ config, projectDir }) {
  const mapping = config.extraResources?.find(entry => typeof entry === 'object' && entry.to === 'agent-payload')
  return mapping?.from ? path.resolve(projectDir, mapping.from) : null
}

// afterPack's signing rewrites binaries on macOS and Windows, and codesign seals bundles on macOS.
export const PACKED_PAYLOAD_CHANGES = {
  linux: { sizes: true, added: () => false },
  darwin: { sizes: false, added: (/** @type {string} */ name) => name.split('/').includes('_CodeSignature') },
  win32: { sizes: false, added: () => false }
}

/** @param {string} root @param {boolean} sizes */
function entryKinds(root, sizes) {
  /** @type {Map<string, string>} */
  const kinds = new Map()
  /** @param {string} relative */
  const visit = relative => {
    for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
      const name = path.posix.join(relative, entry.name)
      if (entry.isDirectory()) {
        kinds.set(name, 'directory')
        visit(name)
      } else if (entry.isSymbolicLink()) {
        kinds.set(name, 'symlink')
      } else if (entry.isFile()) {
        kinds.set(name, sizes ? `file of ${fs.lstatSync(path.join(root, name)).size} bytes` : 'file')
      } else {
        kinds.set(name, 'special file')
      }
    }
  }
  visit('')
  return kinds
}

/**
 * @param {string} prepared
 * @param {string} packaged
 * @param {{ sizes: boolean, added: (name: string) => boolean }} changes
 * @returns {string[]}
 */
export function payloadDifferences(prepared, packaged, { sizes, added }) {
  const expected = entryKinds(prepared, sizes)
  const actual = entryKinds(packaged, sizes)
  const differences = []
  for (const [name, kind] of expected) {
    const found = actual.get(name)
    if (found === undefined) differences.push(`missing ${kind}: ${name}`)
    else if (found !== kind) differences.push(`${name}: ${kind} became ${found}`)
  }
  for (const [name, kind] of actual) {
    if (!expected.has(name) && !added(name)) differences.push(`unexpected ${kind}: ${name}`)
  }
  return differences.sort()
}

const SHOWN_DIFFERENCES = 40

/**
 * @param {string} prepared
 * @param {string} packaged
 * @param {keyof typeof PACKED_PAYLOAD_CHANGES} platform
 */
export function assertPackagedPayload(prepared, packaged, platform) {
  const differences = payloadDifferences(prepared, packaged, PACKED_PAYLOAD_CHANGES[platform])
  if (differences.length === 0) return
  const hidden = differences.length - SHOWN_DIFFERENCES
  throw new Error(
    `packaged payload ${packaged} differs from the prepared payload ${prepared} in ${differences.length} entries:\n  ` +
      differences.slice(0, SHOWN_DIFFERENCES).join('\n  ') +
      (hidden > 0 ? `\n  ...and ${hidden} more` : '')
  )
}
