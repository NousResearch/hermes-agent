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

/** @param {string} root */
function entryKinds(root) {
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
        kinds.set(name, `symlink to ${fs.readlinkSync(path.join(root, name))}`)
      } else if (entry.isFile()) {
        kinds.set(name, `file of ${fs.lstatSync(path.join(root, name)).size} bytes`)
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
 * @returns {string[]}
 */
function payloadDifferences(prepared, packaged) {
  const expected = entryKinds(prepared)
  const actual = entryKinds(packaged)
  const differences = []
  for (const [name, kind] of expected) {
    const found = actual.get(name)
    if (found === undefined) differences.push(`missing ${kind}: ${name}`)
    else if (found !== kind) differences.push(`${name}: ${kind} became ${found}`)
  }
  for (const [name, kind] of actual) {
    if (!expected.has(name)) differences.push(`unexpected ${kind}: ${name}`)
  }
  return differences.sort()
}

const SHOWN_DIFFERENCES = 40

/** @param {string} prepared @param {string} packaged */
export function assertPackagedPayload(prepared, packaged) {
  // electron-builder only warns when an extraResources source is missing, then packs without it.
  if (!fs.existsSync(packaged)) {
    throw new Error(`packaged payload ${packaged} is missing; the build maps ${prepared} there`)
  }
  const differences = payloadDifferences(prepared, packaged)
  if (differences.length === 0) return
  const hidden = differences.length - SHOWN_DIFFERENCES
  throw new Error(
    `packaged payload ${packaged} differs from the prepared payload ${prepared} in ${differences.length} entries:\n  ` +
      differences.slice(0, SHOWN_DIFFERENCES).join('\n  ') +
      (hidden > 0 ? `\n  ...and ${hidden} more` : '')
  )
}
