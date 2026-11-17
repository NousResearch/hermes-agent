// Reads the importers of pnpm-lock.yaml without a YAML parser: node-deps.mjs runs
// before any dependency (js-yaml included) is installed. pnpm writes this file
// itself in a fixed shape (two-space indent, one key per line); anything that
// does not fit that shape is an error, never a guess.

function unquote(value) {
  if (value?.startsWith("'") && value.endsWith("'")) return value.slice(1, -1).replaceAll("''", "'")
  if (value?.startsWith('"') && value.endsWith('"')) return JSON.parse(value)
  return value
}

/**
 * The `importers` map: `{ [projectDir]: { [field]: value | { [name]: { [key]: value } } } }`.
 * The repository lock is one YAML document (`pmOnFail: ignore` keeps pnpm from prepending a
 * `packageManagerDependencies` document); several documents are refused, because lockfile
 * scanners read only the first one and would scan the pin instead of the workspace.
 */
export function lockImporters(text) {
  const documents = text.split(/^---\r?\n/m).filter(part => part.trim())
  if (documents.length !== 1) throw new Error(`pnpm-lock.yaml: expected one YAML document, found ${documents.length}`)
  const [document] = documents
  const version = /^lockfileVersion: '([^']+)'$/m.exec(document)?.[1]
  if (version !== '9.0') throw new Error(`pnpm-lock.yaml: unsupported lockfileVersion ${version ?? '<missing>'}`)
  const lines = document.split(/\r?\n/)
  const start = lines.indexOf('importers:')
  if (start < 0) throw new Error('pnpm-lock.yaml: no importers section')
  const importers = {}
  let importer
  let field
  let entry
  for (const line of lines.slice(start + 1)) {
    if (!line.trim()) continue
    if (!line.startsWith(' ')) break
    const match = /^( *)(.+?):(?: (.*))?$/.exec(line)
    if (!match) throw new Error(`pnpm-lock.yaml: unreadable importers line: ${line}`)
    const [, indent, rawKey, rawValue] = match
    const key = unquote(rawKey)
    const value = unquote(rawValue)
    // A project without dependencies is written as `path: {}`.
    if (indent.length === 2 && (value === undefined || value === '{}')) {
      importer = importers[key] = {}
    } else if (indent.length === 4 && importer) {
      field = undefined
      if (value === undefined) field = importer[key] = {}
      else importer[key] = value
    } else if (indent.length === 6 && field && value === undefined) {
      entry = field[key] = {}
    } else if (indent.length === 8 && entry && value !== undefined) {
      entry[key] = value
    } else {
      throw new Error(`pnpm-lock.yaml: unexpected importers line: ${line}`)
    }
  }
  return importers
}
