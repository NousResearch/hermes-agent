// Frontmatter parse for display: the YAML block between the leading `---`
// fences, flattened to top-level `key: value` rows (nested blocks render as
// their raw indented text). Display-only — never fed back to the backend.
export function parseFrontmatter(content: string): { body: string; meta: [string, string][] } {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(content)

  if (!match) {
    return { body: content, meta: [] }
  }

  const meta: [string, string][] = []
  let currentKey: null | string = null
  let block: string[] = []

  const flush = () => {
    if (currentKey !== null) {
      meta.push([currentKey, block.join('\n').trim()])
    }

    currentKey = null
    block = []
  }

  for (const line of match[1].split(/\r?\n/)) {
    const kv = /^(\w[\w-]*):\s?(.*)$/.exec(line)

    if (kv) {
      flush()
      currentKey = kv[1]
      block = kv[2] ? [kv[2]] : []
    } else if (currentKey !== null) {
      block.push(line.replace(/^ {2}/, ''))
    }
  }

  flush()

  return { body: content.slice(match[0].length), meta }
}
