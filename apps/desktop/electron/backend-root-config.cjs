const os = require('node:os')
const path = require('node:path')

// parseDesktopBackendRoot(configText) — read ONLY `desktop.backend_root` from raw config.yaml text with a
// pinned, indentation-anchored grammar (SPEC §4.2). This is deliberately NOT a general YAML parser: the
// desktop app has no YAML dependency and must not add one just to read a single scalar path. Everything
// outside the accepted grammar deterministically returns null (→ the caller auto-resolves the backend root),
// so a malformed or ambiguous config can never silently select the wrong tree.
//
// Accepted grammar:
//   1. A top-level (column-0) `desktop:` key opens the block; only a `backend_root:` nested UNDER it (deeper
//      indent, before the next column-0 key) is read. A `backend_root:` under any other block is ignored.
//   2. Comments: a line whose first non-space char is `#` is skipped; a trailing ` #...` is stripped.
//   3. Scalar: unquoted / single- / double-quoted; surrounding whitespace trimmed; leading `~` or `$HOME`
//      expanded. Empty → null.
//   4. First matching `backend_root:` under `desktop:` wins.
//   5. Rejected → null (fail-safe): tab-indented key, flow-style (`desktop: {…}`), and multi-document
//      content after a document separator (`---` that appears AFTER real content; a LEADING `---`
//      document-start marker is honored as part of the first document), plus anything the scanner can't
//      unambiguously read.
function parseDesktopBackendRoot(configText) {
  if (typeof configText !== 'string' || configText.length === 0) return null

  const home = os.homedir()
  const lines = configText.split(/\r?\n/)
  let inDesktop = false
  let seenContent = false // has meaningful content appeared? (distinguishes leading doc-start --- from a separator)

  for (const rawLine of lines) {
    // Rule 5: a `---` marker. A LEADING `---` (before any content) is a valid YAML document-start marker and
    // is part of the first document → skip it. A `---` AFTER content is a multi-document separator → stop
    // (only the first document is honored).
    if (/^---\s*$/.test(rawLine)) {
      if (seenContent) break
      continue
    }

    // Skip blank lines and full-line comments (Rule 2).
    const firstNonSpace = rawLine.replace(/^[ \t]*/, '')
    if (firstNonSpace === '' || firstNonSpace.startsWith('#')) continue
    seenContent = true // a non-blank, non-comment line: subsequent `---` is a separator, not a doc-start

    // A column-0 (no leading whitespace) key. Tabs at col 0 are not valid block keys here.
    const isTopLevel = !/^[ \t]/.test(rawLine)
    if (isTopLevel) {
      // Rule 1 + 5: `desktop:` opens the block; flow-style `desktop: {…}` is rejected (not entered).
      const m = /^desktop:\s*(.*)$/.exec(rawLine)
      if (m) {
        const rest = m[1].replace(/\s+#.*$/, '').trim()
        if (rest.startsWith('{')) return null // flow-style → reject
        inDesktop = true
      } else {
        // any other top-level key closes the desktop block
        inDesktop = false
      }
      continue
    }

    // Indented line. Only meaningful while inside the desktop block.
    if (!inDesktop) continue

    // Rule 5: a tab anywhere in the indentation → reject (YAML forbids tabs; don't guess nesting).
    const indent = rawLine.slice(0, rawLine.length - firstNonSpace.length)
    if (indent.includes('\t')) return null

    const km = /^backend_root:\s*(.*)$/.exec(firstNonSpace)
    if (!km) continue // some other desktop.* key; keep scanning

    // Rule 2: strip a trailing comment (only when not inside quotes — handled by quote-first below).
    let val = km[1]
    // Rule 3: quoted scalar (quote wins over comment stripping).
    const dq = /^"([^"]*)"\s*(?:#.*)?$/.exec(val)
    const sq = /^'([^']*)'\s*(?:#.*)?$/.exec(val)
    if (dq) {
      val = dq[1]
    } else if (sq) {
      val = sq[1]
    } else {
      val = val.replace(/\s+#.*$/, '').trim()
    }

    if (val === '') return null // Rule 3: empty → null (Rule 4: first key wins, and it's empty)

    // Rule 3: ~ and $HOME expansion.
    if (val === '~' || val.startsWith('~/')) {
      val = path.join(home, val.slice(1))
    } else if (val === '$HOME' || val.startsWith('$HOME/')) {
      val = path.join(home, val.slice('$HOME'.length))
    }
    return val // Rule 4: first match wins
  }
  return null
}

module.exports = { parseDesktopBackendRoot }
