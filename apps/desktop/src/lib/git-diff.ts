// Pure unified-diff parser + per-hunk patch reconstruction for the Source
// Control diff viewer. Kept dependency-free and side-effect-free so it can be
// unit-tested in isolation and reused by the hunk-staging path.

export interface DiffLine {
  kind: 'add' | 'context' | 'del' | 'meta'
  // Original line text WITHOUT the leading +/-/space marker.
  text: string
  // 1-based line numbers in the old / new file (null on the side that doesn't
  // have the line, e.g. an added line has no oldLine).
  oldLine: number | null
  newLine: number | null
}

export interface DiffHunk {
  // The raw "@@ -a,b +c,d @@" header line.
  header: string
  oldStart: number
  oldLines: number
  newStart: number
  newLines: number
  lines: DiffLine[]
}

export interface ParsedDiff {
  // The file header lines (diff --git, index, ---/+++, etc.) above the hunks.
  fileHeader: string[]
  oldPath: string | null
  newPath: string | null
  binary: boolean
  hunks: DiffHunk[]
}

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/

// Parse a single-file unified diff (the output of `git diff [--cached] -- file`).
export function parseUnifiedDiff(diff: string): ParsedDiff {
  const result: ParsedDiff = {
    fileHeader: [],
    oldPath: null,
    newPath: null,
    binary: false,
    hunks: []
  }

  if (!diff) {
    return result
  }

  const lines = diff.split('\n')
  let i = 0
  let inHunks = false
  let current: DiffHunk | null = null
  let oldLine = 0
  let newLine = 0

  for (; i < lines.length; i += 1) {
    const line = lines[i]

    if (!inHunks) {
      // File header region (everything before the first @@).
      if (line.startsWith('@@')) {
        inHunks = true
      } else {
        if (line.startsWith('--- ')) {result.oldPath = stripDiffPath(line.slice(4))}
        else if (line.startsWith('+++ ')) {result.newPath = stripDiffPath(line.slice(4))}
        else if (line.startsWith('Binary files') || /^GIT binary patch/.test(line)) {result.binary = true}

        result.fileHeader.push(line)

        continue
      }
    }

    if (line.startsWith('@@')) {
      const m = HUNK_RE.exec(line)

      if (m) {
        current = {
          header: line,
          oldStart: parseInt(m[1], 10),
          oldLines: m[2] ? parseInt(m[2], 10) : 1,
          newStart: parseInt(m[3], 10),
          newLines: m[4] ? parseInt(m[4], 10) : 1,
          lines: []
        }
        oldLine = current.oldStart
        newLine = current.newStart
        result.hunks.push(current)
      }

      continue
    }

    if (!current) {
      continue
    }

    const marker = line[0]

    if (marker === '+') {
      current.lines.push({ kind: 'add', text: line.slice(1), oldLine: null, newLine })
      newLine += 1
    } else if (marker === '-') {
      current.lines.push({ kind: 'del', text: line.slice(1), oldLine, newLine: null })
      oldLine += 1
    } else if (marker === '\\') {
      // "\ No newline at end of file" — attach as meta, no line numbers.
      current.lines.push({ kind: 'meta', text: line, oldLine: null, newLine: null })
    } else {
      // Context line (leading space) or a blank trailing line.
      const text = marker === ' ' ? line.slice(1) : line
      current.lines.push({ kind: 'context', text, oldLine, newLine })
      oldLine += 1
      newLine += 1
    }
  }

  return result
}

// git's a/foo.txt → foo.txt (also strips a trailing tab + timestamp).
function stripDiffPath(raw: string): string {
  const path = raw.split('\t')[0].trim()

  if (path === '/dev/null') {return path}

  return path.replace(/^[ab]\//, '')
}

// Rebuild a minimal, apply-able patch for ONE hunk. `reverse` adjusts the header
// counts so `git apply --reverse` can unstage. The file header is required so
// git knows which file the hunk targets.
export function buildHunkPatch(parsed: ParsedDiff, hunk: DiffHunk): string {
  // Reuse the file's --- / +++ lines; if missing, synthesize from paths.
  const headerLines = parsed.fileHeader.filter(
    l => l.startsWith('diff ') || l.startsWith('index ') || l.startsWith('--- ') || l.startsWith('+++ ') || l.startsWith('new file') || l.startsWith('deleted file') || l.startsWith('rename ')
  )

  const body: string[] = []

  for (const line of hunk.lines) {
    if (line.kind === 'add') {body.push('+' + line.text)}
    else if (line.kind === 'del') {body.push('-' + line.text)}
    else if (line.kind === 'meta') {body.push(line.text)}
    else {body.push(' ' + line.text)}
  }

  return [...headerLines, hunk.header, ...body].join('\n') + '\n'
}

// Count add / delete lines in a parsed diff (for the file row +N -M summary).
export function diffStats(parsed: ParsedDiff): { additions: number; deletions: number } {
  let additions = 0
  let deletions = 0

  for (const hunk of parsed.hunks) {
    for (const line of hunk.lines) {
      if (line.kind === 'add') {additions += 1}
      else if (line.kind === 'del') {deletions += 1}
    }
  }

  return { additions, deletions }
}
