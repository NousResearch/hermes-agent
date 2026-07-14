// Adapted from condensed-milk-pi's command dispatcher: split compounds first,
// strip pipe tails (`| head`, `| tail`, ...), then clean redirects/env prefixes
// before deciding which segment is meaningful. This is display-only; the full
// command remains available through Copy / detail.
const SILENT_HEADS = new Set(['cd', 'pushd', 'popd', 'export', 'set', 'unset', 'source', '.', 'true', 'false', ':'])
const PIPE_TAIL_HEADS = new Set(['head', 'tail', 'wc', 'sort', 'uniq'])

const basename = (head: string): string => head.split('/').pop() || head

// Split on command-chain separators, but NOT pipe. A pipe usually belongs to
// the segment's output plumbing (`cmd 2>&1 | tail -20`); condensed-milk strips
// that after segmenting instead of treating it as a separate producer.
function splitCompoundCommand(input: string): string[] {
  const segments: string[] = []
  let buf = ''
  let quote: '"' | "'" | null = null

  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i]!

    if (quote) {
      buf += ch

      if (ch === quote && input[i - 1] !== '\\') {
        quote = null
      }

      continue
    }

    if (ch === '"' || ch === "'") {
      quote = ch
      buf += ch

      continue
    }

    const op =
      input.startsWith('&&', i) || input.startsWith('||', i)
        ? input.slice(i, i + 2)
        : ch === ';' || ch === '\n'
          ? ch
          : ''

    if (op) {
      segments.push(buf)
      buf = ''
      i += op.length - 1

      continue
    }

    buf += ch
  }

  segments.push(buf)

  return segments.map(segment => stripPipeTail(segment.trim())).filter(Boolean)
}

function splitWords(segment: string): string[] {
  const words: string[] = []
  let buf = ''
  let quote: '"' | "'" | null = null

  for (let i = 0; i < segment.length; i += 1) {
    const ch = segment[i]!

    if (quote) {
      buf += ch

      if (ch === quote && segment[i - 1] !== '\\') {
        quote = null
      }

      continue
    }

    if (ch === '"' || ch === "'") {
      quote = ch
      buf += ch

      continue
    }

    if (/\s/.test(ch)) {
      if (buf) {
        words.push(buf)
        buf = ''
      }

      continue
    }

    buf += ch
  }

  if (buf) {
    words.push(buf)
  }

  return words
}

// The command word of a segment, skipping any `FOO=bar` env assignments.
function headWord(segment: string): string {
  const tokens = splitWords(segment)
  let index = 0

  while (index < tokens.length && /^[A-Za-z_]\w*=/.test(tokens[index]!)) {
    index += 1
  }

  return basename(tokens[index] ?? '')
}

function stripPipeTail(segment: string): string {
  const words = splitWords(segment)
  const out: string[] = []

  for (let i = 0; i < words.length; i += 1) {
    const word = words[i]!

    if (word === '|' && PIPE_TAIL_HEADS.has(basename(words[i + 1] ?? ''))) {
      break
    }

    out.push(word)
  }

  return out.join(' ').trim()
}

function cleanSegment(segment: string): string {
  const words = splitWords(segment)
  const out: string[] = []

  for (let i = 0; i < words.length; i += 1) {
    const word = words[i]!

    if (/^\d*(?:>>?|<)$/.test(word)) {
      i += 1

      continue
    }

    if (/^\d*(?:>&|<&)\d+$/.test(word) || /^\d*>&\d+$/.test(word)) {
      continue
    }

    out.push(word)
  }

  return out.join(' ').trim()
}

function isBoundaryEcho(segment: string): boolean {
  const words = splitWords(segment)

  if (basename(words[0] ?? '') !== 'echo') {
    return false
  }

  // Banner/status echoes are UI plumbing. Do not treat arbitrary `echo $VALUE`
  // as noise; it may be the command's actual output.
  const rest = words.slice(1).join(' ')

  return /-{2,}|_exit=|(?:^|\s|=)\$[?{]|PIPESTATUS/.test(rest)
}

/**
 * Reduce a verbose shell command to the "main" command, for display only.
 *
 * Agents wrap real work in plumbing — `cd <dir> && <cmd> 2>&1 | tail -N; echo
 * "x_exit=${PIPESTATUS[0]}"` — which buries the command the user actually cares
 * about. This peels that wrapper off using small head-word allowlists instead of
 * one giant regex:
 *
 *   1. split into segments on top-level `&&` `||` `;` (quote-aware)
 *   2. strip trailing pipe tails (`| head`, `| tail`, `| wc`, ...)
 *   3. clean env var prefixes / redirects
 *   4. drop setup/banner/status segments
 *
 * If one real command survives, show it. If multiple real commands survive,
 * show a short `first command + N commands` label instead of flooding the row
 * with every probe. The full command is always still available via Copy/detail.
 */
export function summarizeShellCommand(raw: string): string {
  const original = (raw ?? '').trim()

  if (!original) {
    return ''
  }

  const segments = splitCompoundCommand(original)

  if (segments.length <= 1) {
    return cleanSegment(original) || original
  }

  const core = segments.map(cleanSegment).filter(segment => {
    const head = headWord(segment)

    return segment && !SILENT_HEADS.has(head) && !isBoundaryEcho(segment)
  })

  if (core.length === 0) {
    return original
  }

  if (core.length === 1) {
    return core[0]!
  }

  return `${core[0]} + ${core.length - 1} ${core.length === 2 ? 'command' : 'commands'}`
}
