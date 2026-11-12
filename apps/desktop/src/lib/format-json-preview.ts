// Minified JSON (one giant line) is unreadable in the source view. These caps
// keep the formatted copy cheap to build and smooth to highlight.
const MAX_INPUT_CHARS = 256 * 1024
const MAX_DEPTH = 64
const MAX_LINES = 10_000

const WS = new Set([' ', '\t', '\n', '\r'])

// Index just past the closing quote of the string opening at `start`.
function stringEnd(text: string, start: number): number {
  let end = start + 1

  while (end < text.length && text[end] !== '"') {
    end += text[end] === '\\' ? 2 : 1
  }

  return end + 1
}

// Index just past a number / true / false / null token.
function scalarEnd(text: string, start: number): number {
  let end = start + 1

  while (end < text.length && !WS.has(text[end]) && !',:]}'.includes(text[end])) {
    end += 1
  }

  return end
}

// Re-indents tokens without parsing values, so numbers past 2^53 and string
// escapes are copied byte-for-byte (JSON.stringify(JSON.parse(x)) would round
// big ids). Returns null when the depth or line budget is exceeded.
function reindent(text: string): string | null {
  const out: string[] = []
  let depth = 0
  let lines = 1

  const newline = () => {
    lines += 1

    return `\n${'  '.repeat(depth)}`
  }

  for (let i = 0; i < text.length;) {
    const ch = text[i]

    if (ch === '"') {
      const end = stringEnd(text, i)

      out.push(text.slice(i, end))
      i = end

      continue
    }

    if (WS.has(ch)) {
      i += 1

      continue
    }

    if (ch === '{' || ch === '[') {
      let next = i + 1

      while (WS.has(text[next])) {
        next += 1
      }

      const close = ch === '{' ? '}' : ']'

      if (text[next] === close) {
        out.push(ch + close)
        i = next + 1

        continue
      }

      depth += 1

      if (depth > MAX_DEPTH) {
        return null
      }

      out.push(ch, newline())
    } else if (ch === '}' || ch === ']') {
      depth -= 1
      out.push(newline(), ch)
    } else if (ch === ',') {
      out.push(',', newline())
    } else if (ch === ':') {
      out.push(': ')
    } else {
      const end = scalarEnd(text, i)

      out.push(text.slice(i, end))
      i = end

      continue
    }

    if (lines > MAX_LINES) {
      return null
    }

    i += 1
  }

  return out.join('')
}

/** Indented copy of a valid single-line JSON document, or null when the text
 *  is not JSON, is already laid out over several lines, or is too big/deep to
 *  format. Only for display: line numbers no longer match the file. */
export function formatMinifiedJson(text: string): string | null {
  const trimmed = text.trim()

  if (!trimmed || trimmed.length > MAX_INPUT_CHARS || /[\n\r]/.test(trimmed)) {
    return null
  }

  if (trimmed[0] !== '{' && trimmed[0] !== '[') {
    return null
  }

  try {
    JSON.parse(trimmed)
  } catch {
    return null
  }

  const formatted = reindent(trimmed)

  return formatted && formatted !== trimmed ? formatted : null
}
