const REQUIRED_BACKEND_CONTRACT = /^\s*(?:export\s+)?const\s+REQUIRED_BACKEND_CONTRACT\s*=\s*(\d+)\s*;?\s*$/gm

function codeOnly(source: string): string {
  let result = ''
  let quote: string | null = null
  let lineComment = false
  let blockComment = false

  for (let index = 0; index < source.length; index += 1) {
    const current = source[index]!
    const next = source[index + 1]

    if (lineComment) {
      if (current === '\n') { lineComment = false; result += '\n' } else result += ' '
      continue
    }
    if (blockComment) {
      if (current === '*' && next === '/') { blockComment = false; result += '  '; index += 1 } else result += current === '\n' ? '\n' : ' '
      continue
    }
    if (quote) {
      if (current === '\\') { result += '  '; index += 1; continue }
      if (current === quote) quote = null
      result += current === '\n' ? '\n' : ' '
      continue
    }
    if (current === '/' && next === '/') { lineComment = true; result += '  '; index += 1; continue }
    if (current === '/' && next === '*') { blockComment = true; result += '  '; index += 1; continue }
    if (current === "'" || current === '"' || current === '`') { quote = current; result += ' '; continue }
    result += current
  }

  return quote || blockComment ? '' : result
}

export function parseRequiredBackendContract(source: string): number | null {
  const matches = [...codeOnly(source).matchAll(REQUIRED_BACKEND_CONTRACT)]
  if (matches.length !== 1) return null
  const value = Number(matches[0]?.[1])
  return Number.isSafeInteger(value) && value >= 0 ? value : null
}