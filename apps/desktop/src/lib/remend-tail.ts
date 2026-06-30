import remend from 'remend'

// Tail-bounded incomplete-markdown repair.
//
// Streamdown's built-in `parseIncompleteMarkdown` runs `remend` over the whole
// accumulated message on every streaming flush (~18% of script time on 50KB+
// messages). But repairs only ever matter in the trailing block: inline
// constructs can't cross a blank line, and Streamdown splits into blocks AFTER
// the repair, so a dangling opener in an earlier block can't reach the tail.
// We run `remend` on just that block instead.

const BACKTICK = 96 // `
const TILDE = 126 // ~
const SPACE = 32
const TAB = 9
const BACKSLASH = 92

const isSpace = (c: number) => c === SPACE || c === TAB

/**
 * Index of the last top-level block start — the char after the most recent
 * blank line that sits outside any open code fence or `$$` math block. An
 * unclosed fence/math always begins after that blank, so it stays wholly
 * inside the window without separate tracking. One cheap char pass, no regex.
 */
export function findRemendWindowStart(text: string): number {
  const n = text.length
  let inFence = false
  let fenceChar = 0
  let fenceRun = 0
  let inMath = false
  let boundary = 0
  let pending = -1 // a blank line, committed to `boundary` once content follows

  for (let lineStart = 0; lineStart <= n; ) {
    let lineEnd = text.indexOf('\n', lineStart)

    if (lineEnd === -1) {
      lineEnd = n
    }

    let i = lineStart

    while (i < lineEnd && isSpace(text.charCodeAt(i))) {
      i += 1
    }

    const first = i < lineEnd ? text.charCodeAt(i) : -1
    let marker = false

    // Fence open/close (``` or ~~~, ≤3 spaces indent).
    if ((first === BACKTICK || first === TILDE) && i - lineStart <= 3) {
      let run = i

      while (run < lineEnd && text.charCodeAt(run) === first) {
        run += 1
      }

      if (run - i >= 3) {
        marker = true

        if (!inFence) {
          inFence = true
          fenceChar = first
          fenceRun = run - i
        } else if (first === fenceChar && run - i >= fenceRun && onlyWhitespace(text, run, lineEnd)) {
          inFence = false
        }
      }
    }

    // Toggle `$$` math state on plain lines ($$ inside a fence is literal).
    if (!inFence && !marker) {
      for (let s = text.indexOf('$$', lineStart); s !== -1 && s < lineEnd - 1; s = text.indexOf('$$', s + 2)) {
        if (s === 0 || text.charCodeAt(s - 1) !== BACKSLASH) {
          inMath = !inMath
        }
      }
    }

    if (first === -1 && !inFence && !inMath) {
      pending = lineEnd + 1
    } else if (pending !== -1) {
      boundary = pending
      pending = -1
    }

    lineStart = lineEnd + 1
  }

  return boundary
}

function onlyWhitespace(text: string, from: number, to: number): boolean {
  for (let i = from; i < to; i += 1) {
    if (!isSpace(text.charCodeAt(i))) {
      return false
    }
  }

  return true
}

export function tailBoundedRemend(text: string): string {
  const start = findRemendWindowStart(text)

  return start <= 0 ? remend(text) : text.slice(0, start) + remend(text.slice(start))
}
