/**
 * Strip terminal bracketed-paste leaks and repeated artifact tails from composer
 * text before it is shown in the UI or sent to the gateway.
 *
 * Mirrors hermes_cli/input_sanitize.py (CLI/TUI gateway defensive path).
 */

const BRACKETED_PASTE_BOUNDARY_START = /(^|[\s\n>:\]\)])\[200~/g
const BRACKETED_PASTE_BOUNDARY_END = /\[201~(?=$|[\s\n<[():;.,!?])/g
const BRACKETED_PASTE_DEGRADED_START = /(^|[\s\n>:\]\)])00~/g
const BRACKETED_PASTE_DEGRADED_END = /01~(?=$|[\s\n<[():;.,!?])/g

const DESKTOP_PASTE_ARTIFACT = '~[[e'

/** Strip leaked bracketed-paste wrapper markers from user-visible text. */
export function stripLeakedBracketedPasteWrappers(text: string): string {
  if (!text) {
    return text
  }

  let cleaned = text
    .replace(/\x1b\[200~/g, '')
    .replace(/\x1b\[201~/g, '')
    .replace(/\^\[\[200~/g, '')
    .replace(/\^\[\[201~/g, '')

  cleaned = cleaned.replace(BRACKETED_PASTE_BOUNDARY_START, '$1')
  cleaned = cleaned.replace(BRACKETED_PASTE_BOUNDARY_END, '')
  cleaned = cleaned.replace(BRACKETED_PASTE_DEGRADED_START, '$1')
  cleaned = cleaned.replace(BRACKETED_PASTE_DEGRADED_END, '')

  return cleaned
}

/** Drop a trailing run of the desktop ~[[e corruption signature (#62557). */
export function collapseRepeatedInputArtifacts(text: string, minRepeats = 4): string {
  if (!text) {
    return text
  }

  const marker = DESKTOP_PASTE_ARTIFACT
  let index = text.length
  let repeatCount = 0

  while (index >= marker.length && text.slice(index - marker.length, index) === marker) {
    repeatCount += 1
    index -= marker.length
  }

  if (repeatCount < minRepeats) {
    return text
  }

  let start = index
  if (start >= 2 && text.slice(start - 2, start) === '[e') {
    start -= 2
  } else if (start >= 1 && text[start - 1] === '[') {
    start -= 1
  }

  return text.slice(0, start)
}

/** Normalize composer text before submit or draft persistence. */
export function sanitizeComposerInput(text: string): string {
  if (!text) {
    return text
  }

  return collapseRepeatedInputArtifacts(stripLeakedBracketedPasteWrappers(text))
}
