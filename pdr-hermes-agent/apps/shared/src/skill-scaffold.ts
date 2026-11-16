/**
 * A `/skill` invocation expands into a model-facing message that embeds the
 * whole skill body. That payload is for the agent — the UI shows the
 * invocation the user typed (`/work fix the leak`) and nothing else.
 *
 * The gateway already projects this (see `_skill_scaffold_projection` in
 * tui_gateway/server.py) and ships the result as `display` on a dispatch and
 * as the `text` of a `skill_invocation` history row. This module is the
 * client-side twin so a desktop/TUI talking to an older gateway — or any
 * future path that hands raw scaffolding to a bubble — still renders the
 * invocation instead of the body.
 *
 * The markers below mirror `agent/skill_commands.py` byte for byte.
 */

const INVOCATION_PREFIX = '[IMPORTANT: The user has invoked the '
const SINGLE_MARKER = 'The full skill content is loaded below.]'
const SINGLE_INSTRUCTION = 'The user has provided the following instruction alongside the skill invocation: '
const RUNTIME_NOTE = '\n\n[Runtime note:'
const BUNDLE_MARKER = ' skill bundle,'
const BUNDLE_INSTRUCTION = '\nUser instruction: '
const BUNDLE_SKILL_BLOCK = '\n\n[Loaded as part of the '

// The skill name is the first quoted span of the activation note, for both the
// single-skill (`work`) and the bundle (`/clean /work`) header.
const NAME_RE = new RegExp(`^${INVOCATION_PREFIX.replace(/[[\]]/g, '\\$&')}"([^"]*)"`)

/** Text between `marker` and `end`, or '' when the marker is absent. */
function between(text: string, marker: string, end: string, fromEnd = false): string {
  const index = fromEnd ? text.lastIndexOf(marker) : text.indexOf(marker)

  if (index < 0) {
    return ''
  }

  const tail = text.slice(index + marker.length)
  const stop = tail.indexOf(end)

  return (stop >= 0 ? tail.slice(0, stop) : tail).trim()
}

/**
 * The invocation a scaffolded turn came from (`/work fix the leak`), or null
 * when `text` is ordinary user prose that should render as written.
 */
export function skillInvocationText(text: string): null | string {
  if (!text.startsWith(INVOCATION_PREFIX)) {
    return null
  }

  const name = (NAME_RE.exec(text)?.[1] ?? '').trim()

  if (!name) {
    return null
  }

  // Bundle headers already carry their typed "/a /b" keys; a single skill is
  // a bare name. The single-skill instruction trails the body (which may quote
  // the marker), so match it from the end.
  const label = name.startsWith('/') ? name : `/${name}`

  const instruction = text.includes(BUNDLE_MARKER)
    ? between(text, BUNDLE_INSTRUCTION, BUNDLE_SKILL_BLOCK)
    : text.includes(SINGLE_MARKER)
      ? between(text, SINGLE_INSTRUCTION, RUNTIME_NOTE, true)
      : ''

  return instruction ? `${label} ${instruction.replace(/\s+/g, ' ')}` : label
}
