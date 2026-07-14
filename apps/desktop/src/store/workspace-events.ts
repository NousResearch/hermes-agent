import { atom } from 'nanostores'

// Event-driven "the working tree changed" signal — the smart replacement for
// polling. The agent only mutates files by running a tool, so the message
// stream's `tool.complete` (esp. ones carrying an inline_diff) is the precise
// trigger. Surfaces that mirror the filesystem / git state — the coding rail,
// the review pane, the file tree — subscribe to this tick and refresh, so they
// move exactly when the agent acts and stay idle otherwise.

export const $workspaceChangeTick = atom(0)

// Throttle so a burst of edits in one turn coalesces: fire on the leading edge
// for instant feedback, then at most once per window (a trailing fire catches
// the last edit of the burst).
const MIN_INTERVAL_MS = 500
let lastFired = 0
let trailing: null | ReturnType<typeof setTimeout> = null

function fire(): void {
  lastFired = Date.now()
  $workspaceChangeTick.set($workspaceChangeTick.get() + 1)
}

export function notifyWorkspaceChanged(): void {
  const since = Date.now() - lastFired

  if (since >= MIN_INTERVAL_MS) {
    if (trailing) {
      clearTimeout(trailing)
      trailing = null
    }

    fire()
  } else if (!trailing) {
    trailing = setTimeout(() => {
      trailing = null
      fire()
    }, MIN_INTERVAL_MS - since)
  }
}

// Tool names that can touch the working tree (everything else — read_file,
// search, web — never does, so it shouldn't trigger a refresh). NB: no bare
// `file` token — it matched the read-only `read_file` / `search_files` /
// `list_files`, firing a git probe on the single most common tool. Real file
// writers carry a verb (`write_file`, `apply_patch`, …) or an inline_diff.
const MUTATING_TOOL_RE =
  /terminal|shell|exec|bash|command|write|edit|patch|replace|apply|create|delete|remove|move|rename|mkdir|format/i

/** True when a finished tool may have changed files (carries a diff, or its
 *  name implies a filesystem/terminal mutation). */
export function toolMayMutateFiles(payload: { name?: unknown; tool?: unknown; inline_diff?: unknown }): boolean {
  if (typeof payload.inline_diff === 'string' && payload.inline_diff.trim()) {
    return true
  }

  const name = String(payload.name ?? payload.tool ?? '')

  return MUTATING_TOOL_RE.test(name)
}
