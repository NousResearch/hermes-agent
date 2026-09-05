import type { SessionInfo } from '@/types/hermes'

/** A `SessionInfo` with every field at its zero value.
 *
 *  `SessionInfo` mirrors a backend row and grows a field whenever the API does.
 *  Spelling the whole shape out per test file meant each new field broke a
 *  handful of unrelated suites; here it breaks one line. Tests override only
 *  what they assert on. */
export function makeSessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: 'session',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

let nextId = 0

/** A real-looking CLI session rooted at `cwd`, with a fresh id. What the
 *  folder-grouping and per-project colour tests bucket; they care about the
 *  path and that the session is otherwise unremarkable. */
export function makeCwdSession(cwd: null | string, overrides: Partial<SessionInfo> = {}): SessionInfo {
  return makeSessionInfo({
    cwd,
    id: `s${nextId++}`,
    last_active: 1_000,
    message_count: 1,
    model: 'claude',
    source: 'cli',
    started_at: 1_000,
    ...overrides
  })
}
