import type { Terminal } from '@xterm/xterm'

// Serialized view of the in-app terminal, handed to the agent's `read_terminal`
// tool. Line indices are absolute into xterm's buffer (0 = oldest scrollback
// line), so the agent can page with start_line/count against `total_lines`.
export interface TerminalReadResult {
  total_lines: number
  start: number
  end: number
  viewport_rows: number
  cursor_row: number
  text: string
}

export interface TerminalReadOptions {
  start?: number
  count?: number
}

type Reader = (opts: TerminalReadOptions) => TerminalReadResult

// Each live terminal registers a reader keyed by its id; a single `activeId`
// (driven by the tab selection) decides which one the agent's `read_terminal`
// tool sees. Keying by id keeps switching race-free — a deactivating tab's
// cleanup can't null out the tab that just activated.
const readers = new Map<string, Reader>()
let activeId: string | null = null

/** Register a live terminal's reader; returns an idempotent unregister. */
export function registerTerminalReader(id: string, reader: Reader): () => void {
  readers.set(id, reader)

  return () => {
    if (readers.get(id) === reader) {
      readers.delete(id)
    }
  }
}

export function setActiveTerminalId(id: string | null): void {
  activeId = id
}

export function readActiveTerminal(opts: TerminalReadOptions = {}): TerminalReadResult | null {
  const reader = activeId === null ? null : readers.get(activeId)

  return reader ? reader(opts) : null
}

export function makeTerminalReader(term: Terminal): Reader {
  return ({ start, count }) => {
    const buf = term.buffer.active
    const total = buf.length
    const rows = term.rows
    // Default window = the visible screen; baseY is the viewport's top row.
    const from = Math.max(0, Math.min(start ?? buf.baseY, total))
    const to = Math.max(from, Math.min(from + Math.max(1, count ?? rows), total))

    const lines: string[] = []

    // translateToString(true) right-trims and resolves wide chars, dropping SGR
    // colors — exactly what the agent wants.
    for (let i = from; i < to; i += 1) {
      lines.push(buf.getLine(i)?.translateToString(true) ?? '')
    }

    while (lines.length && !lines[lines.length - 1].trim()) {
      lines.pop()
    }

    return {
      total_lines: total,
      start: from,
      end: to,
      viewport_rows: rows,
      cursor_row: buf.baseY + buf.cursorY,
      text: lines.join('\n')
    }
  }
}
