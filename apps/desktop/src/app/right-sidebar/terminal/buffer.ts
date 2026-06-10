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

// The persistent terminal is a singleton (one xterm mounted forever), so a
// module-level slot is enough — set while the session is live, cleared on
// dispose. The gateway `terminal.read.request` handler reads through this.
let activeReader: Reader | null = null

export function setActiveTerminalReader(reader: Reader | null): void {
  activeReader = reader
}

export function readActiveTerminal(opts: TerminalReadOptions = {}): TerminalReadResult | null {
  return activeReader ? activeReader(opts) : null
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
