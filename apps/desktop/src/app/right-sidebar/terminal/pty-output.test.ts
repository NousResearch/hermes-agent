import { Terminal } from '@xterm/xterm'
import { describe, expect, it } from 'vitest'

import { PTY_TERMINAL_LINE_OPTIONS } from './pty-output'

const write = (term: Terminal, data: string) => new Promise<void>(resolve => term.write(data, resolve))
const row = (term: Terminal, y: number) => term.buffer.active.getLine(y)?.translateToString(true) ?? ''

describe('PTY terminal line discipline', () => {
  it('keeps the caret at the input column through a ConPTY spinner frame (#157373)', async () => {
    const term = new Terminal({ ...PTY_TERMINAL_LINE_OPTIONS, allowProposedApi: true, cols: 40, rows: 6 })

    // Windows ConPTY re-encodes the classic CLI's spinner tick (cursor up, glyph, cursor down)
    // as: hide caret, address the spinner cell, glyph, bare LF back to the input row, show caret.
    await write(term, '\x1b[2;5H')
    await write(term, '\x1b[?25l\x1b[1;4H⠙\n\x1b[?25h')

    expect([term.buffer.active.cursorX, term.buffer.active.cursorY]).toEqual([4, 1])
    term.dispose()
  })

  it('a redraw of indented rows lands on the indent, as a full-screen TUI emits it', async () => {
    const term = new Terminal({ ...PTY_TERMINAL_LINE_OPTIONS, allowProposedApi: true, cols: 40, rows: 6 })

    // Previous frame: two indented paragraph rows.
    await write(term, '\x1b[1;3HThe old line\x1b[2;3HWhat was here')
    // Captured repaint shape: address the indent, clear, bare LF, text, clear.
    await write(term, '\x1b[1;3H\x1b[K\nfresh\x1b[K')

    expect(row(term, 1)).toBe('  fresh')
    term.dispose()
  })
})
