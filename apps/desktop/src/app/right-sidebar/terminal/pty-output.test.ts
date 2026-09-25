import { Terminal } from '@xterm/xterm'
import { describe, expect, it } from 'vitest'

import { PTY_TERMINAL_LINE_OPTIONS } from './pty-output'

const write = (term: Terminal, data: string) => new Promise<void>(resolve => term.write(data, resolve))
const row = (term: Terminal, y: number) => term.buffer.active.getLine(y)?.translateToString(true) ?? ''

describe('PTY terminal line discipline', () => {
  it('keeps the column on a bare LF, as a cursor-addressed redraw expects', async () => {
    const term = new Terminal({ ...PTY_TERMINAL_LINE_OPTIONS, allowProposedApi: true, cols: 40, rows: 6 })

    await write(term, '\x1b[2;3H\x1b[K\n')

    expect(term.buffer.active.cursorX).toBe(2)
    term.dispose()
  })

  it('a redraw of indented rows lands on the indent, as Claude Code emits it', async () => {
    const term = new Terminal({ ...PTY_TERMINAL_LINE_OPTIONS, allowProposedApi: true, cols: 40, rows: 6 })

    // Previous frame: two indented paragraph rows.
    await write(term, '\x1b[1;3HThe old line\x1b[2;3HWhat was here')
    // Captured repaint shape: address the indent, clear, bare LF, text, clear.
    await write(term, '\x1b[1;3H\x1b[K\nfresh\x1b[K')

    expect(row(term, 1)).toBe('  fresh')
    term.dispose()
  })
})
