import type { Terminal } from '@xterm/xterm'

// Line discipline for the embedded xterm instances.
//
// A PTY already emits CR LF where the program meant a new line, so a terminal
// fed by one must honour a bare LF as VT does: move down, KEEP the column.
// `convertEol` turns every LF into CR LF, which breaks cursor-addressed redraws
// — Claude Code positions at column 3 (`CSI row;3H`), clears, then sends a bare
// LF to start the next indented row; with the conversion that row lands at
// column 1 and the old frame's first two glyphs survive as ghosts (`Th`, `Wh`)
// after a scroll repaint.
//
// Pipe output is the opposite case: the program writes a bare LF meaning "new
// line", and without the conversion each line would start where the previous
// one ended. The user shell is always a PTY; an agent background process is one
// only when it was spawned with `pty=true`, which the backend flags per process.
export const terminalLineOptions = (pty: boolean) => ({ convertEol: !pty })

/** Writer for an agent terminal whose PTY flag may arrive after the terminal was
 *  created (the tab can open from the command header before the first chunk).
 *  Only a CHANGE of the flag touches the option, so a program that sets the LNM
 *  mode itself (`CSI 20 h` / `CSI 20 l`) keeps it. */
export function lineDisciplineWriter(term: Pick<Terminal, 'options' | 'write'>, initialPty: boolean) {
  let pty = initialPty

  return (chunk: string, nextPty: boolean) => {
    if (nextPty !== pty) {
      pty = nextPty
      term.options.convertEol = terminalLineOptions(pty).convertEol
    }

    term.write(chunk)
  }
}
