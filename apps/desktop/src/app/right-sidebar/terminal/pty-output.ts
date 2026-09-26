// Line discipline for xterm instances fed by a real PTY.
//
// A PTY already emits CR LF where the program meant a new line, so the terminal
// must honour a bare LF as VT does: move down, KEEP the column. `convertEol`
// turns every LF into CR LF, which breaks cursor-addressed redraws — Claude Code
// positions at column 3 (`CSI row;3H`), clears, then sends a bare LF to start
// the next indented row; with the conversion that row lands at column 1 and
// the old frame's first two glyphs survive as ghosts (`Th`, `Wh`) after a
// scroll repaint. Only non-PTY streams (agent process logs) may convert.
export const PTY_TERMINAL_LINE_OPTIONS = { convertEol: false } as const
