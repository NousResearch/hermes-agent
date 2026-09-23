// ANSI escape stripping shared by every TS surface (TUI, desktop, web).
// Covers CSI (complete and truncated tails), OSC (hyperlinks, titles), DCS/SOS/PM/APC
// strings, multi-byte non-CSI ESC sequences, stray ESC bytes and C0 controls — a
// weaker SGR-only regex leaves `]8;;url\x07` payloads and `[12;` tails visible.

const ESC = String.fromCharCode(27)
const BEL = String.fromCharCode(7)
const ANSI_CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const ANSI_CSI_WITH_CMD_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*([@-~])`, 'g')
const ANSI_INCOMPLETE_CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*(?=${ESC}|\\n|$)`, 'g')
const ANSI_OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')
const ANSI_STRING_RE = new RegExp(`${ESC}[PX^_][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')
const ANSI_NON_CSI_ESC_SEQ_RE = new RegExp(`${ESC}(?!\\[|\\]|P|X|\\^|_)[ -/]*[0-~]`, 'g')
const ANSI_STRAY_ESC_RE = new RegExp(`${ESC}(?!\\[)[\\s\\S]?`, 'g')
// eslint-disable-next-line no-control-regex -- intentionally strips C0/C1 control chars
const CONTROL_RE = /[\x00-\x08\x0B\x0C\x0D\x0E-\x1A\x1C-\x1F\x7F]/g

/** Remove every escape sequence and control byte, returning plain visible text. */
export const stripAnsi = (s: string) =>
  s
    .replace(ANSI_OSC_RE, '')
    .replace(ANSI_STRING_RE, '')
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_CSI_RE, '')
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_NON_CSI_ESC_SEQ_RE, '')
    .replace(ANSI_STRAY_ESC_RE, '')
    .replace(CONTROL_RE, '')

/** Like stripAnsi but keeps SGR (`m`) sequences so a styled renderer can colorize. */
export const sanitizeAnsiForRender = (s: string) =>
  s
    .replace(ANSI_OSC_RE, '')
    .replace(ANSI_STRING_RE, '')
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_CSI_WITH_CMD_RE, (seq, cmd: string) => (cmd === 'm' ? seq : ''))
    .replace(ANSI_INCOMPLETE_CSI_RE, '')
    .replace(ANSI_NON_CSI_ESC_SEQ_RE, '')
    .replace(ANSI_STRAY_ESC_RE, '')
    .replace(CONTROL_RE, '')

export const hasAnsi = (s: string) => s.includes(ESC)
