import type { IDecoration, IMarker, ITheme, Terminal } from '@xterm/xterm'

export type SyntaxTokenKind = 'assignment' | 'command' | 'comment' | 'keyword' | 'operator' | 'option' | 'path' | 'string' | 'variable'

export interface SyntaxToken {
  end: number
  kind: SyntaxTokenKind
  start: number
  text: string
}

export interface SyntaxDecorationSegment {
  kind: SyntaxTokenKind
  rowOffset: number
  width: number
  x: number
}

const COMMAND_PREFIX_KEYWORDS = new Set(['arch', 'builtin', 'command', 'env', 'exec', 'noglob', 'nohup', 'sudo', 'time'])

const SHELL_KEYWORDS = new Set([
  'case',
  'do',
  'done',
  'elif',
  'else',
  'esac',
  'fi',
  'for',
  'function',
  'if',
  'in',
  'select',
  'then',
  'until',
  'while'
])

const OPERATORS = ['&&', '||', ';;', '|&', '>>', '<<', '2>', '1>', ';', '|', '&', '(', ')', '<', '>'] as const

const CONTROL_SEQUENCES = {
  arrowDown: '\x1b[B',
  arrowLeft: '\x1b[D',
  arrowRight: '\x1b[C',
  arrowUp: '\x1b[A',
  delete: '\x1b[3~',
  end: '\x1b[F',
  endTilde: '\x1b[4~',
  home: '\x1b[H',
  homeTilde: '\x1b[1~'
} as const

function isWhitespace(ch: string | undefined) {
  return ch === ' ' || ch === '\t' || ch === '\r' || ch === '\n'
}

function operatorAt(input: string, index: number): string | null {
  return OPERATORS.find(op => input.startsWith(op, index)) ?? null
}

function isCommentStart(input: string, index: number) {
  return input[index] === '#' && (index === 0 || isWhitespace(input[index - 1]))
}

function readQuoted(input: string, index: number) {
  const quote = input[index]
  let i = index + 1

  while (i < input.length) {
    if (input[i] === '\\') {
      i += 2

      continue
    }

    if (input[i] === quote) {
      return i + 1
    }

    i += 1
  }

  return input.length
}

function readWord(input: string, index: number) {
  let i = index

  while (i < input.length) {
    if (isWhitespace(input[i]) || isCommentStart(input, i) || operatorAt(input, i)) {
      break
    }

    if (input[i] === '"' || input[i] === "'") {
      i = readQuoted(input, i)
    } else {
      i += 1
    }
  }

  return i
}

function pushToken(tokens: SyntaxToken[], kind: SyntaxTokenKind, input: string, start: number, end: number) {
  if (end <= start) {
    return
  }

  tokens.push({ end, kind, start, text: input.slice(start, end) })
}

function pushVariableTokens(tokens: SyntaxToken[], input: string, word: string, wordStart: number) {
  const variablePattern = /\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}\n]*\})/g
  let match: RegExpExecArray | null = null

  while ((match = variablePattern.exec(word))) {
    pushToken(tokens, 'variable', input, wordStart + match.index, wordStart + match.index + match[0].length)
  }
}

function pushEmbeddedStringTokens(tokens: SyntaxToken[], input: string, word: string, wordStart: number) {
  let i = 0

  while (i < word.length) {
    if (word[i] === '"' || word[i] === "'") {
      const end = readQuoted(word, i)
      pushToken(tokens, 'string', input, wordStart + i, wordStart + end)
      i = end
    } else {
      i += 1
    }
  }
}

function isPathWord(word: string) {
  return word.startsWith('./') || word.startsWith('../') || word.startsWith('~/') || word.startsWith('/') || word.includes('/')
}

function isAssignmentWord(word: string) {
  return /^[A-Za-z_][A-Za-z0-9_]*=/.test(word)
}

function optionEnd(word: string) {
  const equals = word.indexOf('=')

  return equals === -1 ? word.length : equals
}

export function tokenizeShellCommand(input: string): SyntaxToken[] {
  const tokens: SyntaxToken[] = []
  let expectingCommand = true
  let index = 0

  while (index < input.length) {
    if (isWhitespace(input[index])) {
      index += 1

      continue
    }

    if (isCommentStart(input, index)) {
      pushToken(tokens, 'comment', input, index, input.length)

      break
    }

    const op = operatorAt(input, index)

    if (op) {
      pushToken(tokens, 'operator', input, index, index + op.length)
      index += op.length
      expectingCommand = true

      continue
    }

    if (input[index] === '"' || input[index] === "'") {
      const end = readQuoted(input, index)
      pushToken(tokens, 'string', input, index, end)
      pushVariableTokens(tokens, input, input.slice(index, end), index)
      index = end
      expectingCommand = false

      continue
    }

    const start = index
    const end = readWord(input, start)
    const word = input.slice(start, end)

    if (!word) {
      index += 1

      continue
    }

    if (expectingCommand && isAssignmentWord(word)) {
      pushToken(tokens, 'assignment', input, start, end)
      pushVariableTokens(tokens, input, word, start)
      index = end

      continue
    }

    if (expectingCommand) {
      if (COMMAND_PREFIX_KEYWORDS.has(word) || SHELL_KEYWORDS.has(word)) {
        pushToken(tokens, 'keyword', input, start, end)
        expectingCommand = COMMAND_PREFIX_KEYWORDS.has(word)
      } else {
        pushToken(tokens, 'command', input, start, end)
        expectingCommand = false
      }
    } else if (word.startsWith('-') && word !== '-') {
      pushToken(tokens, 'option', input, start, start + optionEnd(word))
    } else if (isPathWord(word)) {
      pushToken(tokens, 'path', input, start, end)
    }

    pushEmbeddedStringTokens(tokens, input, word, start)
    pushVariableTokens(tokens, input, word, start)
    index = end
  }

  return tokens.sort((a, b) => a.start - b.start || b.end - a.end)
}

function insertAt(value: string, index: number, text: string) {
  return `${value.slice(0, index)}${text}${value.slice(index)}`
}

function removeBefore(value: string, index: number) {
  return `${value.slice(0, Math.max(0, index - 1))}${value.slice(index)}`
}

function removeAt(value: string, index: number) {
  return `${value.slice(0, index)}${value.slice(index + 1)}`
}

function previousWordBoundary(value: string, index: number) {
  let cursor = index

  while (cursor > 0 && /\s/.test(value[cursor - 1] ?? '')) {
    cursor -= 1
  }

  while (cursor > 0 && !/\s/.test(value[cursor - 1] ?? '')) {
    cursor -= 1
  }

  return cursor
}

function escapeStringSequenceLength(data: string, index: number) {
  const bell = data.indexOf('\x07', index + 2)
  const stringTerminator = data.indexOf('\x1b\\', index + 2)
  const terminator = [bell, stringTerminator].filter(candidate => candidate !== -1).sort((a, b) => a - b)[0]

  if (terminator === undefined) {
    return data.length - index
  }

  return terminator - index + (terminator === stringTerminator ? 2 : 1)
}

function escapeSequenceLength(data: string, index: number) {
  if (data[index] !== '\x1b') {
    return 0
  }

  const next = data[index + 1]

  if (!next) {
    return 1
  }

  if (next === '[') {
    let cursor = index + 2

    while (cursor < data.length) {
      const code = data.charCodeAt(cursor)

      if (code >= 0x40 && code <= 0x7e) {
        return cursor - index + 1
      }

      cursor += 1
    }

    return data.length - index
  }

  if (next === ']') {
    return escapeStringSequenceLength(data, index)
  }

  if (next === 'O') {
    return Math.min(3, data.length - index)
  }

  if (next === 'P' || next === 'X' || next === '^' || next === '_') {
    return escapeStringSequenceLength(data, index)
  }

  return Math.min(2, data.length - index)
}

export class TerminalCommandDraft {
  cursor = 0
  text = ''

  applyUserInput(data: string) {
    let changed = false
    let index = 0

    while (index < data.length) {
      const rest = data.slice(index)

      if (rest.startsWith(CONTROL_SEQUENCES.arrowLeft)) {
        this.cursor = Math.max(0, this.cursor - 1)
        index += CONTROL_SEQUENCES.arrowLeft.length
        changed = true

        continue
      }

      if (rest.startsWith(CONTROL_SEQUENCES.arrowRight)) {
        this.cursor = Math.min(this.text.length, this.cursor + 1)
        index += CONTROL_SEQUENCES.arrowRight.length
        changed = true

        continue
      }

      if (rest.startsWith(CONTROL_SEQUENCES.home) || rest.startsWith(CONTROL_SEQUENCES.homeTilde)) {
        this.cursor = 0
        index += rest.startsWith(CONTROL_SEQUENCES.home) ? CONTROL_SEQUENCES.home.length : CONTROL_SEQUENCES.homeTilde.length
        changed = true

        continue
      }

      if (rest.startsWith(CONTROL_SEQUENCES.end) || rest.startsWith(CONTROL_SEQUENCES.endTilde)) {
        this.cursor = this.text.length
        index += rest.startsWith(CONTROL_SEQUENCES.end) ? CONTROL_SEQUENCES.end.length : CONTROL_SEQUENCES.endTilde.length
        changed = true

        continue
      }

      if (rest.startsWith(CONTROL_SEQUENCES.delete)) {
        if (this.cursor < this.text.length) {
          this.text = removeAt(this.text, this.cursor)
        }

        index += CONTROL_SEQUENCES.delete.length
        changed = true

        continue
      }

      if (rest.startsWith(CONTROL_SEQUENCES.arrowUp) || rest.startsWith(CONTROL_SEQUENCES.arrowDown)) {
        this.clear()
        index += rest.startsWith(CONTROL_SEQUENCES.arrowUp) ? CONTROL_SEQUENCES.arrowUp.length : CONTROL_SEQUENCES.arrowDown.length
        changed = true

        continue
      }

      const escapeLength = escapeSequenceLength(data, index)

      if (escapeLength > 0) {
        index += escapeLength
        changed = true

        continue
      }

      const ch = data[index]

      if (ch === '\r' || ch === '\n' || ch === '\x03') {
        this.clear()
        index += 1
        changed = true

        continue
      }

      if (ch === '\x01') {
        this.cursor = 0
        index += 1
        changed = true

        continue
      }

      if (ch === '\x05') {
        this.cursor = this.text.length
        index += 1
        changed = true

        continue
      }

      if (ch === '\x0b') {
        this.text = this.text.slice(0, this.cursor)
        index += 1
        changed = true

        continue
      }

      if (ch === '\x15') {
        this.text = this.text.slice(this.cursor)
        this.cursor = 0
        index += 1
        changed = true

        continue
      }

      if (ch === '\x17') {
        const start = previousWordBoundary(this.text, this.cursor)
        this.text = `${this.text.slice(0, start)}${this.text.slice(this.cursor)}`
        this.cursor = start
        index += 1
        changed = true

        continue
      }

      if (ch === '\b' || ch === '\x7f') {
        if (this.cursor > 0) {
          this.text = removeBefore(this.text, this.cursor)
          this.cursor -= 1
        }

        index += 1
        changed = true

        continue
      }

      if (ch === '\t') {
        this.clear()
        index += 1
        changed = true

        continue
      }

      if (ch >= ' ' && ch !== '\x7f') {
        this.text = insertAt(this.text, this.cursor, ch)
        this.cursor += 1
        index += 1
        changed = true

        continue
      }

      index += 1
    }

    return changed
  }

  clear() {
    this.text = ''
    this.cursor = 0
  }
}

export function layoutSyntaxDecorationSegments({
  cols,
  cursor,
  cursorX,
  cursorY,
  tokens
}: {
  cols: number
  cursor: number
  cursorX: number
  cursorY: number
  tokens: SyntaxToken[]
}): SyntaxDecorationSegment[] {
  if (cols <= 0) {
    return []
  }

  const commandStartCell = cursorY * cols + cursorX - cursor
  const segments: SyntaxDecorationSegment[] = []

  for (const token of tokens) {
    let start = commandStartCell + token.start
    const end = commandStartCell + token.end

    while (start < end) {
      const row = Math.floor(start / cols)
      const x = ((start % cols) + cols) % cols
      const rowEnd = (row + 1) * cols
      const width = Math.max(0, Math.min(end, rowEnd) - start)

      if (width > 0) {
        segments.push({ kind: token.kind, rowOffset: row - cursorY, width, x })
      }

      start = rowEnd
    }
  }

  return segments
}

export function readVisibleDraftText({
  cols,
  cursor,
  cursorX,
  cursorY,
  lineAt,
  textLength
}: {
  cols: number
  cursor: number
  cursorX: number
  cursorY: number
  lineAt: (row: number) => string | null
  textLength: number
}): string | null {
  if (cols <= 0 || textLength < 0) {
    return null
  }

  const startCell = cursorY * cols + cursorX - cursor

  if (startCell < 0) {
    return null
  }

  const endCell = startCell + textLength
  let out = ''
  let current = startCell

  while (current < endCell) {
    const row = Math.floor(current / cols)
    const startCol = ((current % cols) + cols) % cols
    const endCol = Math.min(cols, endCell - row * cols)
    const line = lineAt(row)

    if (line === null) {
      return null
    }

    out += line.padEnd(cols, ' ').slice(startCol, endCol)
    current = row * cols + endCol
  }

  return out
}

export function isDraftTextVisible({
  cols,
  cursor,
  cursorX,
  cursorY,
  lineAt,
  text
}: {
  cols: number
  cursor: number
  cursorX: number
  cursorY: number
  lineAt: (row: number) => string | null
  text: string
}) {
  return readVisibleDraftText({ cols, cursor, cursorX, cursorY, lineAt, textLength: text.length }) === text
}

const FALLBACK_COLORS: Record<SyntaxTokenKind, string> = {
  assignment: '#4ec9b0',
  command: '#23d18b',
  comment: '#6a9955',
  keyword: '#c586c0',
  operator: '#569cd6',
  option: '#dcdcaa',
  path: '#9cdcfe',
  string: '#ce9178',
  variable: '#4ec9b0'
}

const THEME_COLOR_SLOT: Partial<Record<SyntaxTokenKind, keyof ITheme>> = {
  assignment: 'cyan',
  command: 'green',
  comment: 'brightBlack',
  keyword: 'magenta',
  operator: 'blue',
  option: 'yellow',
  path: 'brightCyan',
  string: 'magenta',
  variable: 'cyan'
}

function decorationColor(kind: SyntaxTokenKind, theme: ITheme) {
  const slot = THEME_COLOR_SLOT[kind]
  const color = slot ? theme[slot] : undefined

  return typeof color === 'string' && /^#[0-9a-f]{6}$/i.test(color) ? color : FALLBACK_COLORS[kind]
}

export class TerminalSyntaxHighlighter {
  private readonly draft = new TerminalCommandDraft()
  private readonly getTheme: () => ITheme
  private readonly term: Terminal
  private decorations: IDecoration[] = []
  private frame = 0
  private markers: IMarker[] = []

  constructor(term: Terminal, getTheme: () => ITheme) {
    this.term = term
    this.getTheme = getTheme
  }

  handleUserInput(data: string) {
    if (this.term.buffer.active.type === 'alternate') {
      this.draft.clear()
      this.clear()

      return
    }

    if (!this.draft.applyUserInput(data)) {
      return
    }

    if (!this.draft.text.trim()) {
      this.clear()

      return
    }

    // Do not render from raw keyboard input. The PTY may have echo disabled
    // (password prompts), so decorations only refresh after visible terminal
    // output confirms the draft is actually present in the xterm buffer.
    this.clear()
  }

  refresh() {
    if (!this.draft.text.trim()) {
      this.clear()

      return
    }

    this.schedule()
  }

  dispose() {
    if (this.frame) {
      window.cancelAnimationFrame(this.frame)
      this.frame = 0
    }

    this.clear()
    this.draft.clear()
  }

  private clear() {
    for (const decoration of this.decorations) {
      decoration.dispose()
    }

    for (const marker of this.markers) {
      marker.dispose()
    }

    this.decorations = []
    this.markers = []
  }

  private schedule() {
    if (this.frame) {
      window.cancelAnimationFrame(this.frame)
    }

    this.frame = window.requestAnimationFrame(() => {
      this.frame = 0
      this.render()
    })
  }

  private render() {
    this.clear()

    const buffer = this.term.buffer.active

    if (buffer.type === 'alternate' || !this.draft.text.trim()) {
      return
    }

    if (
      !isDraftTextVisible({
        cols: this.term.cols,
        cursor: this.draft.cursor,
        cursorX: buffer.cursorX,
        cursorY: buffer.cursorY,
        lineAt: row => buffer.getLine(buffer.baseY + row)?.translateToString(false) ?? null,
        text: this.draft.text
      })
    ) {
      return
    }

    const tokens = tokenizeShellCommand(this.draft.text)

    const segments = layoutSyntaxDecorationSegments({
      cols: this.term.cols,
      cursor: this.draft.cursor,
      cursorX: buffer.cursorX,
      cursorY: buffer.cursorY,
      tokens
    })

    const theme = this.getTheme()

    for (const segment of segments) {
      const marker = this.term.registerMarker(segment.rowOffset)

      if (!marker) {
        continue
      }

      const decoration = this.term.registerDecoration({
        foregroundColor: decorationColor(segment.kind, theme),
        layer: 'top',
        marker,
        width: segment.width,
        x: segment.x
      })

      if (decoration) {
        this.markers.push(marker)
        this.decorations.push(decoration)
      } else {
        marker.dispose()
      }
    }
  }
}
