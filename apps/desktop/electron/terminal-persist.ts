/** Named tmux sessions so a Desktop quit only detaches the SSH client.
 *  Cursor keeps running on the remote until the user closes the tab.
 *  Reopen reattaches; if the session is gone, resume the matching Cursor chat. */

export const SYDNEY_TZ = 'Australia/Sydney'

export function shellSingleQuote(value: string): string {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

/** Force the tmux status clock onto Sydney even if the server process started in UTC.
 *  `%%` is required: tmux expands `%H`/`%M` in status formats before `#()` runs, so a
 *  single `%` makes `date` print the already-expanded UTC wall clock. */
export function buildTmuxSydneyClockCommands(): string {
  return [
    `tmux set-environment -g TZ ${SYDNEY_TZ} 2>/dev/null || true`,
    `tmux set-option -g status-interval 15 2>/dev/null || true`,
    `tmux set-option -g history-limit 100000 2>/dev/null || true`,
    `tmux set-option -g status-right '#{?window_bigger,[#{window_offset_x}#,#{window_offset_y}] ,}\"#{=21:pane_title}\" #(TZ=${SYDNEY_TZ} date "+%%H:%%M %%d-%%b-%%y %%Z")' 2>/dev/null || true`
  ].join('\n')
}

export const INK_PAGE_UP_CSI = '\x1b[5~'
export const INK_PAGE_DOWN_CSI = '\x1b[6~'

/** Detect a write that is only Ink/Desktop Page Up/Down CSI (possibly repeated). */
export function parseInkPageKeyWrite(data: string): { direction: -1 | 1; pages: number } | null {
  const text = String(data || '')

  if (!text) {
    return null
  }

  const up = text.split(INK_PAGE_UP_CSI).length - 1
  const down = text.split(INK_PAGE_DOWN_CSI).length - 1
  const onlyUp = up > 0 && down === 0 && text.split(INK_PAGE_UP_CSI).join('') === ''
  const onlyDown = down > 0 && up === 0 && text.split(INK_PAGE_DOWN_CSI).join('') === ''

  if (onlyUp) {
    return { direction: -1, pages: up }
  }

  if (onlyDown) {
    return { direction: 1, pages: down }
  }

  return null
}

/**
 * Cursor Ink `U()` only scrolls the current TUI frame. Long chats live in tmux
 * pane history (thousands of lines) while the live frame is often just the todos
 * + prompt — so Page Up into Ink is a silent no-op. Browse that history instead.
 */
export function buildTmuxHistoryScrollCommand(
  persistKey: string,
  direction: -1 | 1,
  pages = 1
): string {
  const session = shellSingleQuote(sanitizeTmuxSessionName(persistKey))
  const lines = Math.max(1, Math.min(40, Math.round(pages) * 8))

  if (direction < 0) {
    return [
      `tmux copy-mode -t ${session} 2>/dev/null || true`,
      `tmux send-keys -t ${session} -X -N ${lines} scroll-up 2>/dev/null || true`
    ].join('; ')
  }

  return [
    `if [ "$(tmux display -t ${session} -p '#{pane_in_mode}' 2>/dev/null)" = 1 ]; then`,
    `  tmux send-keys -t ${session} -X -N ${lines} scroll-down 2>/dev/null || true`,
    `  pos=$(tmux display -t ${session} -p '#{scroll_position}' 2>/dev/null || echo)`,
    `  if [ -z "$pos" ] || [ "$pos" = 0 ]; then tmux send-keys -t ${session} -X cancel 2>/dev/null || true; fi`,
    'fi'
  ].join('\n')
}

export function buildTmuxCancelCopyModeCommand(persistKey: string): string {
  const session = shellSingleQuote(sanitizeTmuxSessionName(persistKey))

  return `tmux send-keys -t ${session} -X cancel 2>/dev/null || true`
}

export function sanitizeTmuxSessionName(persistKey: string): string {
  const cleaned = String(persistKey || '')
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)

  return `h-${cleaned || 'term'}`
}

export function buildKillPersistedSessionCommand(persistKey: string): string {
  return `tmux kill-session -t ${shellSingleQuote(sanitizeTmuxSessionName(persistKey))} 2>/dev/null || true`
}

export interface PersistentRemoteCommandOptions {
  cwd?: string
  persistKey: string
  cursorChatId?: string
  resumeOnCreate?: boolean
}

export function buildPersistentRemoteCommand({
  cwd = '',
  persistKey,
  cursorChatId = '',
  resumeOnCreate = false
}: PersistentRemoteCommandOptions): string {
  const session = shellSingleQuote(sanitizeTmuxSessionName(persistKey))
  const remoteCwd = shellSingleQuote(String(cwd || '').trim())
  const chat = shellSingleQuote(String(cursorChatId || '').trim())

  return [
    'export TZ="${TZ:-Australia/Sydney}"',
    'export PATH="$HOME/bin:$HOME/.local/bin:$PATH"',
    `HERMES_TERM=${session}`,
    `HERMES_CWD=${remoteCwd}`,
    `HERMES_CHAT=${chat}`,
    resumeOnCreate ? 'HERMES_RESUME=1' : 'HERMES_RESUME=',
    'if [ -x "$HOME/bin/hermes-term" ]; then',
    '  exec "$HOME/bin/hermes-term" --session "$HERMES_TERM" --cwd "$HERMES_CWD" --chat "$HERMES_CHAT" ${HERMES_RESUME:+--resume}',
    'fi',
    'if [ -n "$HERMES_CWD" ]; then cd "$HERMES_CWD" 2>/dev/null || true; fi',
    'if command -v tmux >/dev/null 2>&1; then',
    buildTmuxSydneyClockCommands(),
    '  if tmux has-session -t "$HERMES_TERM" 2>/dev/null; then exec tmux attach-session -t "$HERMES_TERM"; fi',
    '  if [ -n "$HERMES_RESUME" ]; then',
    '    if [ -x "$HOME/bin/hermes-term-resume" ]; then',
    '      exec tmux new-session -s "$HERMES_TERM" ${HERMES_CWD:+-c "$HERMES_CWD"} -- "$HOME/bin/hermes-term-resume"',
    '    fi',
    '    if [ -n "$HERMES_CHAT" ]; then',
    '      exec tmux new-session -s "$HERMES_TERM" ${HERMES_CWD:+-c "$HERMES_CWD"} -- agent --resume "$HERMES_CHAT"',
    '    fi',
    '    exec tmux new-session -s "$HERMES_TERM" ${HERMES_CWD:+-c "$HERMES_CWD"} -- agent --continue',
    '  fi',
    '  exec tmux new-session -s "$HERMES_TERM" ${HERMES_CWD:+-c "$HERMES_CWD"}',
    'fi',
    'if [ -n "$HERMES_RESUME" ]; then',
    '  if [ -x "$HOME/bin/hermes-term-resume" ]; then exec "$HOME/bin/hermes-term-resume"; fi',
    '  if [ -n "$HERMES_CHAT" ]; then exec agent --resume "$HERMES_CHAT"; fi',
    '  exec agent --continue',
    'fi',
    'exec "$SHELL" -l'
  ].join('\n')
}
