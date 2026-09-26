// Revive-buffer text handling for the GUI terminal: the ANSI escape scanner the
// boot-gap stripper and snapshot cleanup share, and the pure functions that
// decide which part of a serialized buffer is persisted for relaunch restore.

// How many scrollback lines to serialize for relaunch restore. Mirrors VS Code's
// terminal.integrated.persistentSessionScrollback default; the store caps the
// resulting string so a long line-wrapped buffer can't blow the storage budget.
export const PERSISTENT_SESSION_SCROLLBACK = 200

function readEscapeSequence(data: string, index: number) {
  if (data.charCodeAt(index) !== 0x1b || index + 1 >= data.length) {
    return null
  }

  const kind = data[index + 1]

  if (kind === '[') {
    for (let i = index + 2; i < data.length; i += 1) {
      const code = data.charCodeAt(i)

      if (code >= 0x40 && code <= 0x7e) {
        return data.slice(index, i + 1)
      }
    }
  }

  if (kind === ']') {
    for (let i = index + 2; i < data.length; i += 1) {
      if (data.charCodeAt(i) === 0x07) {
        return data.slice(index, i + 1)
      }

      if (data.charCodeAt(i) === 0x1b && data[i + 1] === '\\') {
        return data.slice(index, i + 2)
      }
    }
  }

  // Character-set and other short ESC forms are three bytes (e.g. ESC ( B).
  // Treating only ESC+( as a sequence leaves the final selector ("B") as
  // printable text, which disarms the initial prompt-gap stripper before it can
  // eat the shell's leading newline.
  if (['(', ')', '*', '+', '-', '.', '/'].includes(kind) && index + 2 < data.length) {
    return data.slice(index, index + 3)
  }

  return data.slice(index, Math.min(index + 2, data.length))
}

function stripEscapeSequences(data: string) {
  let index = 0
  let text = ''

  while (index < data.length) {
    const sequence = readEscapeSequence(data, index)

    if (sequence) {
      index += sequence.length
    } else {
      text += data[index]
      index += 1
    }
  }

  return text
}

// Keep only the ANSI escape sequences from a chunk, dropping printable text. Lets
// us apply control codes (e.g. a clear-screen) while discarding boot spacers and
// zsh's reverse-video "%" partial-line marker.
function keepEscapeSequences(data: string) {
  let index = 0
  let out = ''

  while (index < data.length) {
    if (data.charCodeAt(index) === 0x1b) {
      const sequence = readEscapeSequence(data, index)

      if (sequence) {
        out += sequence
        index += sequence.length

        continue
      }
    }

    index += 1
  }

  return out
}

function stripInitialPromptGap(data: string) {
  let index = 0
  let prefix = ''

  while (index < data.length) {
    const sequence = readEscapeSequence(data, index)

    if (sequence) {
      prefix += sequence
      index += sequence.length
    } else if (data[index] === '\r' || data[index] === '\n') {
      index += 1
    } else {
      return prefix + data.slice(index)
    }
  }

  return prefix
}

// While armed, strip leading blank rows so the first prompt lands at the very
// top (no starship `add_newline` gap). Returns the text to write, or null to
// write nothing. This only filters renderer output: never inject Ctrl-L or
// other cleanup keystrokes into the user's shell.
export function createBootGapFilter(): (data: string) => string | null {
  let stripLeading = true

  return data => {
    if (!stripLeading) {
      return data
    }

    const next = stripInitialPromptGap(data)
    const visible = stripEscapeSequences(next).replace(/[\s%]/g, '')

    if (!visible) {
      // Spacer / lone clear-screen / zsh `%` marker: apply control codes but
      // drop the blank text and stay armed so the prompt still lands at top.
      return keepEscapeSequences(next) || null
    }

    stripLeading = false

    return next
  }
}

// A row's content with ANSI escapes and all whitespace stripped — '' for a
// spacer / prompt-gap / zsh `%` marker row.
const visibleText = (line: string) => stripEscapeSequences(line).replace(/[\s%]/g, '')

const FISH_WELCOME = 'Welcometofish,thefriendlyinteractiveshell'
const FISH_HELP = 'Typehelpforinstructionsonhowtousefish'

const isFishShell = (shell: string) => shell.split(/[\\/]/).pop()?.toLowerCase() === 'fish'

// This function receives only PTY output produced after the restore boundary,
// so the leading greeting belongs to the fresh Fish process. Historical command
// output is held separately and never enters this classifier.
function stripLiveFishGreeting(lines: string[]): string[] {
  if (visibleText(lines[0] ?? '') !== FISH_WELCOME || visibleText(lines[1] ?? '') !== FISH_HELP) {
    return lines
  }

  return lines.slice(2)
}

// Trim the shell's trailing idle prompt from a serialized snapshot before it's
// persisted. Without it, the saved buffer ends in the old prompt, so the next
// launch replays it directly above the fresh shell's prompt ("double bar").
//
// An interactive shell always reprints its prompt after a command finishes, so
// the tail of an idle buffer is the prompt, never real history. Two prompt
// shapes exist:
//   - Spaced/multi-line (starship add_newline, powerline): a blank line sits
//     just above the prompt, so the short block after the last blank is dropped.
//   - Single-line (default PowerShell `PS C:\..>`, bash `user@host:~$`): no blank
//     separator, so the final line itself is the prompt and is dropped.
// The fresh shell reprints the current prompt on boot either way, so only the
// redundant idle prompt is removed — command output is preserved.
export function cleanReviveSnapshot(serialized: string, shell = '', startsAtLiveBoundary = true): string {
  const fish = isFishShell(shell)

  const lines =
    fish && startsAtLiveBoundary ? stripLiveFishGreeting(serialized.split(/\r?\n/)) : serialized.split(/\r?\n/)

  while (lines.length && !visibleText(lines[lines.length - 1])) {
    lines.pop()
  }

  if (lines.length === 0) {
    return ''
  }

  if (fish) {
    // The live boundary proves which greeting belongs to this new Fish process,
    // but plain terminal rows cannot prove whether a prompt-looking tail is a
    // prompt or command output. Preserve the tail rather than guessing away data.
    return lines.join('\r\n')
  }

  const lastBlank = lines.findLastIndex(line => !visibleText(line))
  const spacedPrompt = lastBlank >= 0 && lines.length - 1 - lastBlank <= 3

  // Spaced prompt (starship/powerline): drop the block after the blank
  // separator. Otherwise the last line is the single-line prompt itself.
  lines.length = spacedPrompt ? lastBlank : lines.length - 1

  return lines.join('\r\n')
}

// Keep restored history byte-for-byte and append only the cleaned output emitted
// by the new PTY. This provenance boundary is what makes greeting/prompt cleanup
// safe: legacy scrollback is never reclassified by its visible text.
export function mergeReviveSnapshot(
  restored: string,
  live: string,
  shell = '',
  startsAtLiveBoundary = true
): string {
  const cleanedLive = cleanReviveSnapshot(live, shell, startsAtLiveBoundary)

  if (!restored) {
    return cleanedLive
  }

  if (!cleanedLive) {
    return restored
  }

  return `${restored}\r\n${cleanedLive}`
}

export function resolveLiveSnapshotWindow(
  markerLine: number,
  end: number,
  cursorLine: number,
  maxRows = PERSISTENT_SESSION_SCROLLBACK,
  markerRegistered = true
): { keepRestored: boolean; start: number } | null {
  // xterm reset paths can leave the marker object numerically valid after it was
  // removed from `term.markers`, or restart the cursor above it. Only a currently
  // registered marker at/before the cursor can still delimit live output.
  if (!markerRegistered || markerLine < 0 || markerLine > end || markerLine > cursorLine) {
    return null
  }

  const start = Math.max(markerLine, end - maxRows + 1)

  return { keepRestored: start === markerLine, start }
}
