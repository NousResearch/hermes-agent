/**
 * Slash-command recognition for text the composer did not watch being typed —
 * a paste, a restored draft, an undo step, a rebuilt line.
 *
 * The typed path chips a command as it's picked or accepted, so the composer
 * agrees with what the sent message renders (`SLASH_SKILL_RE` in
 * directive-text). Text that arrives whole never passed through that path, so
 * it needs the same commands recognized in place — on exactly the terms the
 * typed path would have used, or hydration invents pills the popover would
 * never have committed.
 */
import type { SlashChipKind } from '@/components/assistant-ui/directive-text'
import {
  desktopSlashCommandArgumentMode,
  isDesktopSlashCommand,
  resolveDesktopCommand
} from '@/lib/desktop-slash-commands'

// A command token starts a word and doesn't continue into a path: `/usr/local`
// is a path, not a `/usr` command. Same shape the sent message uses to decide
// what renders as a pill, so the composer and the transcript agree.
const SLASH_COMMAND_RE = /(?<=^|\s)\/([a-zA-Z][\w-]*)(?![\w-]*\/)/g

export interface SlashCommandMatch {
  /** The command with its leading slash, e.g. `/clean`. */
  command: string
  end: number
  kind: SlashChipKind
  start: number
}

export interface SlashCommandScanOptions {
  /**
   * Whether the text is preceded by a token boundary. False when it's being
   * inserted mid-word (a paste landing against existing characters), which
   * disqualifies a token at index 0 — `foo/clean` is not a command. It also
   * makes that token mid-message rather than an invocation.
   */
  boundaryBefore?: boolean
  /**
   * Whether a token ending the text counts as committed. True for inert text
   * (a paste, dropped content): nothing is being typed, so `/clean` at the end
   * is the whole command. False while editing live, where a trailing `/wor` is
   * a half-typed query the popover owns and must leave editable.
   */
  trailingCommitted?: boolean
}

/**
 * Only commands with NO argument stage chip: their committed pill is exactly
 * the bare `/name`, so the boundary is unambiguous. Arg-taking commands
 * (`/goal ship it`) stay text — their tail may be prose. Commands with no
 * desktop surface at all (`/exit`, `/config`) stay text too.
 */
function chippableKind(command: string): SlashChipKind | null {
  if (!isDesktopSlashCommand(command) || desktopSlashCommandArgumentMode(command) !== null) {
    return null
  }

  return resolveDesktopCommand(command) ? 'command' : 'skill'
}

/** Every `/command` in `text` that should render as a pill, in source order. */
export function slashCommandMatches(text: string, options: SlashCommandScanOptions = {}): SlashCommandMatch[] {
  const { boundaryBefore = true, trailingCommitted = false } = options

  if (!text.includes('/')) {
    return []
  }

  const matches: SlashCommandMatch[] = []

  for (const match of text.matchAll(SLASH_COMMAND_RE)) {
    const start = match.index ?? 0
    const command = match[0]
    const end = start + command.length
    const after = text[end]

    // A committed pill always carries its auto-inserted trailing space, which
    // is what separates it from a token still being typed.
    if (after === undefined ? !trailingCommitted : !/\s/.test(after)) {
      continue
    }

    // Only the FIRST token can be an invocation, and only when the text lands
    // on a token boundary — `foo` + a pasted `/clean` is `foo/clean`.
    const invocation = start === 0

    if (invocation && !boundaryBefore) {
      continue
    }

    const kind = chippableKind(command)

    // Later tokens are references dropped into prose, where the popover offers
    // SKILLS alone — a built-in like `/new` acts on the app and means nothing
    // mid-sentence. Hydration has to agree, or pasted text grows pills typing
    // never would.
    if (kind && (invocation || kind === 'skill')) {
      matches.push({ command, end, kind, start })
    }
  }

  return matches
}
