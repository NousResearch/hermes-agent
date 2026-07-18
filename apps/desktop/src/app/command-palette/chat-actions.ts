import { type DesktopChatActionCommand, desktopChatActionCommands } from '@/lib/desktop-slash-commands'
import { Terminal } from '@/lib/icons'

import { appViewForPath } from '../routes'

import type { PaletteGroup, PaletteItem } from './index'

interface ChatActionsGroupOptions {
  /** Group heading — `t.commandCenter.chatActions`. */
  heading: string
  /** False → rows render disabled with `hint`, since the command has no session
   *  to act on. */
  hasActiveSession: boolean
  /** Disabled-row reason — `t.commandCenter.chatActionsHint`. */
  hint: string
  /** Stages the picked command into the composer (never executes). */
  onStage: (command: string) => void
  /** Injectable for tests; defaults to the spec-derived command list. */
  commands?: DesktopChatActionCommand[]
}

/** A retained session id is not enough: the one-shot staging event needs the
 * main chat composer to be mounted on the active route. */
export function canStageChatAction(pathname: string, hasActiveSession: boolean): boolean {
  return hasActiveSession && appViewForPath(pathname) === 'chat'
}

/** Keywords so a row matches BOTH its plain-English description AND the literal
 *  slash string — typing `compress` or `/handoff` both surface it. */
function chatActionKeywords({ command, description }: DesktopChatActionCommand): string[] {
  return ['chat action', 'slash', 'command', command, command.slice(1), ...description.toLowerCase().split(/\s+/)]
}

/**
 * The ⌘K "Chat actions" group: the composer's slash actions, made searchable
 * and stage-able from the palette. Selecting a row hands the command to
 * `onStage` (the chip-insertion path) — it NEVER runs the command. Returns
 * `null` when nothing is eligible so the caller can omit the group entirely.
 */
export function buildChatActionsGroup({
  commands = desktopChatActionCommands(),
  hasActiveSession,
  heading,
  hint,
  onStage
}: ChatActionsGroupOptions): PaletteGroup | null {
  if (commands.length === 0) {
    return null
  }

  const items: PaletteItem[] = commands.map(entry => ({
    disabled: !hasActiveSession,
    hint: hasActiveSession ? undefined : hint,
    icon: Terminal,
    id: `chat-action-${entry.command}`,
    keywords: chatActionKeywords(entry),
    label: entry.description,
    run: () => onStage(entry.command)
  }))

  return { heading, items }
}
