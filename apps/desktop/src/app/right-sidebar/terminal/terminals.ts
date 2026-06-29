import { atom, computed } from 'nanostores'

import { $currentCwd } from '@/store/session'

import { setTerminalTakeover } from '../store'

/** One in-app terminal tab. `id` is the renderer-side handle (distinct from the
 *  PTY session id the main process mints); each instance owns its own shell. */
export interface TerminalEntry {
  id: string
  /** Display label. `auto` adopts the resolved shell name until the user renames. */
  title: string
  auto: boolean
  /** Working directory, snapshotted once at creation. Terminals live outside
   *  session/project state — the only thing they inherit is this initial cwd
   *  (the project root if opened in one, else the backend's default). Switching
   *  sessions never moves or recreates a terminal. */
  cwd: string
}

export const $terminals = atom<readonly TerminalEntry[]>([])
export const $activeTerminalId = atom<string | null>(null)

export const $activeTerminal = computed(
  [$terminals, $activeTerminalId],
  (list, id) => list.find(term => term.id === id) ?? null
)

const newId = () =>
  globalThis.crypto?.randomUUID?.() ?? `term-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`

/** Append a fresh terminal and focus it. Captures the current cwd once (its only
 *  tie to session/project state); pass an explicit cwd to override. Returns the id. */
export function createTerminal(cwd: string = $currentCwd.get()): string {
  const id = newId()
  $terminals.set([...$terminals.get(), { id, title: 'Terminal', auto: true, cwd }])
  $activeTerminalId.set(id)

  return id
}

/** Guarantee at least one terminal exists (called when the pane opens). */
export function ensureTerminal(): void {
  if (!$terminals.get().length) {
    createTerminal()
  }
}

export function selectTerminal(id: string): void {
  if ($terminals.get().some(term => term.id === id)) {
    $activeTerminalId.set(id)
  }
}

/** Move the active tab by `direction` (+1 next / -1 prev), wrapping around. */
export function cycleTerminal(direction: 1 | -1): void {
  const list = $terminals.get()

  if (list.length < 2) {
    return
  }

  const current = Math.max(0, list.findIndex(term => term.id === $activeTerminalId.get()))

  $activeTerminalId.set(list[(current + direction + list.length) % list.length].id)
}

/** Drop a terminal. Focus slides to the neighbor that fills its slot; closing
 *  the last one closes the whole pane. */
export function closeTerminal(id: string): void {
  const list = $terminals.get()
  const index = list.findIndex(term => term.id === id)

  if (index < 0) {
    return
  }

  const next = list.filter(term => term.id !== id)
  $terminals.set(next)

  if ($activeTerminalId.get() === id) {
    $activeTerminalId.set((next[index] ?? next[index - 1])?.id ?? null)
  }

  if (!next.length) {
    setTerminalTakeover(false)
  }
}

export function closeActiveTerminal(): void {
  const id = $activeTerminalId.get()

  if (id) {
    closeTerminal(id)
  }
}

export function closeOtherTerminals(id: string): void {
  const keep = $terminals.get().find(term => term.id === id)

  if (keep) {
    $terminals.set([keep])
    $activeTerminalId.set(keep.id)
  }
}

export function renameTerminal(id: string, title: string): void {
  const trimmed = title.trim()

  $terminals.set($terminals.get().map(term => (term.id === id ? { ...term, title: trimmed || term.title, auto: false } : term)))
}

/** A live terminal reports its resolved shell; adopt it as the label only while
 *  the user hasn't named the tab themselves. */
export function reportTerminalShell(id: string, shell: string): void {
  const name = shell.trim()

  if (!name) {
    return
  }

  $terminals.set($terminals.get().map(term => (term.id === id && term.auto ? { ...term, title: name } : term)))
}
