import { atom } from 'nanostores'

/** Whether the global command palette (Cmd/Ctrl+K) is currently open. */
export const $commandPaletteOpen = atom(false)

export function openCommandPalette(): void {
  $commandPaletteOpen.set(true)
}

export function closeCommandPalette(): void {
  $commandPaletteOpen.set(false)
}

export function setCommandPaletteOpen(open: boolean): void {
  $commandPaletteOpen.set(open)
}

export function toggleCommandPalette(): void {
  $commandPaletteOpen.set(!$commandPaletteOpen.get())
}
