/**
 * Turn Electron accelerator syntax into the text shown in Settings.
 *
 * The accelerator remains unchanged in persisted state; only the renderer's
 * presentation replaces Electron's CommandOrControl token and adds readable
 * spacing between keys.
 */
export function formatQuickEntryShortcut(shortcut: string, commandOrControlLabel: string): string {
  return shortcut
    .split('+')
    .map(part => {
      const token = part.trim()

      return token === 'CommandOrControl' ? commandOrControlLabel : token
    })
    .join(' + ')
}

/** Convert the editable display text back to Electron accelerator syntax. */
export function parseQuickEntryShortcut(shortcut: string, commandOrControlLabel: string): string {
  return shortcut
    .split('+')
    .map(part => {
      const token = part.trim()

      return token === commandOrControlLabel ? 'CommandOrControl' : token
    })
    .filter(Boolean)
    .join('+')
}
