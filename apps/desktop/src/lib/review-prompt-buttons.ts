import { map } from 'nanostores'

import { type CommandsCatalogLike, isDesktopOwnedCommand } from '@/lib/desktop-slash-commands'

/** Each composer's OWN catalog, keyed by composer target (`main`, `tile:<id>`):
 *  the scope (profile, session) that composer's chat runs in. Panes acting on
 *  one chat (the Review pane's prompt buttons) read their target's entry, so a
 *  catalog another profile's tile fetched later never relabels or reroutes them. */
export const $composerCommandsCatalogs = map<Record<string, CommandsCatalogLike | undefined>>({})

export function rememberComposerCommandsCatalog(target: string, catalog: CommandsCatalogLike | undefined): void {
  $composerCommandsCatalogs.setKey(target, catalog)
}

export interface ReviewPromptButton {
  command: string
  label: string
}

/** `type: prompt` quick commands marked `review_button` in config.yaml, sorted by label. A name the
 *  desktop handles itself (`/new`, `/help`, any local or registry spec) is skipped: submitting it
 *  would run that action, not the saved prompt the button names. */
export function reviewPromptButtons(catalog: CommandsCatalogLike | undefined): ReviewPromptButton[] {
  return Object.entries(catalog?.commands ?? {})
    .flatMap(([command, meta]) => {
      const label = meta?.review_button?.trim()

      return label && !isDesktopOwnedCommand(command)
        ? [{ command: command.startsWith('/') ? command : `/${command}`, label }]
        : []
    })
    .sort((a, b) => a.label.localeCompare(b.label))
}
