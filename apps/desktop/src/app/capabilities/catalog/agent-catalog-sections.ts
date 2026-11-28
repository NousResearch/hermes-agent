import type { AgentCapabilityCopy } from '../ui/agent-capability-copy'

import { type CatalogEntry, type CatalogKind, catalogLabel } from './catalog-data'

/** Built-in skills the card lists before "Show all" (895:14967 "Showing 7 of 58"). */
export const BUILT_IN_PREVIEW = 7

/** Library entries "Not added to this agent" lists before paging (895:14967 draws a handful). */
export const AVAILABLE_PREVIEW = 8

export interface AgentCatalogSection {
  id: 'built-in' | 'added' | 'on-agent' | 'in-app' | 'available'
  title: string
  /** The note under the title, given how many rows the card shows of how many. */
  describe: (shown: number, total: number) => string
  entries: CatalogEntry[]
  /** Rows the card shows before "Show all"; the library section pages instead. */
  preview?: number
}

interface SectionInput {
  kind: CatalogKind
  entries: CatalogEntry[]
  isInstalled: (entry: CatalogEntry) => boolean
  copy: AgentCapabilityCopy
}

const builtIn = (entry: CatalogEntry) => entry.source === 'built-in'
// Installed packages carry their half: desktop-only ones are app-wide (plugins-tab.tsx installedEntries).
const appWide = (entry: CatalogEntry) => entry.category === 'desktop'

/** One agent's catalog split the way Figma reads it: Skills are Built in / Added from the library /
 *  Not added; Plugins are On this agent / Runs in the app / Not added. Empty sections drop out. */
export function agentCatalogSections({ kind, entries, isInstalled, copy }: SectionInput): AgentCatalogSection[] {
  const installed = entries.filter(isInstalled)
  const available = entries.filter(entry => !isInstalled(entry))

  const notAdded: AgentCatalogSection = {
    id: 'available',
    title: copy.notAdded,
    describe: () => copy.notAddedDesc,
    entries: available
  }

  const sections: AgentCatalogSection[] =
    kind === 'skills'
      ? [
          {
            id: 'built-in',
            title: copy.skills.builtIn,
            describe: copy.skills.builtInDesc,
            entries: installed.filter(builtIn),
            preview: BUILT_IN_PREVIEW
          },
          {
            id: 'added',
            title: copy.skills.added,
            describe: () => copy.skills.addedDesc,
            entries: installed.filter(entry => !builtIn(entry))
          },
          notAdded
        ]
      : [
          {
            id: 'on-agent',
            title: copy.plugins.onAgent,
            describe: () => copy.plugins.onAgentDesc,
            entries: installed.filter(entry => !appWide(entry))
          },
          {
            id: 'in-app',
            title: copy.plugins.inApp,
            describe: () => copy.plugins.inAppDesc,
            entries: installed.filter(appWide)
          },
          notAdded
        ]

  return sections.filter(section => section.entries.length > 0)
}

/** A row's tags: a skill's category, a plugin's provenance (895:14967, 895:16810). */
export function agentCatalogTags(kind: CatalogKind, entry: CatalogEntry): string[] {
  const tag = kind === 'skills' ? entry.categoryLabel : entry.source && catalogLabel(entry.source)

  return tag ? [tag] : []
}
