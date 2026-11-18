import { useEffect, useMemo, useRef } from 'react'

import { codiconIcon } from '@/components/ui/codicon'
import type { IconComponent } from '@/lib/icons'
import { matchOldName, type RenamedDestination } from '@/lib/old-name-aliases'
import { recordIaAction } from '@/store/ia-metrics'

import { useDiscoveryCopy } from '../discovery/discovery-copy'
import { useRenamedNames } from '../discovery/renamed-names'
import { ARTIFACTS_ROUTE, CRON_ROUTE, NEW_CHAT_ROUTE, PROFILES_ROUTE, SETTINGS_ROUTE } from '../routes'

/** Each renamed place's palette row: the rail's glyph and where the rail (or Settings) opens it. */
const DESTINATIONS: Record<RenamedDestination, { icon: IconComponent; route: string }> = {
  chats: { icon: codiconIcon('comment-discussion'), route: NEW_CHAT_ROUTE },
  agents: { icon: codiconIcon('robot'), route: PROFILES_ROUTE },
  library: { icon: codiconIcon('files'), route: ARTIFACTS_ROUTE },
  routines: { icon: codiconIcon('clock'), route: CRON_ROUTE },
  machines: { icon: codiconIcon('server'), route: `${SETTINGS_ROUTE}?tab=gateway` }
}

interface OldNameRow {
  icon: IconComponent
  id: string
  keywords: string[]
  label: string
  run: () => void
}

/**
 * The redesign's ⌘K answer to a classic word (Figma 63:3642): typing "sessions" offers "Chats,
 * formerly Sessions" first. The words come from the alias table (lib/old-name-aliases.ts) plus the
 * classic word of the current language. A query that is a whole classic word counts once per open
 * of the palette, by the word's code (never the query).
 */
export function useOldNameRows(
  search: string,
  enabled: boolean,
  go: (path: string) => () => void
): { items: OldNameRow[] }[] {
  const copy = useDiscoveryCopy()
  const names = useRenamedNames()
  const match = enabled ? matchOldName(search, copy.classicNames) : null
  const code = match?.alias.code
  const destination = match?.alias.destination
  const words = match?.alias.words
  const recorded = useRef(new Set<string>())

  useEffect(() => {
    if (code && match?.exact && !recorded.current.has(code)) {
      recorded.current.add(code)
      recordIaAction(`ia.alias_search.${code}`, 'palette')
    }
  }, [code, match?.exact])

  return useMemo(() => {
    if (!code || !destination || !words) {
      return []
    }

    const classic = copy.classicNames[code]

    return [
      {
        items: [
          {
            icon: DESTINATIONS[destination].icon,
            id: `renamed-${destination}`,
            keywords: [...words, classic],
            label: copy.formerly(names[destination], classic),
            run: go(DESTINATIONS[destination].route)
          }
        ]
      }
    ]
  }, [code, copy, destination, go, names, words])
}
