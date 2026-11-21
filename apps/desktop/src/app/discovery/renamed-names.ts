import { useMemo } from 'react'

import { useI18n } from '@/i18n'
import type { RenamedDestination } from '@/lib/old-name-aliases'

import { redesignNavigationCopy } from '../settings/redesign-navigation-copy'

/** What the redesign calls each renamed place, read where the rail and Settings read it, so the
 *  splash and ⌘K never name a place differently from the nav they point at. */
export function useRenamedNames(): Record<RenamedDestination, string> {
  const { locale, t } = useI18n()

  return useMemo(
    () => ({
      chats: t.simplifiedAgents.chats,
      agents: t.profiles.title,
      library: t.simplifiedAgents.library,
      routines: t.cron.title,
      machines: redesignNavigationCopy(locale).gateways
    }),
    [locale, t]
  )
}
