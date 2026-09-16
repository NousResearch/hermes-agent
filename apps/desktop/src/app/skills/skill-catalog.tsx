import { useCallback, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { installHubSkill, notifyHubActionFailed } from '@/store/hub-actions'
import { notify } from '@/store/notifications'

import { CatalogBrowser } from './catalog-browser'
import type { CatalogEntry } from './catalog-data'

interface SkillCatalogProps {
  installedNames: ReadonlySet<string>
  profile?: ProfileScope
  query?: string
  onQueryChange?: (value: string) => void
}

export function SkillCatalog({ installedNames, profile, query = '', onQueryChange }: SkillCatalogProps) {
  const { t } = useI18n()
  const h = t.skills.hub
  const [installing, setInstalling] = useState<ReadonlySet<string>>(new Set())
  const isInstalled = useCallback((entry: CatalogEntry) => installedNames.has(entry.name) || installedNames.has(entry.identifier), [installedNames])

  const install = (entry: CatalogEntry) => {
    if (!entry.installIdentifier || isInstalled(entry) || installing.has(entry.id)) {
      return
    }

    setInstalling(current => new Set(current).add(entry.id))
    notify({ kind: 'success', title: h.installStarted(entry.name), message: h.actionLog })
    void installHubSkill(entry.installIdentifier, profile)
      .catch(err => notifyHubActionFailed(err, h.actionFailed, entry.name, profile))
      .finally(() => setInstalling(current => new Set([...current].filter(id => id !== entry.id))))
  }

  return (
    <CatalogBrowser
      isInstalled={isInstalled}
      isInstalling={entry => installing.has(entry.id)}
      kind="skills"
      onInstall={install}
      onQueryChange={onQueryChange}
      query={query}
    />
  )
}
