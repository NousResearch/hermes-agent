import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { getOfficialSkills, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { HUB_SOURCES_KEY, installHubSkill, notifyHubActionFailed, OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'
import { notify } from '@/store/notifications'
import type { SkillHubSourcesResponse, SkillInfo } from '@/types/hermes'

import { CatalogAlert } from './catalog-alert'
import { CatalogBrowser } from './catalog-browser'
import { type CatalogEntry, parseCatalog } from './catalog-data'

interface SkillCatalogProps {
  skills: SkillInfo[]
  profile: ProfileScope
  query?: string
  onQueryChange?: (value: string) => void
  actions?: ReactNode
  installedPending?: boolean
  notice?: ReactNode
  renderInstalledDetail: (skill: SkillInfo) => ReactNode
  renderInstalledAction?: (skill: SkillInfo) => ReactNode
}

/** Public discovery and the profile's local skills share one browser. Management
 * stays in SkillsTab; catalog install actions have exactly one owner here. */
export function SkillCatalog(props: SkillCatalogProps) {
  return <ScopedSkillCatalog key={profileScopeKey(props.profile)} {...props} />
}

function ScopedSkillCatalog({
  skills,
  profile,
  query,
  onQueryChange,
  actions,
  installedPending,
  notice,
  renderInstalledDetail,
  renderInstalledAction
}: SkillCatalogProps) {
  const { t } = useI18n()
  const h = t.skills.hub
  const mounted = useRef(true)
  const pending = useRef(new Set<string>())
  const [installing, setInstalling] = useState<ReadonlySet<string>>(new Set())

  // eslint-disable-next-line no-restricted-syntax -- lifecycle guard drops stale async completions; it does not mirror an atom
  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  const {
    data: officialData,
    error: officialError,
    refetch: refreshOfficial
  } = useQuery({
    queryKey: [...OFFICIAL_SKILLS_KEY, profileScopeKey(profile)],
    queryFn: () => getOfficialSkills(profile),
    staleTime: 60_000,
    retry: false
  })

  const hasHubSkills = skills.some(skill => skill.provenance === 'hub')

  const {
    data: hubData,
    isPending: hubPending,
    error: hubError,
    refetch: refreshHub
  } = useQuery({
    queryKey: [...HUB_SOURCES_KEY, profileScopeKey(profile)],
    // SkillInfo has provenance, not a source identifier. Read the installed
    // lock map rather than guessing identity from a community skill's name.
    // The older getSkillHubSources helper cannot carry a connection pin.
    queryFn: () =>
      window.hermesDesktop.api<SkillHubSourcesResponse>({
        ...capabilityScoped(profile),
        path: '/api/skills/hub/sources',
        timeoutMs: 45_000
      }),
    enabled: hasHubSkills,
    staleTime: 60_000,
    retry: false
  })

  const catalog = useMemo(() => {
    const skillsById = new Map<string, SkillInfo>()
    const skillsByName = new Map(skills.map(skill => [skill.name, skill]))
    const installedByName = new Map<string, CatalogEntry>()
    const installedByIdentifier = new Map<string, CatalogEntry>()
    const installedIdentifiers = new Set(Object.keys(hubData?.installed ?? {}))

    const localEntries = parseCatalog(
      'skills',
      skills.map(skill => ({
        name: skill.name,
        description: skill.description,
        category: skill.category,
        source: skill.provenance === 'bundled' ? 'built-in' : skill.provenance === 'hub' ? 'hub' : 'local'
      }))
    ).map(entry => {
      const skill = skillsByName.get(entry.name)!
      const installed = { ...entry, id: `installed:${skill.name}`, installIdentifier: null }
      skillsById.set(installed.id, skill)
      installedByName.set(skill.name, installed)

      return installed
    })

    for (const [identifier, record] of Object.entries(hubData?.installed ?? {})) {
      const installed = record.name ? installedByName.get(record.name) : undefined

      if (installed && skillsById.get(installed.id)?.provenance === 'hub') {
        installedByIdentifier.set(identifier, installed)
      }
    }

    const official = officialData?.skills ?? []

    const officialEntries = parseCatalog(
      'skills',
      official.map(skill => ({
        ...skill,
        source: 'optional',
        installIdentifier: skill.identifier
      }))
    )

    const officialByIdentifier = new Map(officialEntries.map(entry => [entry.identifier, entry]))

    for (const skill of official) {
      if (!skill.installed) {
        continue
      }

      installedIdentifiers.add(skill.identifier)
      const installed = installedByName.get(skill.name)

      if (installed && skillsById.get(installed.id)?.provenance === 'hub') {
        installedByIdentifier.set(skill.identifier, installed)
      }
    }

    const officialFor = (entry: CatalogEntry) => {
      if (entry.source !== 'optional') {
        return undefined
      }

      const identifier = entry.installIdentifier ?? entry.identifier
      const exact = officialByIdentifier.get(identifier)

      if (exact) {
        return exact
      }

      if (identifier !== entry.name && identifier !== `official/${entry.name}`) {
        return undefined
      }

      // Old public snapshots omit optional identifiers. Only the official
      // optional namespace can use this fallback, never community lookalikes.
      return officialEntries.find(skill => skill.name === entry.name && skill.category === entry.category)
    }

    const matchInstalled = (entry: CatalogEntry): CatalogEntry | undefined => {
      const exact = installedByIdentifier.get(entry.installIdentifier ?? entry.identifier)

      if (exact) {
        return exact
      }

      if (entry.source === 'built-in') {
        const bundled = installedByName.get(entry.name)

        if (bundled && skillsById.get(bundled.id)?.provenance === 'bundled') {
          return bundled
        }
      }

      const optional = officialFor(entry)

      return optional ? (installedByIdentifier.get(optional.identifier) ?? optional) : undefined
    }

    // CatalogBrowser merges these supplemental rows with the public feed.
    // Optional rows remain available even offline / ahead of the snapshot;
    // membership here is NOT installation — isInstalled owns that decision.
    const entries = [
      ...localEntries,
      ...officialEntries.filter(entry => !installedByIdentifier.has(entry.identifier) && !skillsByName.has(entry.name))
    ]

    return { entries, skillsById, skillsByName, installedIdentifiers, matchInstalled, officialFor }
  }, [skills, hubData, officialData])

  // Skills install by name, so a same-named entry can never be added beside the installed one.
  const isInstalled = useCallback(
    (entry: CatalogEntry) => {
      if (catalog.skillsById.has(entry.id) || catalog.skillsByName.has(entry.name)) {
        return true
      }

      const matched = catalog.matchInstalled(entry)

      if (matched && catalog.skillsById.has(matched.id)) {
        return true
      }

      const optional = catalog.officialFor(entry)

      return catalog.installedIdentifiers.has(optional?.identifier ?? entry.installIdentifier ?? entry.identifier)
    },
    [catalog]
  )

  const installIdentifier = (entry: CatalogEntry) => catalog.officialFor(entry)?.identifier ?? entry.installIdentifier

  const identityPending = hasHubSkills && (hubPending || Boolean(hubError))

  const install = (entry: CatalogEntry) => {
    const identifier = installIdentifier(entry)

    if (installedPending || identityPending || !identifier || isInstalled(entry) || pending.current.has(identifier)) {
      return
    }

    pending.current.add(identifier)
    setInstalling(new Set(pending.current))
    notify({ kind: 'success', title: h.installStarted(entry.name), message: h.actionLog })
    void installHubSkill(identifier, profile)
      .catch(err => {
        if (mounted.current) {
          notifyHubActionFailed(err, h.actionFailed, entry.name, profile)
        }
      })
      .finally(() => {
        pending.current.delete(identifier)

        if (mounted.current) {
          setInstalling(new Set(pending.current))
        }
      })
  }

  return (
    <CatalogBrowser
      actions={actions}
      installedEntries={catalog.entries}
      installedPending={installedPending || identityPending}
      isInstalled={isInstalled}
      isInstalling={entry => installing.has(installIdentifier(entry) ?? '')}
      kind="skills"
      matchInstalled={catalog.matchInstalled}
      notice={
        <>
          {notice}
          {(officialError || hubError) && (
            <CatalogAlert
              onRetry={() => {
                if (officialError) {
                  void refreshOfficial()
                }

                if (hubError) {
                  void refreshHub()
                }
              }}
              retryLabel={t.skills.refresh}
              title={t.skills.skillsLoadFailed}
            >
              {(hubError ?? officialError)?.message}
            </CatalogAlert>
          )}
          {hasHubSkills && hubPending && !installedPending && !notice && (
            <p className="px-3 py-2 text-xs text-(--ui-text-tertiary)" role="status">
              {t.skills.loading}
            </p>
          )}
        </>
      }
      onInstall={install}
      onQueryChange={onQueryChange}
      query={query}
      renderInstalledAction={entry => {
        const skill = catalog.skillsById.get(entry.id)

        return skill && renderInstalledAction ? renderInstalledAction(skill) : null
      }}
      renderInstalledDetail={entry => {
        const skill = catalog.skillsById.get(entry.id)

        return skill ? renderInstalledDetail(skill) : null
      }}
    />
  )
}
