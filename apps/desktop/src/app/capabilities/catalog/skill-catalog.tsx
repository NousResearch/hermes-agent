import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { Loader } from '@/components/ui/loader'
import { getOfficialSkills, getSkillHubUpdates, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { HUB_SOURCES_KEY, installHubSkill, notifyHubActionFailed, OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'
import { notify } from '@/store/notifications'
import type { SkillHubSourcesResponse, SkillInfo } from '@/types/hermes'

import { skillSearchTerms } from '../skills/skills-data'

import { CatalogAlert } from './catalog-alert'
import { CatalogBrowser } from './catalog-browser'
import type { CatalogEntry } from './catalog-data'
import { useProviderAction } from './provider-action'
import { useProviderCatalog } from './provider-catalog'
import { ProviderActions, ProviderDetail, ProviderNotices } from './provider-detail'
import { ProviderReviewDialog } from './provider-review-dialog'
import { buildSkillCatalog } from './skill-catalog-data'

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

  // "Try …" nudges from the user's own skills: search reads categories, not just names.
  const searchHints = useMemo(
    () => (skills.length ? skillSearchTerms(skills).map(term => t.common.tryHint(term)) : undefined),
    [skills, t]
  )

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

  const providerSources = (officialData?.provider_sources ?? []).filter(source => source.state !== 'hidden')
  const providers = useProviderCatalog(providerSources, profile, query ?? '')
  const providerAction = useProviderAction(profile)
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

  // Upstream check fetches every hub skill's source: cache it for a while, and
  // never let a slow or failed check hold up the page.
  const { data: updatesData } = useQuery({
    queryKey: [...HUB_SOURCES_KEY, 'updates', profileScopeKey(profile)],
    queryFn: () => getSkillHubUpdates(profile),
    enabled: hasHubSkills,
    staleTime: 15 * 60_000,
    retry: false
  })

  const catalog = buildSkillCatalog(skills, hubData, officialData, updatesData, providerSources)

  // Skills install by name, so a same-named entry can never be added beside the installed one.
  const isInstalled = useCallback(
    (entry: CatalogEntry) => {
      if (entry.provider) { return catalog.skillsById.has(entry.id) || entry.provider.managed_state === 'installed' }

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

  // The name guarantee above only holds for first-party namespaces, where a
  // shared name is the same skill. Community feeds carry distinct skills that
  // reuse popular names, so a community row whose name is taken by an installed
  // skill is neither that skill nor installable beside it — hide it instead of
  // showing it as installed. Rows already merged into an installed one (their
  // id is the installed row's) stay visible.
  const isSuperseded = useCallback(
    (entry: CatalogEntry) =>
      !entry.provider && entry.source !== 'official' && !entry.id.startsWith('installed:') && catalog.skillsByName.has(entry.name),
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
    <>
    {providerAction.state && <ProviderReviewDialog controller={providerAction} key={`${providerAction.state.identifier}:${providerAction.state.action}`} />}
    <CatalogBrowser
      actions={actions}
      detailSuppressed={!!providerAction.state}
      installedEntries={[...catalog.entries, ...providers.entries.filter(entry => !catalog.skillsById.has(entry.id))]}
      installedPending={installedPending || identityPending}
      isInstalled={isInstalled}
      isInstalling={entry => installing.has(installIdentifier(entry) ?? '')}
      isSuperseded={isSuperseded}
      kind="skills"
      matchInstalled={catalog.matchInstalled}
      notice={
        <>
          {notice}
          <ProviderNotices catalog={providers} onRefresh={() => void refreshOfficial()} sources={providerSources} />
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
            <Loader
              className="mx-auto my-2 size-6 text-(--ui-text-tertiary)"
              label={t.skills.loading}
              type="rose-curve"
            />
          )}
        </>
      }
      onInstall={install}
      onQueryChange={onQueryChange}
      query={query}
      renderInstalledAction={entry => {
        if (entry.provider) { return <ProviderActions disabled={!!providerAction.state} entry={entry} onAction={(item, action) => void providerAction.begin(item, action)} /> }
        const skill = catalog.skillsById.get(entry.id)

        return skill && renderInstalledAction ? renderInstalledAction(skill) : null
      }}
      renderInstalledDetail={entry => {
        if (entry.provider) { return <ProviderDetail entry={entry} key={entry.id} profile={profile} /> }
        const skill = catalog.skillsById.get(entry.id)

        if (skill?.provider && !entry.provider) { return <p role="status">{t.skills.loading}</p> }

        return skill ? renderInstalledDetail(skill) : null
      }}
      searchHints={searchHints}
      sources={providerSources.length ? Object.fromEntries(providerSources.map(source => [source.id, source.label])) : undefined}
    />
    </>
  )
}
