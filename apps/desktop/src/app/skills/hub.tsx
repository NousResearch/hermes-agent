import { useStore } from '@nanostores/react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'

import { useDebounced } from '@/app/hooks/use-debounced'
import { DetailPane } from '@/app/master-detail'
import { LogTail } from '@/components/chat/log-tail'
import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { SegmentedControl } from '@/components/ui/segmented-control'
import {
  getMcpCatalog,
  getSkillHubSources,
  previewSkillHub,
  scanSkillHub,
  searchSkillsHub,
  type SkillHubResult,
  type SkillHubScanResult
} from '@/hermes'
import { useI18n } from '@/i18n'
import { stripAnsi } from '@/lib/ansi'
import { FileText, Loader2, Package } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $hubActions,
  $hubActiveLog,
  $hubInstalledOverride,
  closeHubLog,
  HUB_SOURCES_KEY,
  installHubSkill,
  uninstallHubSkill,
  UPDATE_ALL_KEY,
  updateHubSkills
} from '@/store/hub-actions'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

import { prettyName } from '../settings/helpers'
import { IntegrationsCatalog, matchesIntegrationCatalogQuery } from '../settings/integrations-catalog'
import { ListRow, Pill, SettingsGroup, SettingsSection } from '../settings/primitives'

const TRUST_RANK: Record<string, number> = { builtin: 2, trusted: 1, community: 0 }

type HubFilter = 'all' | 'integrations' | 'skills'

function trustTone(level: string): 'muted' | 'primary' | 'warn' {
  if (level === 'builtin') {
    return 'primary'
  }

  if (level === 'community') {
    return 'warn'
  }

  return 'muted'
}

function verdictTone(policy: string): string {
  if (policy === 'allow') {
    return 'text-emerald-400'
  }

  if (policy === 'block') {
    return 'text-destructive'
  }

  return 'text-amber-400'
}

function HubSkillRow({
  installedName,
  onPreview,
  rawInstalled,
  skill
}: {
  installedName: null | string
  onPreview: (skill: SkillHubResult) => void
  rawInstalled: boolean
  skill: SkillHubResult
}) {
  const { t } = useI18n()
  const h = t.skills.hub
  const action = useStore($hubActions)[skill.identifier]
  const override = useStore($hubInstalledOverride)[skill.identifier]
  const installed = override ?? rawInstalled
  const running = action?.running ?? false

  const doInstall = () => {
    notify({ kind: 'success', title: h.installStarted(skill.name), message: h.actionLog })
    void installHubSkill(skill.identifier).catch(error => notifyError(error, h.actionFailed))
  }

  const doUninstall = () => {
    notify({ kind: 'success', title: h.uninstallStarted(skill.name), message: h.actionLog })
    void uninstallHubSkill(skill.identifier, installedName || skill.name).catch(error =>
      notifyError(error, h.actionFailed)
    )
  }

  return (
    <ListRow
      action={
        <div className="flex shrink-0 items-center gap-1.5">
          <Button onClick={() => onPreview(skill)} size="xs" variant="text">
            {h.preview}
          </Button>
          {installed ? (
            <Button
              className="hover:text-destructive"
              disabled={running}
              onClick={doUninstall}
              size="xs"
              variant="text"
            >
              {running && <Loader2 className="size-3 animate-spin" />}
              {running ? h.uninstalling : h.uninstall}
            </Button>
          ) : (
            <Button disabled={running} onClick={doInstall} size="xs">
              {running && <Loader2 className="size-3 animate-spin" />}
              {running ? h.installing : h.install}
            </Button>
          )}
        </div>
      }
      description={skill.description}
      title={
        <span className="flex flex-wrap items-center gap-1.5">
          <span>{skill.name}</span>
          <Pill>{h.skill}</Pill>
          <Pill tone={trustTone(skill.trust_level)}>{h.trust[skill.trust_level] ?? skill.trust_level}</Pill>
          {installed && <span className="text-[0.6875rem] font-medium text-emerald-500">{h.installed}</span>}
        </span>
      }
    />
  )
}

function InstalledChip({ kind, name }: { kind: 'integration' | 'skill'; name: string }) {
  const { t } = useI18n()
  const h = t.skills.hub
  const Icon = kind === 'skill' ? FileText : Package
  const label = kind === 'integration' ? prettyName(name) : name

  return (
    <div className="flex min-w-0 items-center gap-2 rounded-[var(--radius-md)] border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/55 px-2.5 py-2">
      <div className="grid size-7 shrink-0 place-items-center rounded-[var(--radius-sm)] bg-background text-(--ui-text-secondary)">
        <Icon className="size-3.5" />
      </div>
      <span className="max-w-40 truncate text-[0.75rem] font-medium text-foreground">{label}</span>
      <span className="text-[0.625rem] text-(--ui-text-tertiary)">{kind === 'skill' ? h.skill : h.integration}</span>
    </div>
  )
}

interface SkillsHubProps {
  query: string
}

export function SkillsHub({ query }: SkillsHubProps) {
  const { t } = useI18n()
  const h = t.skills.hub
  const activeProfile = useStore($activeGatewayProfile)
  const [filter, setFilter] = useState<HubFilter>('all')
  const term = useDebounced(query.trim(), 350)

  const sourcesQuery = useQuery({
    queryKey: HUB_SOURCES_KEY,
    queryFn: getSkillHubSources,
    staleTime: 5 * 60_000
  })

  const integrationsQuery = useQuery({
    queryKey: ['integrations-catalog', normalizeProfileKey(activeProfile)],
    queryFn: getMcpCatalog,
    staleTime: 5 * 60_000
  })

  const searchableSources = useMemo(
    () => (sourcesQuery.data?.sources ?? []).filter(source => source.searchable !== false),
    [sourcesQuery.data]
  )

  const sourceSearches = useQueries({
    queries: searchableSources.map(source => ({
      queryKey: ['skill-hub-search', term, source.id],
      queryFn: () => searchSkillsHub(term, source.id),
      enabled: term.length > 0 && filter !== 'integrations',
      staleTime: 60_000
    }))
  })

  const actions = useStore($hubActions)
  const overrides = useStore($hubInstalledOverride)
  const activeLogKey = useStore($hubActiveLog)
  const activeLog = activeLogKey ? actions[activeLogKey] : undefined
  const [detail, setDetail] = useState<null | SkillHubResult>(null)
  const [scan, setScan] = useState<null | SkillHubScanResult>(null)
  const [scanning, setScanning] = useState(false)

  const previewQuery = useQuery({
    queryKey: ['skill-hub-preview', detail?.identifier],
    queryFn: () => previewSkillHub(detail!.identifier),
    enabled: detail !== null,
    staleTime: 5 * 60_000
  })

  const install = useCallback(
    (identifier: string, name: string) => {
      setDetail(null)
      notify({ kind: 'success', title: h.installStarted(name), message: h.actionLog })
      void installHubSkill(identifier).catch(error => notifyError(error, h.actionFailed))
    },
    [h]
  )

  const updateAll = useCallback(() => {
    notify({ kind: 'success', title: h.updateStarted, message: h.actionLog })
    void updateHubSkills().catch(error => notifyError(error, h.actionFailed))
  }, [h])

  const runScan = useCallback(
    (identifier: string) => {
      setScanning(true)
      scanSkillHub(identifier)
        .then(setScan)
        .catch(error => notifyError(error, h.scanFailed))
        .finally(() => setScanning(false))
    },
    [h]
  )

  const openDetail = useCallback((skill: SkillHubResult) => {
    setDetail(skill)
    setScan(null)
  }, [])

  const searchStateById = new Map<string, { failed: boolean; fetching: boolean }>()
  searchableSources.forEach((source, index) => {
    const sourceQuery = sourceSearches[index]

    searchStateById.set(source.id, {
      failed: sourceQuery.isError,
      fetching: term.length > 0 && filter !== 'integrations' && sourceQuery.isFetching
    })
  })

  const searchResults = useMemo(() => {
    const seen = new Map<string, SkillHubResult>()

    for (const sourceQuery of sourceSearches) {
      for (const result of sourceQuery.data?.results ?? []) {
        const previous = seen.get(result.identifier)

        if (!previous || (TRUST_RANK[result.trust_level] ?? 0) > (TRUST_RANK[previous.trust_level] ?? 0)) {
          seen.set(result.identifier, result)
        }
      }
    }

    return [...seen.values()].sort(
      (a, b) => (TRUST_RANK[b.trust_level] ?? 0) - (TRUST_RANK[a.trust_level] ?? 0) || a.name.localeCompare(b.name)
    )
  }, [sourceSearches])

  const installed = { ...(sourcesQuery.data?.installed ?? {}) }

  for (const sourceQuery of sourceSearches) {
    Object.assign(installed, sourceQuery.data?.installed ?? {})
  }

  const isInstalled = (identifier: string) => overrides[identifier] ?? Boolean(installed[identifier])
  const sources = sourcesQuery.data?.sources ?? []
  const featured = sourcesQuery.data?.featured ?? []
  const integrationEntries = integrationsQuery.data?.entries ?? []
  const matchingIntegrations = integrationEntries.filter(entry => matchesIntegrationCatalogQuery(entry, term))
  const showLanding = term.length === 0

  const anySearching =
    term.length > 0 && filter !== 'integrations' && sourceSearches.some(sourceQuery => sourceQuery.isFetching)

  const skillLoading = sourcesQuery.isLoading || (anySearching && searchResults.length === 0)
  const skillItems = filter === 'integrations' ? [] : showLanding ? featured : searchResults
  const integrationItems = filter === 'skills' ? [] : matchingIntegrations
  const hasInstalledSkills = Object.keys(installed).length > 0

  const installedItems = [
    ...Object.entries(installed).map(([identifier, entry]) => ({
      id: identifier,
      kind: 'skill' as const,
      name: entry.name || identifier.split('/').pop() || identifier
    })),
    ...integrationEntries
      .filter(entry => entry.installed)
      .map(entry => ({ id: entry.name, kind: 'integration' as const, name: entry.name }))
  ].filter(item => filter === 'all' || (filter === 'skills' ? item.kind === 'skill' : item.kind === 'integration'))

  const showSkillSection = filter !== 'integrations' && (skillLoading || sourcesQuery.isError || skillItems.length > 0)
  const showIntegrations = filter !== 'skills'
  const skillsSettled = !skillLoading && !sourcesQuery.isError
  const integrationsSettled = !integrationsQuery.isLoading && !integrationsQuery.isError
  const noMatchingSkills = filter !== 'integrations' && skillItems.length === 0 && skillsSettled
  const noMatchingIntegrations = filter !== 'skills' && integrationItems.length === 0 && integrationsSettled

  const showEmpty =
    (filter === 'skills'
      ? noMatchingSkills
      : filter === 'integrations'
        ? noMatchingIntegrations
        : noMatchingSkills && noMatchingIntegrations) && !(showLanding && installedItems.length > 0)

  const filterOptions = [
    { id: 'all', label: h.filterAll },
    { id: 'skills', label: h.filterSkills },
    { id: 'integrations', label: h.filterIntegrations }
  ] as const

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable]">
        <div className="mx-auto w-full max-w-[52rem] px-5 pb-20 pt-5">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
            <div>
              <SegmentedControl onChange={setFilter} options={filterOptions} value={filter} />
            </div>
            <div className="flex flex-wrap items-center justify-end gap-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
              <span className="mr-0.5">{h.sources}</span>
              {sourcesQuery.isLoading ? (
                <span>{h.connectingHubs}</span>
              ) : (
                sources.map(source => {
                  const state = searchStateById.get(source.id)
                  const degraded = source.available === false || source.rate_limited === true || state?.failed

                  return (
                    <span
                      className={cn(
                        'inline-flex h-6 items-center gap-1 rounded-[var(--radius-sm)] border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/55 px-2 text-[0.625rem] font-medium transition-opacity duration-200',
                        degraded ? 'text-amber-500' : 'text-(--ui-text-secondary)',
                        term.length > 0 &&
                          filter !== 'integrations' &&
                          !state?.fetching &&
                          !state?.failed &&
                          'opacity-55'
                      )}
                      key={source.id}
                    >
                      {state?.fetching && <Loader2 className="size-2.5 animate-spin" />}
                      {source.label}
                    </span>
                  )
                })
              )}
            </div>
          </div>

          {showLanding && installedItems.length > 0 && (
            <SettingsSection icon={Package} meta={h.installedCount(installedItems.length)} title={h.installedTitle}>
              <SettingsGroup>
                <div className="flex flex-wrap gap-2 px-4 py-4 sm:px-5">
                  {installedItems.map(item => (
                    <InstalledChip key={item.kind + '-' + item.id} kind={item.kind} name={item.name} />
                  ))}
                </div>
              </SettingsGroup>
            </SettingsSection>
          )}

          {showSkillSection && (
            <SettingsSection
              aside={
                hasInstalledSkills ? (
                  <Button disabled={actions[UPDATE_ALL_KEY]?.running} onClick={updateAll} size="xs" variant="text">
                    {actions[UPDATE_ALL_KEY]?.running && <Loader2 className="size-3 animate-spin" />}
                    {actions[UPDATE_ALL_KEY]?.running ? h.updating : h.updateAll}
                  </Button>
                ) : undefined
              }
              icon={FileText}
              meta={showLanding ? undefined : h.resultCount(searchResults.length, null)}
              title={showLanding ? h.featured : h.results}
            >
              {skillLoading ? (
                <SettingsGroup>
                  <PageLoader className="min-h-28" label={h.searching} />
                </SettingsGroup>
              ) : sourcesQuery.isError ? (
                <SettingsGroup>
                  <ListRow description={h.loadFailed} title={showLanding ? h.featured : h.results} />
                </SettingsGroup>
              ) : (
                <SettingsGroup>
                  {skillItems.map(skill => (
                    <HubSkillRow
                      installedName={installed[skill.identifier]?.name ?? null}
                      key={skill.identifier}
                      onPreview={openDetail}
                      rawInstalled={Boolean(installed[skill.identifier])}
                      skill={skill}
                    />
                  ))}
                </SettingsGroup>
              )}
            </SettingsSection>
          )}

          {showIntegrations && (
            <IntegrationsCatalog
              description={showLanding ? h.integrationsDescription : null}
              hideInstalled={showLanding}
              hideWhenEmpty
              meta={null}
              query={term}
              title={h.integrationsTitle}
            />
          )}

          {showEmpty && (
            <SettingsGroup>
              <ListRow
                description={showLanding ? h.landingHint : h.noResults}
                title={showLanding ? h.featured : h.results}
              />
            </SettingsGroup>
          )}
        </div>
      </div>

      {activeLogKey && (
        <DetailPane
          defaultCollapsed
          defaultHeight={176}
          id="hub-action-log"
          onClose={closeHubLog}
          title={
            <span className="flex items-center gap-1.5 text-[0.68rem] font-normal text-muted-foreground/60">
              {h.actionLog}
              {activeLog?.running && <Codicon name="loading" size="0.75rem" spinning />}
            </span>
          }
        >
          <LogTail emptyLabel={h.searching} lines={activeLog?.lines.length ? activeLog.lines.map(stripAnsi) : null} />
        </DetailPane>
      )}

      <Dialog onOpenChange={open => !open && setDetail(null)} open={detail !== null}>
        <DialogContent bodyClassName="overflow-hidden" className="max-h-[80vh] max-w-2xl">
          {detail && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <span className="truncate">{detail.name}</span>
                  <Badge
                    className="capitalize"
                    variant={
                      detail.trust_level === 'builtin'
                        ? 'default'
                        : detail.trust_level === 'community'
                          ? 'warn'
                          : 'muted'
                    }
                  >
                    {h.trust[detail.trust_level] ?? detail.trust_level}
                  </Badge>
                </DialogTitle>
                <DialogDescription className="truncate">{detail.identifier}</DialogDescription>
              </DialogHeader>

              <div className="min-h-0 space-y-3 overflow-y-auto">
                {scan && (
                  <div className="rounded-[var(--radius-md)] border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 text-xs">
                    <div className={cn('font-medium', verdictTone(scan.policy))}>
                      {scan.policy === 'allow' ? h.policyAllow : scan.policy === 'block' ? h.policyBlock : h.policyAsk}
                      {' - '}
                      {scan.verdict === 'safe'
                        ? h.verdictSafe
                        : scan.verdict === 'dangerous'
                          ? h.verdictDangerous
                          : h.verdictCaution}
                    </div>
                    <div className="mt-1 text-muted-foreground">
                      {scan.findings.length === 0 ? h.noFindings : h.findings(scan.findings.length)}
                    </div>
                    {scan.findings.slice(0, 12).map((finding, index) => (
                      <div className="mt-1.5 font-mono text-[0.65rem] text-(--ui-text-tertiary)" key={index}>
                        [{finding.severity}] {finding.file}
                        {finding.line !== null ? ':' + finding.line : ''} - {finding.description}
                      </div>
                    ))}
                  </div>
                )}

                {previewQuery.isLoading ? (
                  <PageLoader className="min-h-32" label={h.searching} />
                ) : previewQuery.data ? (
                  <>
                    <pre
                      className="max-h-72 overflow-auto whitespace-pre-wrap wrap-break-word rounded-[var(--radius-md)] border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 font-mono text-[0.68rem] leading-relaxed"
                      data-selectable-text="true"
                    >
                      {previewQuery.data.skill_md || h.noReadme}
                    </pre>
                    {previewQuery.data.files.length > 0 && (
                      <div className="text-xs text-muted-foreground">
                        <span className="font-medium">{h.files}:</span> {previewQuery.data.files.join(', ')}
                      </div>
                    )}
                  </>
                ) : null}
              </div>

              <DialogFooter>
                <Button disabled={scanning} onClick={() => runScan(detail.identifier)} size="sm" variant="text">
                  {scanning ? h.scanning : h.scan}
                </Button>
                <Button
                  disabled={actions[detail.identifier]?.running || isInstalled(detail.identifier)}
                  onClick={() => install(detail.identifier, detail.name)}
                  size="sm"
                >
                  {isInstalled(detail.identifier) ? h.installed : h.install}
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
