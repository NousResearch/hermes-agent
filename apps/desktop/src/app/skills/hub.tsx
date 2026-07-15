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
import {
  getSkillHubSources,
  previewSkillHub,
  scanSkillHub,
  searchSkillsHub,
  type SkillHubResult,
  type SkillHubScanResult
} from '@/hermes'
import { useI18n } from '@/i18n'
import { stripAnsi } from '@/lib/ansi'
import { Loader2 } from '@/lib/icons'
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

// Dedup rank when the same skill surfaces from multiple sources — higher trust
// wins. Mirrors the backend's unified_search `_TRUST_RANK`.
const TRUST_RANK: Record<string, number> = { builtin: 2, trusted: 1, community: 0 }

function trustTone(level: string): string {
  switch (level) {
    case 'builtin':
      return 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'

    case 'trusted':
      return 'bg-emerald-500/15 text-emerald-400'

    default:
      return 'bg-amber-500/15 text-amber-400'
  }
}

function verdictTone(policy: string): string {
  switch (policy) {
    case 'allow':
      return 'text-emerald-400'

    case 'block':
      return 'text-destructive'

    default:
      return 'text-amber-400'
  }
}

// One hub result — a self-contained row that installs/uninstalls ITSELF and
// reads its own action status from the store, so parallel installs never desync.
// `rawInstalled` is the sources/search truth; the store's optimistic override
// wins so the row flips the instant its own action resolves.
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
    void installHubSkill(skill.identifier).catch(err => notifyError(err, h.actionFailed))
  }

  const doUninstall = () => {
    notify({ kind: 'success', title: h.uninstallStarted(skill.name), message: h.actionLog })
    void uninstallHubSkill(skill.identifier, installedName || skill.name).catch(err => notifyError(err, h.actionFailed))
  }

  return (
    <div className="row-hover flex items-start gap-3 rounded-md px-2 py-2.5">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="truncate text-[0.78rem] font-medium text-foreground/85">{skill.name}</span>
          <span className={cn('rounded px-1.5 py-0.5 text-[0.6rem]', trustTone(skill.trust_level))}>
            {h.trust[skill.trust_level] ?? skill.trust_level}
          </span>
          {installed && <span className="text-[0.6rem] text-emerald-400">{h.installed}</span>}
        </div>
        <p className="mt-0.5 line-clamp-2 text-[0.68rem] text-muted-foreground/70">{skill.description}</p>
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <Button onClick={() => onPreview(skill)} size="xs" variant="text">
          {h.preview}
        </Button>
        {installed ? (
          <Button className="hover:text-destructive" disabled={running} onClick={doUninstall} size="xs" variant="text">
            {running && <Loader2 className="size-3 animate-spin" />}
            {running ? h.uninstalling : h.uninstall}
          </Button>
        ) : (
          <Button disabled={running} onClick={doInstall} size="xs" variant="textStrong">
            {running && <Loader2 className="size-3 animate-spin" />}
            {running ? h.installing : h.install}
          </Button>
        )}
      </div>
    </div>
  )
}

interface SkillsHubProps {
  query: string
}

export function SkillsHub({ query }: SkillsHubProps) {
  const { t } = useI18n()
  const h = t.skills.hub

  // Sources + featured + the installed map — one cached fetch, revalidated on
  // mount and re-fetched (from the store) after an action lands.
  const sourcesQuery = useQuery({
    queryKey: HUB_SOURCES_KEY,
    queryFn: getSkillHubSources,
    staleTime: 5 * 60_000
  })

  // Debounced hub search, keyed on the settled query so RQ dedupes/caches per
  // term and abandons stale terms for us (no hand-rolled sequence guard).
  const term = useDebounced(query.trim(), 350)

  // Progressive per-source search: one query per source the backend says is
  // worth hitting individually (it marks index-covered API sources unsearchable
  // so we don't re-hammer ~70 GitHub calls). Each resolves independently, so the
  // list fills in as sources return instead of blocking on the slowest one, and
  // each source shows its own spinner. Stale terms key out and are abandoned.
  const searchableSources = useMemo(
    () => (sourcesQuery.data?.sources ?? []).filter(source => source.searchable !== false),
    [sourcesQuery.data]
  )

  const sourceSearches = useQueries({
    queries: searchableSources.map(source => ({
      queryKey: ['skill-hub-search', term, source.id],
      queryFn: () => searchSkillsHub(term, source.id),
      enabled: term.length > 0,
      staleTime: 60_000
    }))
  })

  // Per-item action lifecycle + log live in the store (store/hub-actions): each
  // row reads ITS own entry, so concurrent installs never desync each other,
  // and an optimistic installed-override flips a row the instant its own action
  // resolves rather than racing the sources refetch.
  const actions = useStore($hubActions)
  const overrides = useStore($hubInstalledOverride)
  const activeLogKey = useStore($hubActiveLog)
  const activeLog = activeLogKey ? actions[activeLogKey] : undefined

  // Preview/scan dialog. Preview is cache-worthy (keyed by identifier); scan is
  // an explicit, on-demand security pass so it stays imperative.
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
      void installHubSkill(identifier).catch(err => notifyError(err, h.actionFailed))
    },
    [h]
  )

  const updateAll = useCallback(() => {
    notify({ kind: 'success', title: h.updateStarted, message: h.actionLog })
    void updateHubSkills().catch(err => notifyError(err, h.actionFailed))
  }, [h])

  const runScan = useCallback(
    (identifier: string) => {
      setScanning(true)
      scanSkillHub(identifier)
        .then(setScan)
        .catch(err => notifyError(err, h.scanFailed))
        .finally(() => setScanning(false))
    },
    [h]
  )

  const openDetail = useCallback((skill: SkillHubResult) => {
    setDetail(skill)
    setScan(null)
  }, [])

  // Per-source progress, keyed by source id (drives the connected-hub chips'
  // spinner/degraded tint while a search is streaming in).
  const searchStateById = new Map<string, { failed: boolean; fetching: boolean }>()
  searchableSources.forEach((source, i) => {
    const q = sourceSearches[i]
    searchStateById.set(source.id, { failed: q.isError, fetching: term.length > 0 && q.isFetching })
  })

  // Merge every source's results, deduped by identifier preferring higher trust
  // (mirrors the backend's unified_search rank). Recomputes as each source lands.
  const results = useMemo(() => {
    const seen = new Map<string, SkillHubResult>()

    for (const q of sourceSearches) {
      for (const r of q.data?.results ?? []) {
        const prev = seen.get(r.identifier)

        if (!prev || (TRUST_RANK[r.trust_level] ?? 0) > (TRUST_RANK[prev.trust_level] ?? 0)) {
          seen.set(r.identifier, r)
        }
      }
    }

    return [...seen.values()].sort(
      (a, b) => (TRUST_RANK[b.trust_level] ?? 0) - (TRUST_RANK[a.trust_level] ?? 0) || a.name.localeCompare(b.name)
    )
  }, [sourceSearches])

  // Installed map: sources seeds it, search results patch it (a term can surface
  // installs the sources list didn't feature); the optimistic override wins so a
  // just-(un)installed row reflects its own outcome without the refetch race.
  const installed = { ...(sourcesQuery.data?.installed ?? {}) }

  for (const q of sourceSearches) {
    Object.assign(installed, q.data?.installed ?? {})
  }

  const isInstalled = (identifier: string) => overrides[identifier] ?? Boolean(installed[identifier])

  const sources = sourcesQuery.data?.sources ?? []
  const featured = sourcesQuery.data?.featured ?? []

  // Still fetching from at least one source; "done" only once every source has
  // settled (so "No results" doesn't flash while slower sources are still in).
  const anyFetching = term.length > 0 && sourceSearches.some(q => q.isFetching)
  const searched = term.length > 0 && sourceSearches.length > 0 && sourceSearches.every(q => !q.isFetching)
  const showLanding = term.length === 0
  const listed = showLanding ? featured : results
  // Only block the whole pane on the first sources landing; after that results
  // stream in progressively while a subtle footer shows more are coming.
  const searching = anyFetching && results.length === 0
  const hasInstalled = Object.keys(installed).length > 0

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Connected hubs — label on its own line, chips below, roomy padding. */}
      <div className="shrink-0 px-4 pt-5 pb-8 text-[0.68rem] text-(--ui-text-tertiary)">
        <span className="mb-1.5 block">{h.connectedHubs}</span>
        <div className="flex flex-wrap items-center gap-1.5">
          {sourcesQuery.isLoading
            ? null
            : sources.map(source => {
                const state = searchStateById.get(source.id)
                const degraded = source.available === false || source.rate_limited === true || state?.failed
                const fetching = state?.fetching ?? false

                return (
                  <span
                    className={cn(
                      'relative rounded px-1.5 py-0.5 text-[0.6rem] transition-opacity',
                      degraded ? 'bg-amber-500/15 text-amber-400' : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)',
                      // While searching, un-hit sources dim so the active ones read clearly.
                      term.length > 0 && !fetching && !state?.failed && 'opacity-55'
                    )}
                    key={source.id}
                  >
                    {/* Spinner overlays the (dimmed) label rather than pushing it,
                        so a chip never resizes as its search starts/finishes. */}
                    <span className={cn(fetching && 'opacity-30')}>{source.label}</span>
                    {fetching && (
                      <span className="absolute inset-0 grid place-items-center">
                        <Loader2 className="size-2.5 animate-spin" />
                      </span>
                    )}
                  </span>
                )
              })}
        </div>
      </div>

      {/* Result summary (left) + Update installed (right) — only when a results
          table is actually on screen, and update only if something's installed. */}
      {listed.length > 0 && (
        <div className="flex shrink-0 items-center justify-between gap-3 px-4 pb-1.5 text-[0.68rem] text-(--ui-text-tertiary)">
          <span className="min-w-0 truncate">
            {term.length > 0 ? h.resultCount(results.length, null) : h.featured}
            {anyFetching && results.length > 0 && (
              <span className="ml-2 text-(--ui-text-quaternary)">{h.searching}</span>
            )}
          </span>

          {hasInstalled && (
            <Button
              className="shrink-0"
              disabled={actions[UPDATE_ALL_KEY]?.running}
              onClick={updateAll}
              size="xs"
              variant="text"
            >
              {actions[UPDATE_ALL_KEY]?.running && <Loader2 className="size-3 animate-spin" />}
              {actions[UPDATE_ALL_KEY]?.running ? h.updating : h.updateAll}
            </Button>
          )}
        </div>
      )}

      {/* Scrollable results. */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4 [scrollbar-gutter:stable]">
        {searching ? (
          <div className="grid min-h-40 place-items-center">
            <PageLoader label={h.searching} />
          </div>
        ) : listed.length === 0 ? (
          <div className="grid min-h-40 place-items-center px-6 text-center">
            <p className="max-w-md text-[0.72rem] text-(--ui-text-tertiary)">
              {searched ? h.noResults : h.landingHint}
            </p>
          </div>
        ) : (
          <div className="flex flex-col">
            {listed.map(skill => (
              <HubSkillRow
                installedName={installed[skill.identifier]?.name ?? null}
                key={skill.identifier}
                onPreview={openDetail}
                rawInstalled={Boolean(installed[skill.identifier])}
                skill={skill}
              />
            ))}
          </div>
        )}
      </div>

      {/* Action log — same resizable, flush-width bottom pane + LogTail surface
          as the MCP logs. ANSI stripped so spawn output reads clean. Tails the
          latest-started action ($hubActiveLog). */}
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
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-hidden">
          {detail && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <span className="truncate">{detail.name}</span>
                  <Badge className={trustTone(detail.trust_level)}>
                    {h.trust[detail.trust_level] ?? detail.trust_level}
                  </Badge>
                </DialogTitle>
                <DialogDescription className="truncate">{detail.identifier}</DialogDescription>
              </DialogHeader>

              <div className="min-h-0 space-y-3 overflow-y-auto">
                {scan && (
                  <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 text-xs">
                    <div className={cn('font-medium', verdictTone(scan.policy))}>
                      {scan.policy === 'allow' ? h.policyAllow : scan.policy === 'block' ? h.policyBlock : h.policyAsk}
                      {' · '}
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
                        {finding.line !== null ? `:${finding.line}` : ''} — {finding.description}
                      </div>
                    ))}
                  </div>
                )}

                {previewQuery.isLoading ? (
                  <PageLoader className="min-h-32" label={h.searching} />
                ) : previewQuery.data ? (
                  <>
                    <pre
                      className="max-h-72 overflow-auto whitespace-pre-wrap wrap-break-word rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 font-mono text-[0.68rem] leading-relaxed"
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
