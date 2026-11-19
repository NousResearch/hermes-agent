import { type QueryClient, useQueryClient } from '@tanstack/react-query'
import { type ReactElement, useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import {
  downloadBrowsedModel,
  type HFFileGroup,
  type HFSearchHit,
  listHFRepoFiles,
  searchHFModels,
  sideloadLocalModel
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Cpu, Download, FolderOpen, Loader2, Search } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  isCurrentLocalModelsOwner,
  localModelsNotificationTitle,
  type LocalModelsOwner,
  localModelsRequestScope,
  refreshLocalModels,
  runningDownloadFor,
  useLocalRuntimeJobs,
  watchLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import { notify, notifyError } from '@/store/notifications'
import type { LocalRuntimeJob } from '@/types/hermes'

import { downloadStatusText, gbLabel, LocalModelDownloadActions, ProgressBar } from './local-model-download-progress'
import { useScopedLocalModelsOwner } from './local-models-owner'
import { ListRow, Pill, SettingsSection } from './primitives'

function fitTone(fit: HFFileGroup['fit']): 'destructive' | 'muted' | 'success' | 'warn' {
  if (fit === 'fits-gpu') {
    return 'success'
  }

  if (fit === 'needs-ram') {
    return 'warn'
  }

  if (fit === 'too-big') {
    return 'destructive'
  }

  return 'muted'
}

function browsedModelId(group: HFFileGroup): string {
  // Mirrors the backend's derivation: first file's name, split-part
  // suffix stripped — the id the download job carries.
  const first = group.paths[0].split('/').pop() ?? group.paths[0]

  return first.replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '')
}

export function LocalModelsBrowseSection(): ReactElement {
  const owner: LocalModelsOwner = useScopedLocalModelsOwner()
  const client: QueryClient = useQueryClient()
  const { t } = useI18n()
  const copy = t.settings.localModels
  const onChanged = useCallback((): void => refreshLocalModels(owner, client), [owner, client])

  const jobs: readonly LocalRuntimeJob[] = useLocalRuntimeJobs(
    owner,
    (value: readonly LocalRuntimeJob[]): readonly LocalRuntimeJob[] => value,
    false
  )

  const [query, setQuery] = useState<string>('')
  const [hits, setHits] = useState<HFSearchHit[]>([])
  const [searching, setSearching] = useState<boolean>(false)
  const [openRepo, setOpenRepo] = useState<null | string>(null)
  const [files, setFiles] = useState<HFFileGroup[]>([])
  const [listing, setListing] = useState<boolean>(false)
  const [error, setError] = useState<null | string>(null)
  // Guard against the past: a stale search result must never overwrite a
  // newer query's hits (the desktop guide's out-of-order rule).
  const searchSeq = useRef(0)

  useEffect(() => {
    const q = query.trim()

    if (q.length < 2) {
      setHits([])
      setSearching(false)

      return
    }

    const seq = ++searchSeq.current
    setSearching(true)

    const handle: ReturnType<typeof setTimeout> = setTimeout((): void => {
      searchHFModels(q, 20, localModelsRequestScope(owner))
        .then((r: { hits: HFSearchHit[] }): void => {
          if (searchSeq.current === seq) {
            setHits(r.hits)
            setError(null)
          }
        })
        .catch((e: Error) => {
          if (searchSeq.current === seq) {
            setError(e.message)
          }
        })
        .finally(() => {
          if (searchSeq.current === seq) {
            setSearching(false)
          }
        })
    }, 350)

    return () => clearTimeout(handle)
  }, [query, owner])

  const openFiles = useCallback(
    (repo: string): void => {
      setOpenRepo(repo)
      setFiles([])
      setListing(true)
      listHFRepoFiles(repo, localModelsRequestScope(owner))
        .then((r: { files: HFFileGroup[] }): void => setFiles(r.files))
        .catch((e: Error) => setError(e.message))
        .finally(() => setListing(false))
    },
    [owner]
  )

  const startBrowsedDownload = useCallback(
    (repo: string, group: HFFileGroup): void => {
      downloadBrowsedModel(repo, group.paths, localModelsRequestScope(owner))
        .then((r: Awaited<ReturnType<typeof downloadBrowsedModel>>): void => {
          if (r.already_downloaded) {
            notify({
              durationMs: 3_000,
              kind: 'info',
              message: copy.browseAlreadyDownloaded,
              title: localModelsNotificationTitle(owner)
            })

            return
          }

          // Same feedback loop as catalog downloads: the job store polls
          // and the tile renders live progress from it.
          watchLocalRuntimeJobs(owner, client)
          notify({
            durationMs: 3_000,
            kind: 'info',
            message: copy.browseDownloadStarted.replace('{name}', r.model_id),
            title: localModelsNotificationTitle(owner)
          })
          onChanged()
        })
        .catch((e: Error): void => {
          if (isCurrentLocalModelsOwner(owner)) {
            notifyError(e, copy.browseTitle)
          }
        })
    },
    [copy.browseAlreadyDownloaded, copy.browseDownloadStarted, copy.browseTitle, onChanged, owner, client]
  )

  const sideload = useCallback((): void => {
    window.hermesDesktop
      .selectPaths({ filters: [{ extensions: ['gguf'], name: 'GGUF models' }], title: copy.sideloadTitle })
      .then((paths: string[]): Promise<void> | undefined => {
        if (!paths.length) {
          return
        }

        return sideloadLocalModel(paths[0], localModelsRequestScope(owner)).then(
          (r: Awaited<ReturnType<typeof sideloadLocalModel>>): void => {
            notify({
              durationMs: 3_000,
              kind: 'success',
              message: r.already_present
                ? copy.sideloadAlreadyPresent
                : copy.sideloadDone.replace('{name}', r.model_id),
              title: localModelsNotificationTitle(owner)
            })
            onChanged()
          }
        )
      })
      .catch((e: Error): void => {
        if (isCurrentLocalModelsOwner(owner)) {
          notifyError(e, copy.browseTitle)
        }
      })
  }, [copy.browseTitle, copy.sideloadAlreadyPresent, copy.sideloadDone, copy.sideloadTitle, onChanged, owner])

  return (
    <SettingsSection
      aside={
        <Button onClick={sideload} size="sm" variant="outline">
          <FolderOpen className="mr-1 size-3.5" />
          {copy.sideloadButton}
        </Button>
      }
      icon={Search}
      title={copy.browseTitle}
    >
      <div id="local-model-browse">
        <p className="text-[0.75rem] text-muted-foreground">{copy.browseHint}</p>

        <SearchField
          containerClassName="w-full"
          inputClassName="flex-1"
          onChange={setQuery}
          placeholder={copy.browsePlaceholder}
          value={query}
        />

        {searching && (
          <p className="flex items-center gap-2 text-[0.75rem] text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            {copy.browseSearching}
          </p>
        )}

        {error && <p className="text-[0.75rem] text-destructive">{error}</p>}

        <div className="grid gap-1">
          {hits.map(hit => (
            <div key={hit.repo}>
              <ListRow
                action={
                  <Button onClick={() => openFiles(hit.repo)} size="sm" variant="ghost">
                    {openRepo === hit.repo ? copy.browseRefresh : copy.browseShowFiles}
                  </Button>
                }
                description={
                  <span>
                    {Intl.NumberFormat().format(hit.downloads)} {copy.browseDownloads}
                    {' · '}
                    {Intl.NumberFormat().format(hit.likes)} {copy.browseLikes}
                    {hit.gated ? ` · ${copy.browseGated}` : ''}
                  </span>
                }
                title={<span className="font-mono text-[0.8rem]">{hit.repo}</span>}
              />

              {openRepo === hit.repo && (
                <div className="ml-4 grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-1.5 border-l border-(--ui-border) py-1 pl-3">
                  {listing && (
                    <p className="col-span-full flex items-center gap-2 py-1 text-[0.75rem] text-muted-foreground">
                      <Loader2 className="size-3 animate-spin" />
                      {copy.browseListing}
                    </p>
                  )}

                  {!listing && files.length === 0 && (
                    <p className="col-span-full py-1 text-[0.75rem] text-muted-foreground">{copy.browseNoGguf}</p>
                  )}

                  {files.map(group => {
                    const dJob = runningDownloadFor(jobs, browsedModelId(group))

                    return (
                      <div
                        className={cn(
                          'flex flex-col gap-1 rounded-md border border-(--ui-border) px-2.5 py-1.5',
                          group.fit === 'too-big' && 'opacity-45'
                        )}
                        key={group.label}
                      >
                        <span className="flex w-full items-center justify-between gap-2">
                          <span className="truncate font-mono text-[0.75rem]">
                            {group.label}
                            {group.paths.length > 1 ? ` ×${group.paths.length}` : ''}
                          </span>

                          <Button
                            aria-label={copy.browseDownloadAria.replace('{name}', group.label)}
                            className="h-6 shrink-0 px-2"
                            disabled={group.fit === 'too-big' || Boolean(dJob)}
                            onClick={() => startBrowsedDownload(hit.repo, group)}
                            size="sm"
                            variant="ghost"
                          >
                            {dJob ? <Loader2 className="size-3.5 animate-spin" /> : <Download className="size-3.5" />}
                          </Button>
                        </span>

                        {dJob ? (
                          <>
                            <ProgressBar paused={dJob.status === 'paused'} percent={dJob.percent} />

                            <span className="text-[0.68rem] text-muted-foreground">
                              {downloadStatusText(dJob, copy)}
                            </span>

                            <LocalModelDownloadActions job={dJob} owner={owner} />
                          </>
                        ) : (
                          <span className="flex items-center justify-between gap-2">
                            <Pill tone={fitTone(group.fit)}>
                              <Cpu className="mr-1 size-3" />
                              {group.fit === 'fits-gpu'
                                ? copy.pillFitsGpu
                                : group.fit === 'needs-ram'
                                  ? copy.pillUsesRam
                                  : group.fit === 'too-big'
                                    ? copy.pillTooBig
                                    : copy.browseFitUnknown}
                            </Pill>

                            <span className="shrink-0 text-[0.7rem] text-muted-foreground">
                              {gbLabel(group.total_bytes)}
                            </span>
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </SettingsSection>
  )
}
