import type { QueryClient } from '@tanstack/react-query'
import {
  MutationObserver,
  type Query,
  type QueryCacheNotifyEvent,
  type QueryKey,
  QueryObserver,
  type QueryObserverResult,
  queryOptions,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
  type UseQueryResult
} from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'

import { $apiRequestScope, getApiRequestConnection, getApiRequestProfile } from '@/api/client'
import type { LocalModelsScope } from '@/api/local-models'
import {
  getLocalCatalog,
  getLocalHardware,
  getLocalModelsJobs,
  getLocalModelsStatus,
  installLocalRuntime
} from '@/hermes'
import { translateNow } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { useStoresSelector } from '@/lib/use-session-slice'
import { notify, notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'
import type { LocalCatalogModel, LocalHardware, LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

export interface LocalModelsOwner extends LocalModelsScope {
  // Legacy primary routes have no registry pin. Fence them by endpoint instead.
  legacyBaseUrl?: string
}

export function localModelsOwner(profile?: string, connectionId?: string | null): LocalModelsOwner {
  const pin: string | null = connectionId ?? getApiRequestConnection()

  return {
    connectionId: pin,
    profile: profile ?? getApiRequestProfile() ?? 'default',
    ...(pin ? {} : { legacyBaseUrl: $connection.get()?.baseUrl ?? '' })
  }
}

export function useLocalModelsOwner(profile?: string, connectionId?: string | null): LocalModelsOwner {
  const identity: string = useStoresSelector([$apiRequestScope, $connection], (): string =>
    JSON.stringify(localModelsOwner(profile, connectionId))
  )

  return useMemo((): LocalModelsOwner => JSON.parse(identity) as LocalModelsOwner, [identity])
}

export function localModelsKey(owner: LocalModelsOwner, resource?: string): QueryKey {
  return [
    'local-models',
    owner.connectionId ?? `legacy:${owner.legacyBaseUrl ?? ''}`,
    owner.profile,
    ...(resource ? [resource] : [])
  ]
}

// A legacy (unpinned) owner is fenced by the endpoint it was minted for;
// once the primary connection moves, its requests and toasts must stop.
export function isLocalModelsOwnerLive(owner: LocalModelsOwner): boolean {
  return Boolean(owner.connectionId) || owner.legacyBaseUrl === ($connection.get()?.baseUrl ?? '')
}

export function localModelsRequestScope(owner: LocalModelsOwner): LocalModelsScope {
  assertLocalModelsOwnerLive(owner)

  return { connectionId: owner.connectionId, profile: owner.profile }
}

export function assertLocalModelsOwnerLive(owner: LocalModelsOwner): void {
  if (!isLocalModelsOwnerLive(owner)) {
    throw new Error(translateNow('settings.localModels.connectionChanged'))
  }
}

export function isCurrentLocalModelsOwner(owner: LocalModelsOwner): boolean {
  return JSON.stringify(localModelsKey(owner)) === JSON.stringify(localModelsKey(localModelsOwner()))
}

export function localModelsNotificationTitle(owner: LocalModelsOwner): string {
  assertLocalModelsOwnerLive(owner)
  const title: string = translateNow('settings.localModels.title')

  return isCurrentLocalModelsOwner(owner)
    ? title
    : `${title} · ${owner.connectionId ?? owner.legacyBaseUrl} / ${owner.profile}`
}

export function localModelsStatusOptions(owner: LocalModelsOwner): UseQueryOptions<LocalModelsStatus> {
  return queryOptions({
    queryKey: localModelsKey(owner, 'status'),
    queryFn: async (): Promise<LocalModelsStatus> => {
      const status: LocalModelsStatus = await getLocalModelsStatus(localModelsRequestScope(owner))
      assertLocalModelsOwnerLive(owner)

      return status
    },
    retry: false
  })
}

interface StatusWatch {
  observer: QueryObserver<LocalModelsStatus>
  users: number
}
const statusWatches: WeakMap<QueryClient, Map<string, StatusWatch>> = new WeakMap()

export function useLocalModelsStatus(
  owner: LocalModelsOwner,
  enabled: boolean = true
): UseQueryResult<LocalModelsStatus> {
  const client: QueryClient = useQueryClient()
  const result: UseQueryResult<LocalModelsStatus> = useQuery({ ...localModelsStatusOptions(owner), enabled: false })
  useEffect((): (() => void) | undefined => {
    if (!enabled) {
      return
    }

    let watches: Map<string, StatusWatch> | undefined = statusWatches.get(client)

    if (!watches) {
      watches = new Map()
      statusWatches.set(client, watches)
    }

    const key: string = JSON.stringify(localModelsKey(owner, 'status'))
    let watch: StatusWatch | undefined = watches.get(key)

    if (!watch) {
      const observer: QueryObserver<LocalModelsStatus> = new QueryObserver<LocalModelsStatus>(client, {
        ...localModelsStatusOptions(owner),
        refetchInterval: 2_000
      })

      watch = { observer, users: 0 }
      watches.set(key, watch)
      observer.subscribe((): void => {})
    }

    const acquired: StatusWatch = watch
    acquired.users += 1

    return (): void => {
      acquired.users -= 1

      if (acquired.users === 0) {
        acquired.observer.destroy()
        watches.delete(key)
      }
    }
  }, [client, owner, enabled])

  return result
}

export function localModelsCatalogOptions(owner: LocalModelsOwner): UseQueryOptions<LocalCatalogModel[]> {
  return queryOptions({
    queryKey: localModelsKey(owner, 'catalog'),
    queryFn: async (): Promise<LocalCatalogModel[]> => {
      const { models } = await getLocalCatalog(localModelsRequestScope(owner))
      assertLocalModelsOwnerLive(owner)

      return models
    },
    retry: false
  })
}

export function localModelsHardwareOptions(owner: LocalModelsOwner): UseQueryOptions<LocalHardware> {
  return queryOptions({
    queryKey: localModelsKey(owner, 'hardware'),
    queryFn: async (): Promise<LocalHardware> => {
      const hardware: LocalHardware = await getLocalHardware(localModelsRequestScope(owner))
      assertLocalModelsOwnerLive(owner)

      return hardware
    },
    retry: false
  })
}

const EMPTY_JOBS: readonly LocalRuntimeJob[] = []

function isActive(status: LocalRuntimeJob['status']): boolean {
  return status === 'paused' || status === 'running'
}

export function localModelsJobsOptions(owner: LocalModelsOwner): UseQueryOptions<readonly LocalRuntimeJob[]> {
  return queryOptions({
    queryKey: localModelsKey(owner, 'jobs'),
    refetchOnMount: 'always',
    queryFn: async (): Promise<readonly LocalRuntimeJob[]> => {
      const { jobs } = await getLocalModelsJobs(localModelsRequestScope(owner))
      assertLocalModelsOwnerLive(owner)

      // Normalize backend ordering; QueryClient does all structural sharing.
      return [...jobs].sort((a: LocalRuntimeJob, b: LocalRuntimeJob): number => a.job_id.localeCompare(b.job_id))
    },
    refetchInterval: (query: Query<readonly LocalRuntimeJob[]>): number | false => {
      if (!isLocalModelsOwnerLive(owner)) {
        return false
      }

      const jobs: readonly LocalRuntimeJob[] = query.state.data ?? EMPTY_JOBS

      return jobs.some((job: LocalRuntimeJob): boolean => job.status === 'running')
        ? 700
        : jobs.some((job: LocalRuntimeJob): boolean => job.status === 'paused')
          ? 3_000
          : false
    },
    refetchIntervalInBackground: true,
    retry: false
  })
}

// Only observers and transition identities live here. All payloads, request
// deduplication, structural sharing and poll scheduling belong to QueryClient.
const watchers: WeakMap<QueryClient, Map<string, QueryObserver<readonly LocalRuntimeJob[]>>> = new WeakMap()

export function refreshLocalModels(owner: LocalModelsOwner, client: QueryClient = queryClient): void {
  if (!owner.connectionId && !isCurrentLocalModelsOwner(owner)) {
    return
  }

  void client.invalidateQueries(
    { queryKey: localModelsKey(owner), predicate: (query: Query): boolean => query.queryKey[3] !== 'jobs' },
    { cancelRefetch: false }
  )
  void client.invalidateQueries(
    {
      predicate: (query: Query): boolean =>
        query.queryKey[0] === 'model-options' &&
        query.queryKey[1] === owner.profile &&
        (owner.connectionId ? query.queryKey[4] === owner.connectionId : query.queryKey.length === 3)
    },
    { cancelRefetch: false }
  )
}

export function watchLocalRuntimeJobs(
  owner: LocalModelsOwner = localModelsOwner(),
  client: QueryClient = queryClient
): void {
  let owners: Map<string, QueryObserver<readonly LocalRuntimeJob[]>> | undefined = watchers.get(client)

  if (!owners) {
    owners = new Map()
    watchers.set(client, owners)
    client.getQueryCache().subscribe((event: QueryCacheNotifyEvent): void => {
      if (event.type === 'removed') {
        const id: string = JSON.stringify(event.query.queryKey)
        const observer: QueryObserver<readonly LocalRuntimeJob[]> | undefined = watchers.get(client)?.get(id)

        if (!observer) {
          return
        }

        observer.destroy()
        watchers.get(client)?.delete(id)
      }
    })
  }

  const key: QueryKey = localModelsKey(owner, 'jobs')
  const id: string = JSON.stringify(key)

  if (owners.has(id)) {
    // In-flight refresh signals need one trailing read. All callers await the
    // same Query promise, then QueryClient coalesces their trailing refetches.
    if (client.getQueryState(key)?.fetchStatus === 'fetching') {
      void client
        .fetchQuery({ ...localModelsJobsOptions(owner), staleTime: 0 })
        .catch((): void => {})
        .then((): Promise<void> => client.refetchQueries({ queryKey: key, exact: true }, { cancelRefetch: false }))
    } else {
      void client.refetchQueries({ queryKey: key, exact: true }, { cancelRefetch: false })
    }

    return
  }

  const observer: QueryObserver<readonly LocalRuntimeJob[]> = new QueryObserver<readonly LocalRuntimeJob[]>(
    client,
    localModelsJobsOptions(owner)
  )

  owners.set(id, observer)
  let active: Set<string> = new Set()
  let downloading: Set<string> = new Set()
  const settledNotified: Set<string> = new Set()
  observer.subscribe((result: QueryObserverResult<readonly LocalRuntimeJob[]>): void => {
    if (!result.isSuccess || result.isFetching) {
      return
    }

    const jobs: readonly LocalRuntimeJob[] = result.data

    const nextDownloads: Set<string> = new Set(
      runningModelDownloads(jobs).map((job: LocalRuntimeJob): string => job.job_id)
    )

    let changed: boolean = [...downloading].some((jobId: string): boolean => !nextDownloads.has(jobId))
    // A stale legacy owner still settles its bookkeeping; it just has no
    // toast to show — a throw here would abort the whole listener instead.
    const live: boolean = isLocalModelsOwnerLive(owner)

    for (const job of jobs) {
      if (isActive(job.status) || !active.has(job.job_id) || settledNotified.has(job.job_id)) {
        continue
      }

      settledNotified.add(job.job_id)
      changed = true

      if (live) {
        notifySettled(owner, job)
      }
    }

    active = new Set(
      jobs
        .filter((job: LocalRuntimeJob): boolean => isActive(job.status))
        .map((job: LocalRuntimeJob): string => job.job_id)
    )
    downloading = nextDownloads

    if (changed) {
      refreshLocalModels(owner, client)
    }
  })
}

export function useLocalRuntimeJobs<T>(
  owner: LocalModelsOwner,
  select: (jobs: readonly LocalRuntimeJob[]) => T,
  enabled: boolean = true
): T {
  const client: QueryClient = useQueryClient()
  const result: UseQueryResult<T> = useQuery({ ...localModelsJobsOptions(owner), enabled: false, select })
  useEffect((): void => {
    if (enabled) {
      watchLocalRuntimeJobs(owner, client)
    }
  }, [client, owner, enabled])

  return result.data ?? select(EMPTY_JOBS)
}

function notifySettled(owner: LocalModelsOwner, job: LocalRuntimeJob): void {
  if (job.status === 'done') {
    notify({
      durationMs: 6_000,
      kind: 'success',
      title: localModelsNotificationTitle(owner),
      message:
        job.kind === 'model-download'
          ? translateNow('settings.localModels.downloadDoneToast', job.target)
          : job.kind === 'model-activate'
            ? translateNow('settings.localModels.activateDoneToast', job.target)
            : job.kind === 'quickstart'
              ? translateNow('settings.localModels.quickstartDoneToast', job.target)
              : translateNow('settings.localModels.installDoneToast')
    })
  } else {
    notifyError(
      new Error(job.error ?? job.detail ?? 'failed'),
      localModelsNotificationTitle(owner) +
        ': ' +
        (job.kind === 'model-download'
          ? translateNow('settings.localModels.downloadFailed', job.target)
          : job.kind === 'model-activate'
            ? translateNow('settings.localModels.activateFailed', job.target)
            : job.kind === 'quickstart'
              ? translateNow('settings.localModels.quickstartFailed')
              : translateNow('settings.localModels.installFailed'))
    )
  }
}

// Selector: the in-flight (running or paused) download job for a catalog
// model id, if any. Paused stays visible — the row parks, it doesn't
// vanish (progress loss is information loss).
export function runningDownloadFor(jobs: readonly LocalRuntimeJob[], modelId: string): LocalRuntimeJob | null {
  return jobs.find(j => j.kind === 'model-download' && isActive(j.status) && j.model_id === modelId) ?? null
}

// Selector: every model on its way to the library right now — plain
// downloads plus quickstart runs while they are still fetching bytes
// (later quickstart phases mean the model is staged and activating),
// paused ones included. The model picker renders these as disabled
// progress rows.
const DOWNLOAD_PHASES = new Set([
  'starting',
  'installing-runtime',
  'downloading-runtime',
  'unpacking-runtime',
  'verifying-runtime',
  'downloading'
])

export function runningModelDownloads(jobs: readonly LocalRuntimeJob[]): LocalRuntimeJob[] {
  return jobs.filter(
    j =>
      isActive(j.status) && (j.kind === 'model-download' || (j.kind === 'quickstart' && DOWNLOAD_PHASES.has(j.phase)))
  )
}

export function runningRuntimeInstall(jobs: readonly LocalRuntimeJob[]): LocalRuntimeJob | null {
  return jobs.find(j => j.kind === 'runtime-install' && isActive(j.status)) ?? null
}

interface RuntimeInstallResult {
  backend: string
  job_id: string
  tag: string
}

export function localRuntimeInstallStarting(
  owner: LocalModelsOwner = localModelsOwner(),
  client: QueryClient = queryClient
): boolean {
  return client.isMutating({ mutationKey: localModelsKey(owner, 'install') }) > 0
}

export function localRuntimeInstallBusy(
  owner: LocalModelsOwner = localModelsOwner(),
  client: QueryClient = queryClient
): boolean {
  const jobs: readonly LocalRuntimeJob[] = client.getQueryData(localModelsKey(owner, 'jobs')) ?? EMPTY_JOBS

  return (
    localRuntimeInstallStarting(owner, client) ||
    jobs.some(
      (job: LocalRuntimeJob): boolean =>
        isActive(job.status) && (job.kind === 'runtime-install' || job.kind === 'quickstart')
    )
  )
}

export async function startLocalRuntimeInstall(
  owner: LocalModelsOwner = localModelsOwner(),
  client: QueryClient = queryClient
): Promise<void> {
  if (localRuntimeInstallBusy(owner, client)) {
    return
  }

  const observer: MutationObserver<RuntimeInstallResult, Error, void> = new MutationObserver(client, {
    mutationKey: localModelsKey(owner, 'install'),
    mutationFn: (): Promise<RuntimeInstallResult> => installLocalRuntime(undefined, localModelsRequestScope(owner))
  })

  try {
    await observer.mutate()
    watchLocalRuntimeJobs(owner, client)
    await client.fetchQuery({ ...localModelsJobsOptions(owner), staleTime: 0 })
  } catch (error) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(error, translateNow('settings.localModels.installFailed'))
    }
  }
}

const updateNotified: Set<string> = new Set()

export async function checkLocalRuntimeUpdate(owner: LocalModelsOwner = localModelsOwner()): Promise<void> {
  const identity: string = JSON.stringify(localModelsKey(owner))

  if (updateNotified.has(identity)) {
    return
  }

  try {
    const status: LocalModelsStatus = await queryClient.fetchQuery(localModelsStatusOptions(owner))
    assertLocalModelsOwnerLive(owner)

    if (status.enabled && status.update_available && !updateNotified.has(identity)) {
      updateNotified.add(identity)
      notify({
        durationMs: 10_000,
        kind: 'info',
        title: localModelsNotificationTitle(owner),
        message: translateNow('settings.localModels.updateToast', status.configured_tag)
      })
    }
  } catch {
    // Older backends may not have this endpoint. The next boot can retry.
  }
}
