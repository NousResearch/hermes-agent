import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { Loader } from '@/components/ui/loader'
import type { HermesGithubPullRequestFilter, HermesGithubPullRequestSummary } from '@/global'
import { useI18n } from '@/i18n'
import { getGithubPullRequestDetail, githubProfileKey, listGithubPullRequests } from '@/lib/desktop-github'

import { PageSearchShell } from '../page-search-shell'

import { PullRequestDetail } from './pull-request-detail'
import { PullRequestList } from './pull-request-list'
import { matchesPullRequest } from './pull-request-utils'

type Tab = 'closed' | 'created' | 'review-requested'
const TABS: Tab[] = ['created', 'review-requested', 'closed']

function filterFor(tab: Tab): HermesGithubPullRequestFilter {
  return tab === 'review-requested'
    ? { kind: 'review-requested', state: 'open', limit: 100 }
    : { kind: 'created', state: tab === 'closed' ? 'closed' : 'open', limit: 100 }
}

export function PullRequestsView() {
  const { t } = useI18n()
  const copy = t.pullRequests
  const [params, setParams] = useSearchParams()
  const requested = params.get('tab') as Tab
  const tab = TABS.includes(requested) ? requested : 'created'
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<HermesGithubPullRequestSummary | null>(null)
  const profile = githubProfileKey()
  const list = useQuery({
    queryKey: ['github-pull-requests', profile, tab],
    queryFn: () => listGithubPullRequests(filterFor(tab)),
    staleTime: 30_000,
    placeholderData: keepPreviousData,
    retry: false
  })
  const detail = useQuery({
    queryKey: ['github-pull-request-detail', profile, selected?.repository, selected?.number],
    queryFn: () => getGithubPullRequestDetail({ repository: selected!.repository, number: selected!.number }),
    enabled: Boolean(selected),
    staleTime: 60_000,
    retry: false
  })
  const items = useMemo(
    () => (list.data?.items ?? []).filter(item => matchesPullRequest(item, search)),
    [list.data?.items, search]
  )
  const tabs = [
    { id: 'created', label: copy.created },
    { id: 'review-requested', label: copy.reviewRequested },
    { id: 'closed', label: copy.closed }
  ].map(value => ({ ...value, meta: value.id === tab ? (list.data?.items.length ?? null) : undefined }))
  const empty = tab === 'created' ? copy.noneCreated : tab === 'review-requested' ? copy.noneReview : copy.noneClosed
  const open = (url: string) => void window.hermesDesktop.openExternal(url)

  const changeTab = (next: string) => {
    setSelected(null)
    setSearch('')
    setParams({ tab: next })
  }
  const setup =
    list.data?.authState === 'gh-missing'
      ? [copy.ghMissing, undefined]
      : list.data?.authState === 'not-authenticated'
        ? [copy.authRequired, copy.authHint]
        : null

  return (
    <PageSearchShell
      activeTab={tab}
      onSearchChange={setSearch}
      onTabChange={changeTab}
      searchHidden={(list.data?.items.length ?? 0) === 0}
      searchPlaceholder={copy.search}
      searchTrailingAction={
        <Button
          aria-label={copy.refresh}
          disabled={list.isFetching}
          onClick={() => void list.refetch()}
          size="icon-sm"
          variant="ghost"
        >
          <Codicon className={list.isFetching ? 'animate-spin' : ''} name="refresh" />
        </Button>
      }
      searchValue={search}
      tabs={tabs}
    >
      {setup ? (
        <EmptyState className="h-full" description={setup[1]} title={setup[0]!} />
      ) : !list.data && list.isPending ? (
        <div className="grid h-full place-items-center">
          <Loader label={copy.loading} />
        </div>
      ) : list.data?.authState === 'error' && !list.data.items.length ? (
        <div className="grid h-full place-items-center">
          <div className="text-center">
            <EmptyState description={list.data.error} title={copy.loadFailed} />
            <Button onClick={() => void list.refetch()} size="sm" variant="secondary">
              {copy.retry}
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid h-full min-w-0 md:grid-cols-[minmax(20rem,42%)_minmax(0,1fr)]">
          <div
            className={`${selected ? 'hidden md:block' : 'block'} min-w-0 overflow-hidden border-r border-(--ui-stroke-tertiary)`}
          >
            {items.length ? (
              <PullRequestList items={items} onOpen={open} onSelect={setSelected} selected={selected?.id} />
            ) : (
              <EmptyState className="h-full" title={empty} />
            )}
          </div>
          <div className={`${selected ? 'block' : 'hidden md:block'} min-w-0 overflow-hidden`}>
            <PullRequestDetail
              copy={copy}
              detail={detail.data}
              error={detail.isError}
              loading={detail.isPending && Boolean(selected)}
              onBack={() => setSelected(null)}
              onCopy={url => void window.hermesDesktop.writeClipboard(url)}
              onOpen={open}
              onRetry={() => void detail.refetch()}
            />
          </div>
        </div>
      )}
    </PageSearchShell>
  )
}
