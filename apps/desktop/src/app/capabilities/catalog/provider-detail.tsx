import { useQuery } from '@tanstack/react-query'

import { type ProfileScope, profileScopeKey } from '@/api/client'
import { previewProvider } from '@/api/skill-catalog'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { CatalogAction, CatalogSource } from '@/types/skill-catalog'

import { CatalogAlert } from './catalog-alert'
import type { CatalogEntry } from './catalog-data'
import { PROVIDER_PREVIEW_KEY, type useProviderCatalog } from './provider-catalog'

export function ProviderActions({ entry, onAction, disabled }: { entry: CatalogEntry; onAction: (entry: CatalogEntry, action: CatalogAction) => void; disabled: boolean }) {
  const { t } = useI18n()
  const c = t.catalog.provider
  const actions = entry.management?.actions ?? entry.provider!.actions

  return <span className="inline-flex flex-wrap gap-2">{actions.map(action => <Button
    aria-label={action === 'install' ? `${t.catalog.install} ${entry.name}` : c.review(c.actions[action])}
    disabled={disabled || entry.management?.settlement === 'reconciling'} key={action} onClick={() => onAction(entry, action)} size="xs" variant="secondary"
  >{action === 'install' ? t.catalog.install : c.review(c.actions[action])}</Button>)}</span>
}

export function ProviderNotices({ catalog, sources, onRefresh }: { catalog: ReturnType<typeof useProviderCatalog>; sources: CatalogSource[]; onRefresh: () => void }) {
  const { t } = useI18n()

  return <>
    {sources.map(source => source.state !== 'ready' ? <CatalogAlert key={source.id} onRetry={onRefresh} retryLabel={t.catalog.retry} title={source.label}>{source.message || (source.state === 'sign_in_required' ? t.catalog.provider.signIn : t.catalog.provider.unavailable)}</CatalogAlert> : null)}
    {sources.length > 0 && catalog.isFetching && <p role="status">{t.catalog.provider.loading}</p>}
    {sources.length > 0 && (catalog.error || catalog.failures.length > 0) && <CatalogAlert onRetry={() => void catalog.refetch()} retryLabel={t.catalog.retry} title={t.catalog.loadFailed}>{catalog.error?.message ?? catalog.failures.join('\n')}</CatalogAlert>}
    {sources.some(source => source.state === 'ready') && catalog.isSuccess && !catalog.failures.length && !catalog.entries.length && <p role="status">{t.catalog.provider.empty}</p>}
    {catalog.hasNextPage && <Button disabled={catalog.isFetchingNextPage} onClick={() => void catalog.fetchNextPage()} size="sm" variant="text">{t.catalog.provider.more(sources.map(source => source.label).join(', '))}</Button>}
  </>
}

export function ProviderDetail({ entry, profile }: { entry: CatalogEntry; profile: ProfileScope }) {
  const { t } = useI18n()
  const c = t.catalog.provider

  const query = useQuery({
    queryKey: [PROVIDER_PREVIEW_KEY, profileScopeKey(profile), entry.identifier, entry.version],
    queryFn: () => previewProvider(profile, entry.identifier, entry.version || undefined),
    staleTime: 60_000, retry: false
  })

  return <section className="space-y-3">
    <p>{c.status[entry.management?.managed_state ?? entry.provider!.managed_state]}</p>
    {entry.management && <p>{c.managed}</p>}
    {entry.management?.settlement === 'reconciling' && <p role="status">{c.finalizing}</p>}
    {query.isPending && <p role="status">{t.skills.loading}</p>}
    {query.error && <><p role="alert">{query.error.message}</p><Button onClick={() => void query.refetch()} size="sm" variant="secondary">{t.catalog.retry}</Button></>}
    {query.data && <><h4>{c.preview}</h4><pre className="whitespace-pre-wrap break-words text-xs text-(--ui-text-secondary)">{query.data.skill_md}</pre></>}
  </section>
}
