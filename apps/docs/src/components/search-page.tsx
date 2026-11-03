'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import config from '../../public/search-config.json';
import {
  algoliaSearch,
  resultBreadcrumb,
  resultTitle,
  transformResultUrl,
  type SearchHit,
} from '@/lib/search';
import { splitLocalePath, type Locale } from '@/lib/urls';

interface SearchCopy {
  label: string;
  placeholder: string;
  submit: string;
  idle: string;
  loading: string;
  error: string;
  retry: string;
  empty: (query: string) => string;
  resultsFor: (count: number) => string;
  more: string;
}

const COPY: Record<Locale, SearchCopy> = {
  en: {
    label: 'Search documentation',
    placeholder: 'Search documentation',
    submit: 'Search',
    idle: 'Type a query to search the documentation.',
    loading: 'Searching…',
    error: 'Search failed — check your connection and try again.',
    retry: 'Try again',
    empty: (query) => `No results for “${query}”.`,
    resultsFor: (count) => `${count.toLocaleString('en')} results`,
    more: 'Show more results',
  },
  'zh-Hans': {
    label: '搜索文档',
    placeholder: '搜索文档',
    submit: '搜索',
    idle: '输入关键词以搜索文档。',
    loading: '正在搜索…',
    error: '搜索失败 — 请检查网络连接后重试。',
    retry: '重试',
    empty: (query) => `没有找到与“${query}”相关的结果。`,
    resultsFor: (count) => `${count.toLocaleString('zh-Hans')} 条结果`,
    more: '显示更多结果',
  },
};

type Status = 'idle' | 'loading' | 'error' | 'done';

/** Where a hit lives in the docs, for a muted URL line under each result. */
function resultPath(url: string): string {
  return splitLocalePath(url).path;
}

export function SearchPage({ locale }: { locale: Locale }) {
  const copy = COPY[locale];
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const qParam = searchParams?.get('q') ?? '';

  const [inputValue, setInputValue] = useState(qParam);
  const [status, setStatus] = useState<Status>(qParam ? 'loading' : 'idle');
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [nbHits, setNbHits] = useState(0);
  const [page, setPage] = useState(0);
  const [nbPages, setNbPages] = useState(1);
  const [appending, setAppending] = useState(false);
  const [appendError, setAppendError] = useState(false);
  const pageRequest = useRef<AbortController | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  // The query we last wrote to the URL ourselves, so keystroke-driven updates
  // never clobber the input the visitor is typing into.
  const pushedRef = useRef<string | null>(null);

  // Sync the input when the URL changes outside our own replaces
  // (back/forward, direct links, the locale switcher).
  useEffect(() => {
    if (qParam !== pushedRef.current) setInputValue(qParam);
  }, [qParam]);

  // Reflect typing into the URL, debounced, so reload, history and shared
  // links all carry the query.
  useEffect(() => {
    const trimmed = inputValue.trim();
    if (trimmed === qParam || !pathname) return;
    const timer = setTimeout(() => {
      pushedRef.current = trimmed;
      router.replace(trimmed ? `${pathname}?q=${encodeURIComponent(trimmed)}` : pathname);
    }, 250);
    return () => clearTimeout(timer);
  }, [inputValue, qParam, pathname, router]);

  // Fetch whenever the URL query, the locale or a retry changes.
  useEffect(() => {
    pageRequest.current?.abort();
    pageRequest.current = null;
    setAppending(false);
    setAppendError(false);
    const query = qParam.trim();
    if (!query) {
      setHits([]);
      setNbHits(0);
      setPage(0);
      setNbPages(1);
      setStatus('idle');
      return;
    }
    const controller = new AbortController();
    setStatus('loading');
    algoliaSearch(config, query, locale, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) return;
        setHits(result.hits);
        setNbHits(result.nbHits);
        setPage(result.page);
        setNbPages(result.nbPages);
        setStatus('done');
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setStatus('error');
      });
    return () => {
      controller.abort();
      pageRequest.current?.abort();
      pageRequest.current = null;
    };
  }, [qParam, locale, retryToken]);

  const loadMore = useCallback(async () => {
    const query = qParam.trim();
    const nextPage = page + 1;
    if (!query || pageRequest.current || status !== 'done' || nextPage >= nbPages) return;
    const controller = new AbortController();
    pageRequest.current = controller;
    setAppending(true);
    setAppendError(false);
    try {
      const result = await algoliaSearch(config, query, locale, { page: nextPage, signal: controller.signal });
      if (controller.signal.aborted) return;
      setHits((current) => [...current, ...result.hits]);
      setPage(result.page);
      setNbPages(result.nbPages);
      setNbHits(result.nbHits);
    } catch {
      if (!controller.signal.aborted) setAppendError(true);
    } finally {
      if (pageRequest.current === controller) {
        pageRequest.current = null;
        setAppending(false);
      }
    }
  }, [locale, nbPages, page, qParam, status]);

  const submit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = inputValue.trim();
    if (trimmed === qParam || !pathname) return;
    pushedRef.current = trimmed;
    router.replace(trimmed ? `${pathname}?q=${encodeURIComponent(trimmed)}` : pathname);
  };

  return (
    <div className="mt-6">
      <form onSubmit={submit} role="search">
        <div className="flex flex-col gap-2 sm:flex-row">
          <label htmlFor="docs-search-input" className="sr-only">
            {copy.label}
          </label>
          <input
            id="docs-search-input"
            name="q"
            type="search"
            autoComplete="off"
            value={inputValue}
            placeholder={copy.placeholder}
            onChange={(event) => setInputValue(event.target.value)}
            className="h-11 w-full rounded-lg border border-fd-border bg-fd-card px-4 text-base text-fd-foreground outline-none placeholder:text-fd-muted-foreground focus-visible:ring-2 focus-visible:ring-fd-ring"
          />
          <button
            type="submit"
            className={buttonVariants({ variant: 'primary', className: 'h-11 shrink-0 px-5 hover:transition-none' })}
          >
            {copy.submit}
          </button>
        </div>
      </form>

      <div role="status" aria-live="polite" className="mt-5 text-sm text-fd-muted-foreground">
        {status === 'idle' && <p>{copy.idle}</p>}
        {status === 'loading' && <p>{copy.loading}</p>}
        {status === 'error' && (
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-fd-error">{copy.error}</p>
            <button
              type="button"
              onClick={() => setRetryToken((token) => token + 1)}
              className={buttonVariants({ variant: 'outline', className: 'px-3 py-1.5 hover:transition-none' })}
            >
              {copy.retry}
            </button>
          </div>
        )}
        {status === 'done' && hits.length === 0 && <p>{copy.empty(qParam.trim())}</p>}
        {status === 'done' && hits.length > 0 && <p>{copy.resultsFor(nbHits)}</p>}
      </div>

      {hits.length > 0 && (
        <ul className={`mt-4 space-y-2 ${status === 'loading' ? 'opacity-60' : ''}`}>
          {hits.map((hit) => {
            const href = transformResultUrl(hit.url);
            if (href === null) return null;
            const title = resultTitle(hit);
            const breadcrumb = resultBreadcrumb(hit);
            return (
              <li key={hit.objectID ?? hit.url}>
                <a
                  href={href}
                  className="block rounded-lg border border-fd-border p-4 transition-colors hover:bg-fd-accent hover:transition-none"
                >
                  {breadcrumb.length > 0 && (
                    <span className="block text-xs text-fd-muted-foreground">{breadcrumb.join(' › ')}</span>
                  )}
                  <span className="block font-medium text-fd-foreground">
                    {title.map((segment, index) =>
                      segment.marked ? (
                        <mark key={index} className="bg-transparent font-semibold text-fd-primary">
                          {segment.text}
                        </mark>
                      ) : (
                        <span key={index}>{segment.text}</span>
                      ),
                    )}
                  </span>
                  <span className="mt-1 block truncate text-sm text-fd-muted-foreground">{resultPath(href)}</span>
                </a>
              </li>
            );
          })}
        </ul>
      )}

      {appendError && <p role="alert" className="mt-3 text-sm text-fd-muted-foreground">{copy.error}</p>}
      {status === 'done' && page + 1 < nbPages && (
        <button
          type="button"
          onClick={loadMore}
          disabled={appending}
          className={buttonVariants({ variant: 'outline', className: 'mt-4 px-4 py-2 hover:transition-none' })}
        >
          {appending ? copy.loading : appendError ? copy.retry : copy.more}
        </button>
      )}
    </div>
  );
}
