import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Pagination,
  PaginationButton,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { getSessionMessages, listAllProfileSessions } from '@/hermes'
import { useIsMobile } from '@/hooks/use-mobile'
import { type Translations, useI18n } from '@/i18n'
import { ExternalLink, ExternalLinkIcon, hostPathLabel, urlSlugTitleLabel, useLinkTitle } from '@/lib/external-link'
import { AlertTriangle, Eye, FileImage, FolderOpen, Loader2, RefreshCw } from '@/lib/icons'
import { downloadGatewayMediaFile, isRemoteGateway } from '@/lib/media'
import { normalize } from '@/lib/text'
import { fmtDayTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { ensureGatewayProfile } from '@/store/profile'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PageSearchShell } from '../page-search-shell'
import { sessionRoute } from '../routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { ArtifactPreviewThumbnail } from './artifact-preview'
import {
  ARTIFACT_FILTERS,
  type ArtifactFilter,
  artifactImageSrc,
  type ArtifactRecord,
  collectArtifactsForSession
} from './artifact-utils'

const SESSION_BATCH_SIZE = 30

function formatArtifactTime(timestamp: number): string {
  return fmtDayTime.format(new Date(timestamp))
}

function artifactDisplayValue(value: string): string {
  if (!/^data:image\//i.test(value)) {
    return value
  }

  const mime = /^data:([^;,]+)/i.exec(value)?.[1] || 'image'

  return `data:${mime};…`
}

function pageRangeLabel(total: number, page: number, pageSize: number, a: Translations['artifacts']): string {
  if (total === 0) {
    return a.zero
  }

  const start = (page - 1) * pageSize + 1
  const end = Math.min(total, page * pageSize)

  return a.rangeOf(start, end, total)
}

function paginationItems(page: number, pageCount: number): Array<number | 'ellipsis'> {
  if (pageCount <= 7) {
    return Array.from({ length: pageCount }, (_, index) => index + 1)
  }

  const pages: Array<number | 'ellipsis'> = [1]
  const start = Math.max(2, page - 1)
  const end = Math.min(pageCount - 1, page + 1)

  if (start > 2) {
    pages.push('ellipsis')
  }

  for (let nextPage = start; nextPage <= end; nextPage += 1) {
    pages.push(nextPage)
  }

  if (end < pageCount - 1) {
    pages.push('ellipsis')
  }

  pages.push(pageCount)

  return pages
}

type CellCtx = {
  onOpen: (artifact: ArtifactRecord) => void | Promise<void>
  onOpenChat: (artifact: ArtifactRecord) => void | Promise<void>
}

interface ArtifactColumn {
  Cell: (props: { artifact: ArtifactRecord; ctx: CellCtx }) => React.ReactElement
  bodyClassName: string
  header: (filter: ArtifactFilter, a: Translations['artifacts']) => string
  id: 'location' | 'primary' | 'session'
  width: (filter: ArtifactFilter) => string
}

const itemsLabel = (f: ArtifactFilter, a: Translations['artifacts']) =>
  f === 'link' ? a.itemsLink : f === 'file' ? a.itemsFile : a.itemsGeneric

interface ArtifactsViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function ArtifactsView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: ArtifactsViewProps) {
  const { t } = useI18n()
  const a = t.artifacts
  const navigate = useNavigate()
  const isMobile = useIsMobile()
  const [artifacts, setArtifacts] = useState<ArtifactRecord[] | null>(null)
  const [query, setQuery] = useState('')
  const [fatalLoadError, setFatalLoadError] = useState(false)
  const [partialLoad, setPartialLoad] = useState<{ failed: number; total: number } | null>(null)
  const [sessionLimit, setSessionLimit] = useState(SESSION_BATCH_SIZE)
  const [sessionScope, setSessionScope] = useState<{ loaded: number; total: number } | null>(null)
  const hasLoadedOnceRef = useRef(false)

  const [kindFilter, setKindFilter] = useRouteEnumParam('tab', ARTIFACT_FILTERS, 'all')

  const [failedImageIds, setFailedImageIds] = useState<Set<string>>(() => new Set())
  const [imagePage, setImagePage] = useState(1)
  const [filePage, setFilePage] = useState(1)

  const [refreshing, setRefreshing] = useState(false)

  const refreshArtifacts = useCallback(async () => {
    setRefreshing(true)
    setFatalLoadError(false)
    setPartialLoad(null)

    try {
      const sessionPage = await listAllProfileSessions(sessionLimit, 1)
      const sessions = sessionPage.sessions
      const results = await Promise.allSettled(sessions.map(session => getSessionMessages(session.id, session.profile)))
      const nextArtifacts: ArtifactRecord[] = []
      const failed = results.filter(result => result.status === 'rejected').length

      results.forEach((result, index) => {
        if (result.status !== 'fulfilled') {
          return
        }

        const session = sessions[index]
        nextArtifacts.push(...collectArtifactsForSession(session, result.value.messages))
      })

      setArtifacts(nextArtifacts.sort((left, right) => right.timestamp - left.timestamp))
      setPartialLoad(failed > 0 ? { failed, total: sessions.length } : null)
      setSessionScope({
        loaded: sessions.length,
        total: Math.max(sessionPage.total ?? sessions.length, sessions.length)
      })
      hasLoadedOnceRef.current = true
    } catch (err) {
      notifyError(err, a.failedLoad)

      if (!hasLoadedOnceRef.current) {
        setFatalLoadError(true)
        setArtifacts([])
      }
    } finally {
      setRefreshing(false)
    }
  }, [a, sessionLimit])

  useRefreshHotkey(refreshArtifacts)

  useEffect(() => {
    void refreshArtifacts()
  }, [refreshArtifacts])

  useEffect(() => {
    setImagePage(1)
    setFilePage(1)
  }, [artifacts, kindFilter, query])

  const visibleArtifacts = useMemo(() => {
    if (!artifacts) {
      return []
    }

    const q = normalize(query)

    return artifacts.filter(artifact => {
      if (kindFilter !== 'all' && artifact.kind !== kindFilter) {
        return false
      }

      if (!q) {
        return true
      }

      return (
        artifact.label.toLowerCase().includes(q) ||
        (!/^data:image\//i.test(artifact.value) && artifact.value.toLowerCase().includes(q)) ||
        artifact.sessionTitle.toLowerCase().includes(q)
      )
    })
  }, [artifacts, kindFilter, query])

  const visibleImageArtifacts = useMemo(
    () => visibleArtifacts.filter(artifact => artifact.kind === 'image'),
    [visibleArtifacts]
  )

  const visibleFileArtifacts = useMemo(
    () => visibleArtifacts.filter(artifact => artifact.kind !== 'image'),
    [visibleArtifacts]
  )

  const imagePageCount = Math.max(1, Math.ceil(visibleImageArtifacts.length / 24))
  const filePageCount = Math.max(1, Math.ceil(visibleFileArtifacts.length / 100))
  const currentImagePage = Math.min(imagePage, imagePageCount)
  const currentFilePage = Math.min(filePage, filePageCount)

  const pagedImageArtifacts = useMemo(
    () => visibleImageArtifacts.slice((currentImagePage - 1) * 24, currentImagePage * 24),
    [currentImagePage, visibleImageArtifacts]
  )

  const pagedFileArtifacts = useMemo(
    () => visibleFileArtifacts.slice((currentFilePage - 1) * 100, currentFilePage * 100),
    [currentFilePage, visibleFileArtifacts]
  )

  // Rotating placeholder nudges from real data — search matches file paths and
  // session titles, not just labels; show it.
  const searchHints = useMemo(() => {
    if (!artifacts?.length) {
      return undefined
    }

    const extensions = [
      ...new Set(artifacts.map(artifact => /\.(\w{2,4})$/.exec(artifact.value)?.[1]?.toLowerCase()).filter(Boolean))
    ].slice(0, 3) as string[]

    const titles = [...new Set(artifacts.map(artifact => artifact.sessionTitle).filter(Boolean))].slice(0, 2)

    const hints = [
      ...extensions.map(ext => t.common.tryHint(`.${ext}`)),
      ...titles.map(title => t.common.tryHint(title))
    ]

    return hints.length > 0 ? hints : undefined
  }, [artifacts, t])

  const counts = useMemo(() => {
    const all = artifacts || []

    return {
      all: all.length,
      image: all.filter(artifact => artifact.kind === 'image').length,
      file: all.filter(artifact => artifact.kind === 'file').length,
      link: all.filter(artifact => artifact.kind === 'link').length
    }
  }, [artifacts])

  const openArtifact = useCallback(
    async (artifact: ArtifactRecord) => {
      try {
        if (!artifact.href) {
          return
        }

        // Gateway-local files must stay on the authenticated filesystem bridge.
        // Never externalize mediaExternalUrl() download URLs because token-auth
        // connections encode their bearer credential in the query string.
        if (isRemoteGateway() && artifact.kind !== 'link' && !/^https?:/i.test(artifact.value)) {
          await downloadGatewayMediaFile(artifact.value)

          return
        }

        if (window.hermesDesktop?.openExternal) {
          await window.hermesDesktop.openExternal(artifact.href)
        } else {
          window.open(artifact.href, '_blank', 'noopener,noreferrer')
        }
      } catch (err) {
        notifyError(err, a.openFailed)
      }
    },
    [a]
  )

  const markImageFailed = useCallback((id: string) => {
    setFailedImageIds(current => {
      if (current.has(id)) {
        return current
      }

      return new Set(current).add(id)
    })
  }, [])

  const openSourceChat = useCallback(
    async (artifact: ArtifactRecord) => {
      try {
        await ensureGatewayProfile(artifact.profile)
        navigate(sessionRoute(artifact.sessionId))
      } catch (err) {
        notifyError(err, a.openChatFailed)
      }
    },
    [a.openChatFailed, navigate]
  )

  const cellCtx: CellCtx = {
    onOpen: openArtifact,
    onOpenChat: openSourceChat
  }

  return (
    <PageSearchShell
      {...props}
      activeTab={kindFilter}
      onSearchChange={setQuery}
      onTabChange={id => setKindFilter(id as typeof kindFilter)}
      searchHidden={counts.all === 0}
      searchHints={searchHints}
      searchPlaceholder={a.search}
      searchTrailingAction={
        <Tip label={refreshing ? a.refreshing : a.refresh}>
          <Button
            aria-label={refreshing ? a.refreshing : a.refresh}
            className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            disabled={refreshing}
            onClick={() => void refreshArtifacts()}
            size="icon-titlebar"
            variant="ghost"
          >
            {refreshing ? <Loader2 className="animate-spin" /> : <RefreshCw />}
          </Button>
        </Tip>
      }
      searchValue={query}
      tabs={[
        { id: 'all', label: a.tabAll, meta: artifacts ? counts.all : null },
        { id: 'image', label: a.tabImages, meta: artifacts ? counts.image : null },
        { id: 'file', label: a.tabFiles, meta: artifacts ? counts.file : null },
        { id: 'link', label: a.tabLinks, meta: artifacts ? counts.link : null }
      ]}
    >
      {!artifacts ? (
        <PageLoader label={a.indexing} />
      ) : fatalLoadError ? (
        <div className="grid h-full place-items-center px-6 text-center" role="alert">
          <div className="max-w-sm">
            <AlertTriangle aria-hidden="true" className="mx-auto mb-2 size-5 text-amber-500" />
            <div className="text-sm font-medium">{a.failedLoadTitle}</div>
            <div className="mt-1 text-xs text-muted-foreground">{a.failedLoadDesc}</div>
            <Button className="mt-3" onClick={() => void refreshArtifacts()} size="sm" type="button" variant="outline">
              {a.tryAgain}
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex h-full min-h-0 flex-col">
          {partialLoad && (
            <div
              className="mx-3 mb-2 flex shrink-0 items-start gap-2 rounded-md border border-amber-500/25 bg-amber-500/8 px-3 py-2"
              role="alert"
            >
              <AlertTriangle aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-amber-500" />
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium">{a.partialLoadTitle}</div>
                <div className="mt-0.5 text-[0.6875rem] text-muted-foreground">
                  {a.partialLoadDesc(partialLoad.failed, partialLoad.total)}
                </div>
              </div>
              <Button onClick={() => void refreshArtifacts()} size="xs" type="button" variant="textStrong">
                {a.tryAgain}
              </Button>
            </div>
          )}
          {sessionScope && sessionScope.loaded < sessionScope.total && (
            <div className="mx-3 mb-2 flex shrink-0 items-center justify-between gap-3 px-1 text-[0.6875rem] text-muted-foreground">
              <span>{a.scopeSummary(sessionScope.loaded, sessionScope.total)}</span>
              <Button
                aria-label={a.loadMoreChats(Math.min(SESSION_BATCH_SIZE, sessionScope.total - sessionScope.loaded))}
                disabled={refreshing}
                onClick={() =>
                  setSessionLimit(current =>
                    Math.min(
                      sessionScope.total,
                      current + Math.min(SESSION_BATCH_SIZE, sessionScope.total - sessionScope.loaded)
                    )
                  )
                }
                size="xs"
                type="button"
                variant="textStrong"
              >
                {a.loadMoreChats(Math.min(SESSION_BATCH_SIZE, sessionScope.total - sessionScope.loaded))}
              </Button>
            </div>
          )}
          {visibleArtifacts.length === 0 ? (
            <div className="grid min-h-0 flex-1 place-items-center px-6 text-center">
              <div>
                <div className="text-sm font-medium">{a.noArtifactsTitle}</div>
                <div className="mt-1 text-xs text-muted-foreground">{a.noArtifactsDesc}</div>
              </div>
            </div>
          ) : (
            <div className="min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable]">
              <div className="flex flex-col gap-3 px-3 pb-2">
                {visibleImageArtifacts.length > 0 && (
                  <section className="flex flex-col">
                    <div className="sticky top-0 z-10 -mx-3 flex h-7 items-center gap-3 overflow-x-auto bg-background px-3">
                      <ArtifactsPagination
                        className="ml-auto justify-end px-0"
                        itemLabel={a.itemsImage}
                        onPageChange={setImagePage}
                        page={currentImagePage}
                        pageSize={24}
                        total={visibleImageArtifacts.length}
                      />
                    </div>
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] items-start gap-2 pt-1.5">
                      {pagedImageArtifacts.map(artifact => (
                        <ArtifactImageCard
                          artifact={artifact}
                          failedImage={failedImageIds.has(artifact.id)}
                          key={artifact.id}
                          onImageError={markImageFailed}
                          onOpen={openArtifact}
                          onOpenChat={openSourceChat}
                        />
                      ))}
                    </div>
                  </section>
                )}

                {visibleFileArtifacts.length > 0 && (
                  <section className="flex flex-col">
                    <div className="sticky top-0 z-10 -mx-3 flex h-7 items-center gap-3 overflow-x-auto bg-background px-3">
                      <ArtifactsPagination
                        className="ml-auto justify-end px-0"
                        itemLabel={itemsLabel(kindFilter, a)}
                        onPageChange={setFilePage}
                        page={currentFilePage}
                        pageSize={100}
                        total={visibleFileArtifacts.length}
                      />
                    </div>
                    {isMobile ? (
                      <ArtifactMobileList artifacts={pagedFileArtifacts} ctx={cellCtx} />
                    ) : (
                      <div className="overflow-x-auto rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)">
                        <ArtifactTable artifacts={pagedFileArtifacts} ctx={cellCtx} filter={kindFilter} />
                      </div>
                    )}
                  </section>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </PageSearchShell>
  )
}

interface ArtifactsPaginationProps {
  className?: string
  itemLabel: string
  onPageChange: (page: number) => void
  page: number
  pageSize: number
  total: number
}

function ArtifactsPagination({ className, itemLabel, onPageChange, page, pageSize, total }: ArtifactsPaginationProps) {
  const { t } = useI18n()
  const a = t.artifacts
  const pageCount = Math.max(1, Math.ceil(total / pageSize))

  return (
    <div className={cn('flex h-6 items-center justify-between gap-2 px-1', className)}>
      <div className="shrink-0 text-[0.62rem] text-muted-foreground">
        {pageRangeLabel(total, page, pageSize, a)} {itemLabel}
      </div>
      {pageCount > 1 && (
        <Pagination className="mx-0 w-auto min-w-0 justify-end">
          <PaginationContent className="gap-0.5">
            <PaginationItem>
              <PaginationPrevious disabled={page <= 1} onClick={() => onPageChange(Math.max(1, page - 1))} />
            </PaginationItem>
            {paginationItems(page, pageCount).map((item, index) => (
              <PaginationItem key={`${item}-${index}`}>
                {item === 'ellipsis' ? (
                  <PaginationEllipsis />
                ) : (
                  <PaginationButton
                    aria-label={a.goToPage(itemLabel, item)}
                    isActive={page === item}
                    onClick={() => onPageChange(item)}
                  >
                    {item}
                  </PaginationButton>
                )}
              </PaginationItem>
            ))}
            <PaginationItem>
              <PaginationNext
                disabled={page >= pageCount}
                onClick={() => onPageChange(Math.min(pageCount, page + 1))}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      )}
    </div>
  )
}

interface ArtifactImageCardProps {
  artifact: ArtifactRecord
  failedImage: boolean
  onImageError: (id: string) => void
  onOpen: (artifact: ArtifactRecord) => void | Promise<void>
  onOpenChat: (artifact: ArtifactRecord) => void | Promise<void>
}

function ArtifactImageCard({ artifact, failedImage, onImageError, onOpen, onOpenChat }: ArtifactImageCardProps) {
  const { t } = useI18n()
  const a = t.artifacts
  const kindLabel = artifact.kind === 'image' ? a.kindImage : artifact.kind === 'file' ? a.kindFile : a.kindLink

  const displayLabel =
    /^data:image\//i.test(artifact.value) && artifact.label === 'data:image' ? a.embeddedImage : artifact.label

  const displayValue = artifactDisplayValue(artifact.value)
  const [src, setSrc] = useState('')
  const [previewResolved, setPreviewResolved] = useState(false)

  useEffect(() => {
    let active = true

    setSrc('')
    setPreviewResolved(false)
    void artifactImageSrc(artifact.value, artifact.href, artifact.previewable, artifact.profile)
      .then(nextSrc => {
        if (active) {
          setSrc(nextSrc)
          setPreviewResolved(true)
        }
      })
      .catch(() => {
        if (active) {
          setPreviewResolved(true)
          onImageError(artifact.id)
        }
      })

    return () => {
      active = false
    }
  }, [artifact.href, artifact.id, artifact.previewable, artifact.profile, artifact.value, onImageError])

  return (
    <article className="group/artifact overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)">
      <div
        className={cn(
          'relative flex h-40 w-full items-center justify-center overflow-hidden border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-1.5',
          failedImage && 'cursor-default'
        )}
      >
        {(failedImage || !src) && (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1.5">
            <ArtifactPreviewThumbnail
              ariaLabel={a.previewArtifact(displayLabel)}
              artifact={artifact}
              className="h-full w-full rounded-none border-0"
            />
            {(failedImage || (previewResolved && !src)) && (
              <span className="absolute bottom-2 text-[0.625rem] text-muted-foreground">{a.previewUnavailable}</span>
            )}
          </div>
        )}
        {!failedImage && src && (
          <ZoomableImage
            alt={displayLabel}
            className="max-h-40 max-w-full cursor-zoom-in rounded-md object-contain"
            containerClassName="max-h-full"
            decoding="async"
            loading="lazy"
            onError={() => onImageError(artifact.id)}
            slot="artifact-media"
            src={src}
          />
        )}
      </div>

      <div className="space-y-1.5 p-2">
        <div className="min-w-0">
          <div className="mb-0.5 flex items-center gap-1 text-[0.625rem] uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
            <FileImage className="size-3" />
            {kindLabel}
          </div>
          <div className="truncate text-[length:var(--conversation-caption-font-size)] font-medium">{displayLabel}</div>
          <div className="mt-0.5 truncate text-[0.625rem] text-(--ui-text-tertiary)">{displayValue}</div>
        </div>

        <div className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
          {artifact.sessionTitle}
          {artifact.profile !== 'default' && ` · ${artifact.profile}`} · {formatArtifactTime(artifact.timestamp)}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {artifact.href && (
            <Button
              aria-label={a.openArtifact(displayLabel)}
              onClick={() => void onOpen(artifact)}
              size="xs"
              type="button"
              variant="textStrong"
            >
              <Eye className="size-3" />
              {a.open}
            </Button>
          )}
          <Button
            aria-label={a.openSourceChat(displayLabel)}
            onClick={() => void onOpenChat(artifact)}
            size="xs"
            type="button"
            variant="textStrong"
          >
            <FolderOpen className="size-3" />
            {a.chat}
          </Button>
        </div>
      </div>
    </article>
  )
}

function ArtifactMobileList({ artifacts, ctx }: { artifacts: ArtifactRecord[]; ctx: CellCtx }) {
  return (
    <div className="space-y-2" data-testid="artifact-mobile-list">
      {artifacts.map(artifact => (
        <ArtifactMobileCard artifact={artifact} ctx={ctx} key={artifact.id} />
      ))}
    </div>
  )
}

function ArtifactMobileCard({ artifact, ctx }: { artifact: ArtifactRecord; ctx: CellCtx }) {
  const { t } = useI18n()
  const a = t.artifacts
  const isLink = artifact.kind === 'link' && Boolean(artifact.href)
  const fetchedTitle = useLinkTitle(isLink ? artifact.href : null)
  const label = isLink ? fetchedTitle || urlSlugTitleLabel(artifact.href) : artifact.label
  const location = isLink ? hostPathLabel(artifact.value) : artifactDisplayValue(artifact.value)
  const copyLabel = isLink ? a.copyUrl : a.copyPath
  const kindLabel = artifact.kind === 'link' ? a.kindLink : a.kindFile

  return (
    <article className="overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)">
      <div className="flex min-w-0 items-start gap-2 p-2.5">
        <ArtifactPreviewThumbnail
          ariaLabel={a.previewArtifact(label)}
          artifact={artifact}
          className="h-14 w-16 shrink-0"
        />
        <div className="min-w-0 flex-1">
          <div className="text-[0.625rem] uppercase tracking-[0.08em] text-(--ui-text-tertiary)">{kindLabel}</div>
          <div className="mt-0.5 truncate text-xs font-medium">{label}</div>
          <div className="group/mobile-location mt-1 flex min-w-0 items-center gap-1.5">
            <Tip label={artifact.value}>
              <div
                className={cn(
                  'min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-tertiary)',
                  !isLink && 'font-mono'
                )}
              >
                {location}
              </div>
            </Tip>
            <CopyButton
              appearance="icon"
              buttonSize="icon-xs"
              className="shrink-0 text-muted-foreground hover:text-foreground"
              iconClassName="size-3.5"
              label={copyLabel}
              text={artifact.value}
              title={copyLabel}
            />
          </div>
          <div className="mt-1 truncate text-[0.6875rem] text-(--ui-text-tertiary)">
            {artifact.sessionTitle}
            {artifact.profile !== 'default' && ` · ${artifact.profile}`} · {formatArtifactTime(artifact.timestamp)}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2 border-t border-(--ui-stroke-tertiary) px-2.5 py-1.5">
        {artifact.href && (
          <Button
            aria-label={a.openArtifact(label)}
            className="min-h-8"
            onClick={() => void ctx.onOpen(artifact)}
            size="sm"
            type="button"
            variant="textStrong"
          >
            <Eye className="size-3.5" />
            {a.open}
          </Button>
        )}
        <Button
          aria-label={a.openSourceChat(label)}
          className="min-h-8"
          onClick={() => void ctx.onOpenChat(artifact)}
          size="sm"
          type="button"
          variant="textStrong"
        >
          <FolderOpen className="size-3.5" />
          {a.chat}
        </Button>
      </div>
    </article>
  )
}

// Single click target for any row cell. External URLs render as <ExternalLink>;
// local actions render as <button>. Padding lives here, NOT on the <td>, so
// the entire cell area is hoverable and clickable in both branches.
function ArtifactCellAction({
  ariaLabel,
  children,
  href,
  onClick,
  title
}: {
  ariaLabel?: string
  children: React.ReactNode
  href?: string
  onClick?: () => void
  title?: string
}) {
  if (href) {
    return (
      <ExternalLink
        aria-label={ariaLabel}
        className="flex h-full w-full min-w-0 items-center gap-2 px-2.5 py-1.5 text-left text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) font-normal text-(--ui-text-secondary) no-underline underline-offset-4 decoration-current/20 transition-colors hover:text-foreground hover:underline"
        href={href}
        showExternalIcon={false}
        title={title}
      >
        {children}
      </ExternalLink>
    )
  }

  if (!onClick) {
    return (
      <div className="flex h-full w-full min-w-0 items-center gap-2 px-2.5 py-1.5 text-left text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) font-normal text-(--ui-text-secondary)">
        {children}
      </div>
    )
  }

  return (
    <RowButton
      aria-label={ariaLabel}
      className="flex h-full w-full min-w-0 items-center gap-2 px-2.5 py-1.5 text-left text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) font-normal text-(--ui-text-secondary) no-underline underline-offset-4 decoration-current/20 transition-colors hover:text-foreground hover:underline"
      onClick={onClick}
    >
      {children}
    </RowButton>
  )
}

function PrimaryCell({ artifact, ctx }: { artifact: ArtifactRecord; ctx: CellCtx }) {
  const { t } = useI18n()
  const isLink = artifact.kind === 'link' && Boolean(artifact.href)
  const canOpen = Boolean(artifact.href)
  const fetchedTitle = useLinkTitle(isLink ? artifact.href : null)
  const label = isLink ? fetchedTitle || urlSlugTitleLabel(artifact.href) : artifact.label

  return (
    <ArtifactCellAction
      ariaLabel={canOpen ? t.artifacts.openArtifact(label) : undefined}
      href={isLink ? artifact.href : undefined}
      onClick={isLink ? undefined : artifact.href ? () => void ctx.onOpen(artifact) : undefined}
      title={label}
    >
      <ArtifactPreviewThumbnail ariaLabel={t.artifacts.previewArtifact(artifact.label)} artifact={artifact} />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5 self-center">
        <span className={cn('min-w-0', isLink ? 'wrap-anywhere' : 'truncate')}>
          {label}
          {isLink && <ExternalLinkIcon />}
        </span>
        {canOpen && (
          <span className="flex items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
            <Eye className="size-3" />
            {t.artifacts.open}
          </span>
        )}
      </span>
    </ArtifactCellAction>
  )
}

function LocationCell({ artifact }: { artifact: ArtifactRecord; ctx: CellCtx }) {
  const { t } = useI18n()
  const isLink = artifact.kind === 'link'
  const value = isLink ? hostPathLabel(artifact.value) : artifact.value
  const copyLabel = isLink ? t.artifacts.copyUrl : t.artifacts.copyPath

  return (
    <div className="group/location flex min-w-0 items-center gap-1.5">
      <Tip label={artifact.value}>
        <div
          className={cn(
            'min-w-0 flex-1 truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)',
            isLink ? 'font-normal' : 'font-mono'
          )}
        >
          {value}
        </div>
      </Tip>
      <CopyButton
        appearance="icon"
        buttonSize="icon-xs"
        className="shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover/location:opacity-100"
        iconClassName="size-3.5"
        label={copyLabel}
        text={artifact.value}
        title={copyLabel}
      />
    </div>
  )
}

function SessionCell({ artifact, ctx }: { artifact: ArtifactRecord; ctx: CellCtx }) {
  const { t } = useI18n()

  return (
    <ArtifactCellAction
      ariaLabel={t.artifacts.openSourceChat(artifact.label)}
      onClick={() => void ctx.onOpenChat(artifact)}
      title={artifact.sessionTitle}
    >
      <span className="flex min-w-0 flex-col">
        <span className="truncate">{artifact.sessionTitle}</span>
        <span className="truncate text-[0.6875rem] font-normal text-(--ui-text-tertiary)">
          {formatArtifactTime(artifact.timestamp)}
          {artifact.profile !== 'default' && ` · ${artifact.profile}`}
        </span>
      </span>
    </ArtifactCellAction>
  )
}

const ARTIFACT_COLUMNS: readonly ArtifactColumn[] = [
  {
    Cell: PrimaryCell,
    bodyClassName: 'p-0',
    header: (filter, a) =>
      filter === 'link' ? a.colTitleLink : filter === 'file' ? a.colTitleFile : a.colTitleDefault,
    id: 'primary',
    width: filter => (filter === 'link' ? 'w-[50%]' : 'w-[40%]')
  },
  {
    Cell: LocationCell,
    bodyClassName: 'px-2.5 py-1.5',
    header: (filter, a) =>
      filter === 'link' ? a.colLocationLink : filter === 'file' ? a.colLocationFile : a.colLocationDefault,
    id: 'location',
    width: filter => (filter === 'link' ? 'w-[30%]' : 'w-[36%]')
  },
  {
    Cell: SessionCell,
    bodyClassName: 'p-0',
    header: (_filter, a) => a.colSession,
    id: 'session',
    width: filter => (filter === 'link' ? 'w-[20%]' : 'w-[24%]')
  }
]

function ArtifactTable({
  artifacts,
  ctx,
  filter
}: {
  artifacts: readonly ArtifactRecord[]
  ctx: CellCtx
  filter: ArtifactFilter
}) {
  const { t } = useI18n()

  return (
    <table className="w-full min-w-176 table-fixed text-left text-[length:var(--conversation-caption-font-size)]">
      <thead className="border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) text-[0.625rem] uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
        <tr>
          {ARTIFACT_COLUMNS.map(col => (
            <th className={cn(col.width(filter), 'px-2.5 py-1.5 font-medium')} key={col.id}>
              {col.header(filter, t.artifacts)}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {artifacts.map(artifact => (
          <tr className="group/artifact border-b border-(--ui-stroke-tertiary) last:border-b-0" key={artifact.id}>
            {ARTIFACT_COLUMNS.map(col => {
              const Cell = col.Cell

              return (
                <td className={cn('align-middle', col.bodyClassName)} key={col.id}>
                  <Cell artifact={artifact} ctx={ctx} />
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
