/**
 * BROWSER BAR: back / forward / reload / address / pop-out (or pop-in) for a
 * URL preview.
 *
 * The Browser tab had no way to move: no history, and the only address on
 * screen was a read-only label. Every other embedded browser (VS Code's Simple
 * Browser included) puts those four controls in one row above the page, so this
 * does too — and it sits with the page it drives rather than on the zone strip,
 * which is shared with every other tab.
 *
 * The console / DevTools toggles moved here for the same reason: they act on
 * THIS page, so they belong beside its address, not on the strip where they
 * were ambiguous the moment a second tab opened. They stay `PaneStripGlyph`
 * buttons, so a glyph here and a glyph on the strip are still the same button.
 */

import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { PaneStripGlyph } from '@/components/ui/pane-tab'
import { Tip } from '@/components/ui/tooltip'
import type { PenImportPick } from '@/global'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { ANNOTATE_BLUE } from '@/lib/preview-annotate'
import { cn } from '@/lib/utils'

interface PreviewBrowserBarProps {
  annotateMode?: boolean
  canGoBack: boolean
  canGoForward: boolean
  commentCount?: number
  consoleOpen: boolean
  devToolsOpen: boolean
  /** Import-to-canvas picker state for THIS page; undefined hides the control. */
  importState?: PenImportStripState
  loading: boolean
  onBack: () => void
  onFlushComments?: () => void
  onForward: () => void
  onImport?: (mode: 'page' | 'selection') => void
  onImportHoverPath?: (index: null | number) => void
  onImportSelectPath?: (index: number) => void
  onNavigate: (url: string) => void
  onOpenExternal?: () => void
  onPopIn?: () => void
  onPopOut?: () => void
  onReload: () => void
  onToggleAnnotate?: () => void
  onToggleConsole: () => void
  onToggleDevTools: () => void
  onToggleImport?: () => void
  /** The page's CURRENT address (it moves as the user navigates), not the
   *  target the tab was opened with. */
  url: string
}

/** What the strip shows of an import: nothing, the crosshair, a live pick, or a capture in flight. */
export interface PenImportStripState {
  picking: boolean
  pick: null | PenImportPick
  progress: null | number
}

/**
 * What the user typed, as something a webview can load — or `null` if it isn't
 * loadable. A bare host is the common case in an address bar, and here it is
 * usually a dev server, so loopback gets `http` (nothing is listening on 443
 * and there's no certificate) and anything else gets `https`.
 *
 * The blocklist is deliberately narrow: `javascript:` and `data:` execute in
 * the guest partition rather than navigating it, so an address bar is not the
 * place to reach them. Everything a browser normally loads — including
 * `about:blank` and `file://` — goes through, because this IS a browser and
 * the person typing already owns the machine it runs on.
 */
export function normalizePreviewAddress(value: string): null | string {
  const address = value.trim()

  if (!address) {
    return null
  }

  // `://`, not a leading-scheme regex: `localhost:5173` and `example.com:8080`
  // both match `scheme:` and would be read as an unknown protocol instead of a
  // host and a port. `about:blank` has no `://` either, hence the explicit
  // scheme test before the host guess.
  const scheme = /^(about|blob|chrome|data|devtools|file|ftp|https?|javascript|view-source):/i.exec(address)?.[1]
  const loopback = /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?(?:[/?#]|$)/i.test(address)
  const candidate = scheme ? address : `${loopback ? 'http' : 'https'}://${address}`

  // Script-bearing schemes run INSIDE the page the user is looking at (or mint
  // a document with an attacker-chosen body) — that's injection wearing a
  // navigation's clothes, and no browser's address bar honors it either.
  if (scheme && /^(data|javascript)$/i.test(scheme)) {
    return null
  }

  try {
    // A parse failure is the real filter for junk: `://broken`, `htp:/x`, and
    // half-typed nonsense all land here.
    new URL(candidate)

    return candidate
  } catch {
    return null
  }
}

export function PreviewBrowserBar({
  annotateMode = false,
  canGoBack,
  canGoForward,
  commentCount = 0,
  consoleOpen,
  devToolsOpen,
  importState,
  loading,
  onBack,
  onFlushComments,
  onForward,
  onImport,
  onImportHoverPath,
  onImportSelectPath,
  onNavigate,
  onOpenExternal,
  onPopIn,
  onPopOut,
  onReload,
  onToggleAnnotate,
  onToggleConsole,
  onToggleDevTools,
  onToggleImport,
  url
}: PreviewBrowserBarProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  // Null while the field is idle, so the address tracks navigation on its own;
  // a string once the user takes it over, so typing survives a page load.
  const [draft, setDraft] = useState<null | string>(null)
  // The address we asked for and are still waiting on. Without it, committing
  // dropped the field straight back to `url` — the page you were LEAVING —
  // so every navigation flashed the old address before the new one arrived.
  const [pending, setPending] = useState<null | string>(null)
  // Only while the user is typing: a page that navigates itself is never the
  // user's mistake to flag.
  const invalid = draft !== null && draft.trim().length > 0 && !normalizePreviewAddress(draft)
  const shown = draft ?? pending ?? url

  // The page moved (or a redirect landed somewhere else entirely), so the real
  // address supersedes what we asked for.
  useEffect(() => setPending(null), [url])

  const commit = (value: string) => {
    const address = normalizePreviewAddress(value)

    if (!address) {
      return
    }

    setDraft(null)
    setPending(address)
    onNavigate(address)
  }

  const bar = (
    <div className="flex min-h-(--titlebar-height) shrink-0 items-center gap-1 border-b border-border/60 bg-background px-1.5 py-1">
      <PaneStripGlyph
        disabled={!canGoBack}
        icon={<Codicon name="arrow-left" size="0.8125rem" />}
        label={copy.goBack}
        onSelect={onBack}
      />
      <PaneStripGlyph
        disabled={!canGoForward}
        icon={<Codicon name="arrow-right" size="0.8125rem" />}
        label={copy.goForward}
        onSelect={onForward}
      />
      <PaneStripGlyph
        icon={<Codicon name="refresh" size="0.8125rem" spinning={loading} />}
        label={copy.reload}
        onSelect={onReload}
      />
      {/* The copy control lives INSIDE the field, on its right edge — the
          same pre-faded inline icon code blocks use, not a toolbar button.
          It copies what the field shows: on a remote gateway, that is the
          reach-resolved address. */}
      <div className="relative min-w-0 flex-1">
        {/* Progress lives IN the field, where the address it belongs to is —
            the reload glyph also spins, but it sits in a row of four and
            reads as chrome rather than as this page's state. */}
        {loading && (
          <Codicon
            className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground"
            name="loading"
            size="0.75rem"
            spinning
          />
        )}
        <Input
          aria-invalid={invalid || undefined}
          aria-label={copy.address}
          className={cn('pr-7', loading && 'pl-6')}
          inputMode="url"
          onBlur={() => setDraft(null)}
          onChange={event => setDraft(event.target.value)}
          onFocus={event => {
            setDraft(shown)
            event.currentTarget.select()
          }}
          onKeyDown={event => {
            if (isSubmitEnter(event)) {
              commit(event.currentTarget.value)
              event.currentTarget.blur()
            }

            if (event.key === 'Escape') {
              setDraft(null)
              event.currentTarget.blur()
            }
          }}
          placeholder={copy.addressPlaceholder}
          size="xs"
          spellCheck={false}
          value={shown}
        />
        <CopyButton
          appearance="inline"
          className="absolute right-1 top-1/2 -translate-y-1/2 rounded-sm p-1"
          iconClassName="size-3"
          label={t.contextMenu.link.copyUrl}
          showLabel={false}
          text={url}
        />
      </div>
      {onToggleAnnotate ? (
        <PaneStripGlyph
          active={annotateMode}
          icon={<Codicon name="comment" size="0.8125rem" />}
          label={annotateMode ? copy.annotateOn : copy.annotate}
          onSelect={onToggleAnnotate}
        />
      ) : null}
      {annotateMode ? (
        <span
          className="hidden shrink-0 items-center rounded-full px-2 py-0.5 text-[0.625rem] font-semibold tracking-wide text-white uppercase sm:inline-flex"
          data-annotate-status="commenting"
          style={{ background: ANNOTATE_BLUE }}
        >
          {copy.commenting}
        </span>
      ) : null}
      {commentCount > 0 && onFlushComments ? (
        <button
          className="shrink-0 rounded-full px-2 py-0.5 text-[0.6875rem] font-semibold text-white"
          onClick={onFlushComments}
          style={{ background: ANNOTATE_BLUE }}
          type="button"
        >
          {copy.addComments(commentCount)}
        </button>
      ) : null}
      {onToggleImport ? (
        <PenImportGlyph
          onImport={onImport}
          onTogglePick={onToggleImport}
          state={importState}
        />
      ) : null}
      {onPopIn ? (
        <PaneStripGlyph
          icon={<Codicon name="screen-normal" size="0.8125rem" />}
          label={t.preview.popIn}
          onSelect={onPopIn}
        />
      ) : onPopOut ? (
        <PaneStripGlyph
          icon={<Codicon name="empty-window" size="0.8125rem" />}
          label={t.preview.popOut}
          onSelect={onPopOut}
        />
      ) : onOpenExternal ? (
        <PaneStripGlyph
          icon={<Codicon name="link-external" size="0.8125rem" />}
          label={t.preview.openInBrowser}
          onSelect={onOpenExternal}
        />
      ) : null}
      <PaneStripGlyph
        active={consoleOpen}
        icon={<Codicon name="terminal" size="0.8125rem" />}
        label={consoleOpen ? copy.hideConsole : copy.showConsole}
        onSelect={onToggleConsole}
      />
      <PaneStripGlyph
        active={devToolsOpen}
        icon={<Codicon name="bug" size="0.8125rem" />}
        label={devToolsOpen ? copy.hideDevTools : copy.openDevTools}
        onSelect={onToggleDevTools}
      />
    </div>
  )

  if (!importState || (!importState.picking && importState.progress === null)) {
    return bar
  }

  return (
    <>
      {bar}
      <PenImportRow
        onHoverPath={onImportHoverPath}
        onImport={onImport}
        onSelectPath={onImportSelectPath}
        onStop={onToggleImport}
        state={importState}
      />
    </>
  )
}

/**
 * The strip's import control. One click offers the two ways in — pick an
 * element, or take the whole page — so cloning a page is a single choice and
 * the crosshair is something you ask for. While picking it is the stop toggle.
 */
function PenImportGlyph({
  onImport,
  onTogglePick,
  state
}: {
  onImport?: (mode: 'page' | 'selection') => void
  onTogglePick: () => void
  state?: PenImportStripState
}) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const busy = state?.progress !== null && state?.progress !== undefined

  if (state?.picking) {
    return (
      <PaneStripGlyph
        active
        disabled={busy}
        icon={<Codicon name="inspect" size="0.8125rem" />}
        label={t.pen.importPicking}
        onSelect={onTogglePick}
      />
    )
  }

  // Controlled so the glyph can show the open state itself: the Tip's tooltip
  // trigger and the menu trigger both write `data-state`, and the tooltip's wins.
  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <Tip label={t.pen.import} placement="toolbar">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={t.pen.import}
            className={cn(
              'self-center select-none [-webkit-app-region:no-drag]',
              open ? 'bg-(--chrome-action-hover) opacity-100' : 'bg-transparent opacity-60 hover:opacity-100'
            )}
            disabled={busy}
            onPointerDown={event => event.stopPropagation()}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="inspect" size="0.8125rem" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="min-w-44" sideOffset={4}>
        <DropdownMenuItem className="gap-2" onSelect={onTogglePick}>
          <Codicon className="text-muted-foreground" name="inspect" size="0.8125rem" />
          {t.pen.importPickElement}
        </DropdownMenuItem>
        <DropdownMenuItem className="gap-2" onSelect={() => onImport?.('page')}>
          <Codicon className="text-muted-foreground" name="window" size="0.8125rem" />
          {t.pen.importPage}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/**
 * The import's own row under the address: what is picked (an ancestor
 * breadcrumb the pointer can walk, like DevTools' crumbs), then Import /
 * Whole page. A second row rather than a pill in the first because the crumbs
 * need the width, and it exists only while a pick or a capture is live.
 */
function PenImportRow({
  onHoverPath,
  onImport,
  onSelectPath,
  onStop,
  state
}: {
  onHoverPath?: (index: null | number) => void
  onImport?: (mode: 'page' | 'selection') => void
  onSelectPath?: (index: number) => void
  onStop?: () => void
  state: PenImportStripState
}) {
  const { t } = useI18n()
  const { pick, progress } = state
  const navRef = useRef<HTMLElement>(null)
  const pickedSelector = pick?.element.selector
  const [overflow, setOverflow] = useState({ end: false, start: false })

  // Which edges hide crumbs; those fade instead of cutting a name in half.
  const measureOverflow = () => {
    const nav = navRef.current

    if (nav) {
      setOverflow({ end: nav.scrollLeft + nav.clientWidth < nav.scrollWidth - 1, start: nav.scrollLeft > 1 })
    }
  }

  // Keep the picked crumb in view as the path changes (the pick is usually
  // the deepest, rightmost entry).
  useEffect(() => {
    navRef.current?.querySelector('[aria-current]')?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    measureOverflow()
  }, [pickedSelector])

  // A resized pane hides or reveals crumbs without any scrolling.
  useEffect(() => {
    const nav = navRef.current

    if (!nav) {
      return
    }

    const observer = new ResizeObserver(measureOverflow)
    observer.observe(nav)

    return () => observer.disconnect()
  }, [])

  return (
    <div
      className="flex min-h-(--titlebar-height) shrink-0 items-center gap-1.5 border-b border-border/60 bg-background px-2 py-1 text-xs"
      data-pen-import-row
    >
      {progress !== null ? (
        <>
          <Codicon className="shrink-0 text-muted-foreground" name="loading" size="0.8125rem" spinning />
          <span className="min-w-0 flex-1 truncate text-muted-foreground">
            {progress < 1 ? t.pen.importProgress(Math.round(progress * 100)) : t.pen.importing}
          </span>
        </>
      ) : pick ? (
        <>
          <nav
            aria-label={pick.element.label ?? pick.element.tag}
            className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            onScroll={measureOverflow}
            ref={navRef}
            style={{ maskImage: crumbEdgeMask(overflow) }}
          >
            {pick.path.map((entry, index) => {
              const current = index === pick.pathIndex

              return (
                <span className="flex shrink-0 items-center gap-0.5" key={`${index}-${entry.label}`}>
                  {index > 0 ? (
                    <Codicon className="shrink-0 text-muted-foreground/60" name="chevron-right" size="0.6875rem" />
                  ) : null}
                  <button
                    aria-current={current ? 'true' : undefined}
                    className={cn(
                      'cursor-pointer rounded-sm px-1 py-0.5 font-mono text-[0.6875rem] leading-4 whitespace-nowrap transition-colors',
                      current
                        ? 'bg-(--chrome-action-hover) text-foreground'
                        : 'text-muted-foreground hover:bg-(--chrome-action-hover) hover:text-foreground'
                    )}
                    onClick={() => onSelectPath?.(index)}
                    onMouseEnter={() => onHoverPath?.(index)}
                    onMouseLeave={() => onHoverPath?.(null)}
                    type="button"
                  >
                    {crumbLabel(entry, current)}
                  </button>
                </span>
              )
            })}
          </nav>
          <Button onClick={() => onImport?.('selection')} size="xs" type="button" variant="default">
            {t.pen.importSelection}
          </Button>
          <Button onClick={() => onImport?.('page')} size="xs" type="button" variant="outline">
            {t.pen.importPage}
          </Button>
        </>
      ) : (
        <>
          <Codicon className="shrink-0 text-muted-foreground" name="inspect" size="0.8125rem" />
          <span className="min-w-0 flex-1 truncate text-muted-foreground">{t.pen.importPickHint}</span>
          <Button onClick={() => onImport?.('page')} size="xs" type="button" variant="outline">
            {t.pen.importPage}
          </Button>
          <Button onClick={onStop} size="xs" type="button" variant="ghost">
            {t.pen.importCancel}
          </Button>
        </>
      )}
    </div>
  )
}

/** Fade the crumb strip's hidden edge(s); none when everything fits. */
function crumbEdgeMask({ end, start }: { end: boolean; start: boolean }): string | undefined {
  if (!start && !end) {
    return undefined
  }

  const stops = [start ? 'transparent, black 2.5rem' : 'black', end ? 'black calc(100% - 2.5rem), transparent' : 'black']

  return `linear-gradient(to right, ${stops.join(', ')})`
}

/** Ancestors read as `tag#id` (classes are noise at crumb width); the pick keeps its full label. */
function crumbLabel(entry: PenImportPick['path'][number], current: boolean): string {
  if (entry.componentName) {
    return entry.componentName
  }

  if (current) {
    return entry.label
  }

  const match = /^([a-z0-9-]+)(#[^.\s]+)?/iu.exec(entry.label)

  return match ? `${match[1]}${match[2] ?? ''}` : entry.label
}
