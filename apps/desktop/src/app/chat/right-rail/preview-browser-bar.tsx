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
 *
 * On the browser-hosted build there is no Electron `<webview>`, only an iframe
 * the renderer cannot read. The bar still navigates and reloads it, but back
 * and forward stay disabled — a frame's history is its own, and this side can
 * neither read nor walk it — and there is no guest console or DevTools to open.
 */

import { useEffect, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { Input } from '@/components/ui/input'
import { PaneStripGlyph } from '@/components/ui/pane-tab'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { ANNOTATE_BLUE } from '@/lib/preview-annotate'
import { cn } from '@/lib/utils'

import { normalizePreviewAddress } from './preview-address'

interface PreviewBrowserBarProps {
  annotateMode?: boolean
  canGoBack: boolean
  canGoForward: boolean
  commentCount?: number
  consoleOpen: boolean
  devToolsOpen: boolean
  loading: boolean
  onBack: () => void
  /** The pane's Close — the one visible way out of a full-width Browser. */
  onClose?: () => void
  onFlushComments?: () => void
  onForward: () => void
  onNavigate: (url: string) => void
  onOpenExternal?: () => void
  /** Lift this Browser out of the layout into the floating window. */
  onFloat?: () => void
  onPopIn?: () => void
  onPopOut?: () => void
  onReload: () => void
  onToggleAnnotate?: () => void
  /** Absent on the browser-hosted iframe: the frame is opaque to the renderer,
   *  so there is no guest console or DevTools to open. */
  onToggleConsole?: () => void
  onToggleDevTools?: () => void
  /** The page's CURRENT address (it moves as the user navigates), not the
   *  target the tab was opened with. */
  url: string
}

/**
 * Address parsing lives in `preview-address` so the Browser's start page can
 * share it; the bar re-exports it, since the bar is where it is used.
 */
export { normalizePreviewAddress }

export function PreviewBrowserBar({
  annotateMode = false,
  canGoBack,
  canGoForward,
  commentCount = 0,
  consoleOpen,
  devToolsOpen,
  loading,
  onBack,
  onClose,
  onFlushComments,
  onForward,
  onNavigate,
  onFloat,
  onOpenExternal,
  onPopIn,
  onPopOut,
  onReload,
  onToggleAnnotate,
  onToggleConsole,
  onToggleDevTools,
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

  return (
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
      {onFloat ? (
        <PaneStripGlyph
          icon={<Codicon name="multiple-windows" size="0.8125rem" />}
          label={t.preview.float.float}
          onSelect={onFloat}
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
      {onToggleConsole ? (
        <PaneStripGlyph
          active={consoleOpen}
          icon={<Codicon name="terminal" size="0.8125rem" />}
          label={consoleOpen ? copy.hideConsole : copy.showConsole}
          onSelect={onToggleConsole}
        />
      ) : null}
      {onToggleDevTools ? (
        <PaneStripGlyph
          active={devToolsOpen}
          icon={<Codicon name="bug" size="0.8125rem" />}
          label={devToolsOpen ? copy.hideDevTools : copy.openDevTools}
          onSelect={onToggleDevTools}
        />
      ) : null}
      {/* The last glyph is the way OUT: a Browser that filled the layout has
          no strip ✕ in reach, and the only other close is asking the agent
          (#92500). Same verb the tab carries — the pane routes it. */}
      {onClose && (
        <PaneStripGlyph icon={<Codicon name="close" size="0.8125rem" />} label={t.common.close} onSelect={onClose} />
      )}
    </div>
  )
}
