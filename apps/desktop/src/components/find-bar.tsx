import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router'

import { appViewForPath, isOverlayView } from '@/app/routes'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { findBarKeyAction, formatMatchLabel } from '@/lib/find-in-page'
import { cn } from '@/lib/utils'
import {
  $findInPage,
  closeFindBar,
  findNext,
  findPrevious,
  initFindInPageListener,
  initOpenFindBarListener,
  setFindQuery
} from '@/store/find-in-page'

/**
 * Find-in-page overlay (⌘F).
 *
 * Drives Electron's `webContents.findInPage` via the preload bridge so the
 * user gets the native browser-like incremental search (highlight, step,
 * Escape to clear) over the rendered chat transcript + editor panels. Multi-
 * window routing is handled in the main process — see
 * apps/desktop/electron/find-in-page.ts.
 *
 * Accelerators, matching the platform convention (and Claude Desktop's set):
 * ⌘F opens (via the `view.findInPage` keybind), ⌘G / ⌘⇧G step next/previous
 * from anywhere while the bar is open, Enter / ⇧Enter step from the input,
 * and Escape closes + clears the native selection.
 *
 * Key routing lives in `lib/find-in-page.ts` as a pure matcher so the
 * accelerator set is testable without a DOM.
 */
export function FindBar() {
  const { t } = useI18n()
  const { active, query, matchOrdinal, matchCount } = useStore($findInPage)
  const inputRef = useRef<HTMLInputElement>(null)
  const nativeSearchRequestRef = useRef(0)
  const [localQuery, setLocalQuery] = useState('')
  const [filesPaneRight, setFilesPaneRight] = useState<number | null>(null)
  const { pathname } = useLocation()

  // Navigating away (opening another session, a settings page, …) closes the
  // bar and clears the native highlight. Electron's findInPage selection is
  // per-webContents, not per-route: without this, highlights (and a stale
  // match counter) from the previous chat would survive onto the next view.
  // Implemented as effect cleanup so the first render never fires it, a
  // pathname change tears down the previous route's search, and unmount
  // (session/profile switches that remount the shell) gets the same teardown.
  // closeFindBar is idempotent, so a closed bar never re-enters the bridge.
  useEffect(() => {
    void pathname

    return () => closeFindBar()
  }, [pathname])

  // Focus input when find bar opens.
  useEffect(() => {
    if (active) {
      setLocalQuery('')
      // Small delay so the DOM paints the input before we focus.
      const id = requestAnimationFrame(() => inputRef.current?.focus())

      return () => cancelAnimationFrame(id)
    }

    return undefined
  }, [active])

  // The files pane (right sidebar, `aside[aria-label="Right sidebar"]`) is a
  // floating right rail. The find bar is `fixed right-4` by default, which
  // would cover the files pane's header + first rows when it is open. Measure
  // the pane's live rect and park the bar just left of it instead.
  //
  // The pane can open, close, or be drag-resized *while the bar is open* (its
  // visibility lives in the contrib workspace registry, not a store we can
  // subscribe to from here), so a window `resize` listener alone is not
  // enough: a ResizeObserver tracks width drags of the current aside, and a
  // body-scoped MutationObserver catches the aside mounting/unmounting and
  // re-attaches the ResizeObserver to the new node. All paths coalesce into
  // one rAF-batched measure; everything tears down when the bar closes.
  useEffect(() => {
    if (!active) {
      setFilesPaneRight(null)

      return undefined
    }

    let rafId: null | number = null
    let observedAside: Element | null = null

    const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => scheduleMeasure())

    const measure = () => {
      rafId = null
      const aside = document.querySelector('aside[aria-label="Right sidebar"]')

      // The aside can be replaced wholesale (pane closed and reopened, panes
      // flipped) — retarget the ResizeObserver whenever its identity changes.
      if (aside !== observedAside) {
        if (observedAside) {
          resizeObserver?.unobserve(observedAside)
        }

        if (aside) {
          resizeObserver?.observe(aside)
        }

        observedAside = aside
      }

      const rect = aside?.getBoundingClientRect()

      if (rect && rect.width > 0 && rect.left < window.innerWidth) {
        setFilesPaneRight(window.innerWidth - rect.left)
      } else {
        setFilesPaneRight(null)
      }
    }

    const scheduleMeasure = () => {
      rafId ??= requestAnimationFrame(measure)
    }

    // Catches the pane opening/closing while the bar is up. Scoped to
    // childList mutations only (no attributes/characterData), and the bar is
    // a transient overlay, so the observer lives only for the find session.
    const mutationObserver = new MutationObserver(scheduleMeasure)
    mutationObserver.observe(document.body, { childList: true, subtree: true })

    measure()
    window.addEventListener('resize', scheduleMeasure)

    return () => {
      window.removeEventListener('resize', scheduleMeasure)
      mutationObserver.disconnect()
      resizeObserver?.disconnect()

      if (rafId != null) {
        cancelAnimationFrame(rafId)
      }
    }
  }, [active, pathname])

  // Subscribe to found-in-page results from the main process. Refcounted in
  // the store, so a remount (connection re-home) can't stack listeners; the
  // subscription is deliberately mount-scoped and NOT tied to `active` —
  // results for an in-flight search must still land if the bar just closed.
  useEffect(() => initFindInPageListener(), [])

  // Mirror the find-results listener for the main-process Ctrl/Cmd+F
  // forward — on Pop!_OS / GNOME the GTK compositor grabs the chord at
  // the windowing layer (#81727).
  useEffect(() => initOpenFindBarListener(), [])

  // Debounce search — fire findInPage 200ms after the user stops typing.
  useEffect(() => {
    const requestId = ++nativeSearchRequestRef.current
    const input = inputRef.current

    if (!active || !localQuery) {
      if (input?.inert) {
        input.inert = false
      }

      return undefined
    }

    const id = setTimeout(() => {
      if (!input) {
        void setFindQuery(localQuery)

        return
      }

      const hadFocus = document.activeElement === input
      const selectionStart = input.selectionStart
      const selectionEnd = input.selectionEnd

      // The HTML inert contract excludes this truthful search control from
      // find-in-page. Keep it inert until the IPC reply confirms Electron has
      // started the request, then restore focus and selection.
      input.inert = true

      void setFindQuery(localQuery).finally(() => {
        if (nativeSearchRequestRef.current !== requestId) {
          return
        }

        input.inert = false

        if (hadFocus && input.isConnected) {
          input.focus({ preventScroll: true })

          if (selectionStart !== null && selectionEnd !== null) {
            input.setSelectionRange(selectionStart, selectionEnd)
          }
        }
      })
    }, 200)

    // Cleanup covers every exit: another keystroke, the bar closing, and
    // unmount. Nothing can fire a find after the bar is gone.
    return () => clearTimeout(id)
  }, [active, localQuery])

  // Global accelerators while the bar is open: Escape closes, ⌘G / ⌘⇧G step.
  // Capture-phase so they win regardless of which element inside the shell
  // owns focus (composer textarea, side panel button, …). ⌘G is also bound to
  // `view.toggleReview` in the keybinds registry — this listener runs in the
  // capture phase and stops propagation, so while the find bar is open ⌘G
  // means "find next" and the review toggle does not also fire. Closing the
  // bar hands ⌘G straight back to the review pane.
  useEffect(() => {
    if (!active) {
      return undefined
    }

    const onKeyDown = (event: KeyboardEvent) => {
      const action = findBarKeyAction(event)

      if (!action) {
        return
      }

      event.preventDefault()
      event.stopPropagation()

      if (action === 'close') {
        closeFindBar()
      } else if (action === 'next') {
        findNext()
      } else {
        findPrevious()
      }
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [active])

  if (!active || isOverlayView(appViewForPath(pathname))) {
    return null
  }

  const onInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value
    setLocalQuery(value)

    // Empty query: clear highlights immediately rather than after the debounce.
    if (!value) {
      void setFindQuery('')
    }
  }

  // Enter / ⇧Enter step while focus is in the input. Escape and ⌘G are handled
  // by the window listener above, so they are intentionally not duplicated
  // here — `inInput` only unlocks the bare-Enter family.
  const onKeyDown = (event: React.KeyboardEvent) => {
    const action = findBarKeyAction(event, { inInput: true })

    if (action !== 'next' && action !== 'previous') {
      return
    }

    event.preventDefault()

    if (action === 'next') {
      findNext()
    } else {
      findPrevious()
    }
  }

  const matchLabel = formatMatchLabel(query, matchOrdinal, matchCount)

  const barStyle = filesPaneRight != null ? { right: `calc(${filesPaneRight}px + 0.75rem)` } : undefined

  return (
    <div
      className={cn(
        // Floats BELOW the titlebar band. The bar mounts at the overlay root,
        // outside the app-shell subtree that defines --titlebar-height, so the
        // fallback must be the real titlebar height (34px, matching
        // floating-hud.ts / notifications.tsx) — a 0px fallback parks the bar
        // inside the titlebar strip, underneath the native min/max/close
        // window-controls overlay on Windows/Linux.
        'pointer-events-auto fixed right-4 top-[calc(var(--titlebar-height,34px)+0.5rem)] z-50',
        'flex items-center gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-surface-background) px-2 py-1.5 shadow-md'
      )}
      role="search"
      style={barStyle}
    >
      <input
        aria-label={t.keybinds.actions['view.findInPage'] ?? 'Find in page'}
        autoComplete="off"
        className="h-6 w-40 bg-transparent text-xs text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary)"
        onChange={onInput}
        onKeyDown={onKeyDown}
        placeholder={t.keybinds.actions['view.findInPage'] ?? 'Find in page'}
        ref={inputRef}
        type="search"
        value={localQuery}
      />

      {matchLabel && (
        <span aria-live="polite" className="min-w-[3rem] text-center text-[0.6875rem] text-(--ui-text-tertiary)">
          {matchLabel}
        </span>
      )}

      <Tip label={t.findInPage.previous}>
        <button
          aria-label={t.findInPage.previous}
          className="flex h-7 w-7 items-center justify-center rounded text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)"
          onClick={findPrevious}
          type="button"
        >
          <svg height="14" viewBox="0 0 16 16" width="14">
            <path d="M4 10l4-4 4 4" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
      </Tip>

      <Tip label={t.findInPage.next}>
        <button
          aria-label={t.findInPage.next}
          className="flex h-7 w-7 items-center justify-center rounded text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)"
          onClick={findNext}
          type="button"
        >
          <svg height="14" viewBox="0 0 16 16" width="14">
            <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
      </Tip>

      <button
        aria-label={t.common.close}
        className="flex h-7 w-7 items-center justify-center rounded text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)"
        onClick={closeFindBar}
        type="button"
      >
        <svg height="12" viewBox="0 0 12 12" width="12">
          <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </button>
    </div>
  )
}
