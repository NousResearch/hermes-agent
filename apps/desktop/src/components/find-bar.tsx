import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $findInPage,
  closeFindBar,
  findNext,
  findPrevious,
  initFindInPageListener,
  setFindQuery
} from '@/store/find-in-page'

/**
 * Find-in-page overlay (Ctrl/Cmd+F).
 *
 * Drives Electron's `webContents.findInPage` via the preload bridge so the
 * user gets the native browser-like incremental search (highlight, Enter to
 * step, Shift+Enter to step backwards, Escape to close) over the rendered
 * chat transcript + editor panels. Multi-window routing is handled in the
 * main process — see apps/desktop/electron/find-in-page.ts.
 */
export function FindBar() {
  const { t } = useI18n()
  const { active, query, matchOrdinal, matchCount } = useStore($findInPage)
  const inputRef = useRef<HTMLInputElement>(null)
  const [localQuery, setLocalQuery] = useState('')

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

  // Subscribe to found-in-page results from the main process.
  useEffect(() => {
    const unsub = initFindInPageListener()

    return unsub
  }, [])

  // Debounce search — fire findInPage 200ms after the user stops typing.
  // The ref lets us cancel the pending timeout when the bar closes.
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!active || !localQuery) {
      return undefined
    }

    const id = setTimeout(() => setFindQuery(localQuery), 200)
    debounceRef.current = id

    return () => {
      clearTimeout(id)
      debounceRef.current = null
    }
  }, [active, localQuery])

  // Cancel pending debounce + close highlights when the bar closes.
  useEffect(() => {
    if (!active && debounceRef.current) {
      clearTimeout(debounceRef.current)
      debounceRef.current = null
    }
  }, [active])

  // Global Escape listener — works even when focus is outside the input.
  // Captured so the find bar can always close regardless of which element
  // inside the shell owns focus (composer textarea, side panel button, etc).
  useEffect(() => {
    if (!active) {
      return undefined
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        closeFindBar()
      }
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [active])

  if (!active) {
    return null
  }

  const onInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value
    setLocalQuery(val)

    // Empty query: clear highlights.
    if (!val) {
      setFindQuery('')
    }
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault()
      closeFindBar()
    } else if (e.key === 'Enter') {
      e.preventDefault()

      if (e.shiftKey) {
        findPrevious()
      } else {
        findNext()
      }
    }
  }

  const matchLabel =
    query && matchCount > 0
      ? `${matchOrdinal}/${matchCount}`
      : query && matchCount === 0
        ? '0/0'
        : ''

  return (
    <div
      className={cn(
        'pointer-events-auto fixed right-4 top-[calc(var(--titlebar-height,0px)+0.5rem)] z-50',
        'flex items-center gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-surface-background) px-2 py-1 shadow-md'
      )}
    >
      <input
        className="h-6 w-40 bg-transparent text-xs text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary)"
        onChange={onInput}
        onKeyDown={onKeyDown}
        placeholder={t.keybinds.actions['view.findInPage'] ?? 'Find'}
        ref={inputRef}
        type="text"
        value={localQuery}
      />

      {matchLabel && (
        <span className="min-w-[3rem] text-center text-[0.6875rem] text-(--ui-text-tertiary)">
          {matchLabel}
        </span>
      )}

      <Tip label="Previous">
        <button
          className="flex h-5 w-5 items-center justify-center rounded text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)"
          onClick={findPrevious}
          type="button"
        >
          <svg height="12" viewBox="0 0 16 16" width="12">
            <path d="M4 10l4-4 4 4" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
      </Tip>

      <Tip label="Next">
        <button
          className="flex h-5 w-5 items-center justify-center rounded text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)"
          onClick={findNext}
          type="button"
        >
          <svg height="12" viewBox="0 0 16 16" width="12">
            <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
      </Tip>

      <Tip label="Close">
        <button
          className="flex h-5 w-5 items-center justify-center rounded text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)"
          onClick={closeFindBar}
          type="button"
        >
          <svg height="10" viewBox="0 0 12 12" width="10">
            <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
      </Tip>
    </div>
  )
}
