import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { KbdCombo } from '@/components/ui/kbd'
import { SearchField } from '@/components/ui/search-field'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Search } from '@/lib/icons'
import { recordIaAction } from '@/store/ia-metrics'
import { $bindings, bindingsFor } from '@/store/keybinds'
import { SESSION_SEARCH_FOCUS_EVENT } from '@/store/layout'

// Brand marks, never translated: the product's wordmark over its maker (Top 16292:15498).
const WORDMARK = 'Hermes'
const WORDMARK_CAPTION = '// Nous Research'

interface PortalSidebarSearch {
  open: boolean
  id: string
  inputRef: RefObject<HTMLInputElement | null>
  toggleRef: RefObject<HTMLButtonElement | null>
  reveal: () => void
  close: () => void
}

/** Search lives behind the header's button (16292:15507): the button, the session-search shortcut
 *  and the command palette all bring the field out with focus in it; putting it away ends the search
 *  so the list returns whole. */
export function usePortalSidebarSearch(setQuery: (query: string) => void): PortalSidebarSearch {
  const inputRef = useRef<HTMLInputElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)
  const id = useId()
  const [open, setOpen] = useState(false)
  const [focusRequest, setFocusRequest] = useState(0)

  const reveal = useCallback(() => {
    setOpen(true)
    setFocusRequest(request => request + 1)
  }, [])

  useEffect(() => {
    if (focusRequest) {
      inputRef.current?.focus({ preventScroll: true })
    }
  }, [focusRequest])

  useEffect(() => {
    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, reveal)

    return () => window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, reveal)
  }, [reveal])

  const close = () => {
    setOpen(false)
    setQuery('')
  }

  return { open, id, inputRef, toggleRef, reveal, close }
}

/** The sidebar's header: the wordmark over its caption, the search button at the end. */
export function PortalSidebarHead({ search }: { search: PortalSidebarSearch }) {
  const { t } = useI18n()

  return (
    <div className="portal-sidebar-head">
      <div aria-hidden className="portal-sidebar-brand">
        <span className="portal-sidebar-wordmark">{WORDMARK}</span>
        <span className="portal-sidebar-brand-caption">{WORDMARK_CAPTION}</span>
      </div>
      <Tip label={t.sidebar.searchAria}>
        <Button
          aria-controls={search.open ? search.id : undefined}
          aria-expanded={search.open}
          aria-label={t.sidebar.searchAria}
          className="portal-sidebar-search-toggle"
          onClick={() => (search.open ? search.close() : search.reveal())}
          ref={search.toggleRef}
          size="icon"
          variant="ghost"
        >
          <Search aria-hidden />
        </Button>
      </Tip>
    </div>
  )
}

interface PortalSidebarSearchFieldProps {
  search: PortalSidebarSearch
  query: string
  pending: boolean
  onChange: (query: string) => void
}

/** The field the header's button brings out: New chat's row on a rim, the shortcut's keycaps at its
 *  end while it rests. Escape on an empty field puts it away, back on its button. */
export function PortalSidebarSearchField({ search, query, pending, onChange }: PortalSidebarSearchFieldProps) {
  const { t } = useI18n()
  const bindings = useStore($bindings)
  // The keycaps name the shortcut while the field rests; typing hands the end of the row to Clear.
  const shortcut = query ? undefined : bindingsFor('session.focusSearch', bindings)[0]

  return (
    <div className="portal-sidebar-search-row" id={search.id}>
      <SearchField
        aria-label={t.sidebar.searchAria}
        containerClassName="portal-sidebar-search"
        escapeClears
        inputRef={search.inputRef}
        loading={pending}
        onChange={onChange}
        onFocus={() => recordIaAction('ia.search.opened')}
        onKeyDown={event => {
          if (event.key === 'Escape' && !query) {
            event.preventDefault()
            search.close()
            search.toggleRef.current?.focus()
          }
        }}
        placeholder={t.sidebar.searchPlaceholder}
        trailingAction={
          shortcut && <KbdCombo aria-hidden className="portal-kbd" combo={shortcut} data-slot="session-search-shortcut" />
        }
        value={query}
      />
    </div>
  )
}
