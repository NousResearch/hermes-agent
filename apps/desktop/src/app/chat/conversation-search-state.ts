import { atom } from 'nanostores'

import { CONVERSATION_SEARCH_PAGE_SIZE, type ConversationSearchPage } from '@/api/conversation-search'
import { isMissingRestEndpoint } from '@/lib/gateway-rpc'

export interface ConversationSearchState {
  open: boolean
  query: string
  phase: 'idle' | 'pending' | 'ready' | 'error'
  page: ConversationSearchPage | null
  selected: number
  error: 'searchFailed' | 'unavailable' | 'jumpFailed' | null
}

export function createConversationSearch(load: (query: string, offset: number) => Promise<ConversationSearchPage>) {
  const empty: ConversationSearchState = {
    open: false,
    query: '',
    phase: 'idle',
    page: null,
    selected: -1,
    error: null
  }

  const state = atom<ConversationSearchState>({ ...empty })
  let generation = 0

  const reset = (open = false) => {
    generation++
    state.set({ ...empty, open })
  }

  const setQuery = (query: string) => {
    generation++
    state.set({ ...empty, open: state.get().open, query, phase: query.trim() ? 'pending' : 'idle' })
  }

  const search = async (offset = 0) => {
    const captured = state.get()

    if (!captured.open || !captured.query.trim()) {
      return false
    }

    const token = ++generation
    state.set({ ...captured, phase: 'pending', error: null, page: null, selected: -1 })

    try {
      const page = await load(captured.query, offset)

      if (token !== generation || !state.get().open) {
        return false
      }

      state.set({ ...state.get(), phase: 'ready', page, selected: -1 })

      return true
    } catch (error) {
      if (token !== generation || !state.get().open) {
        return false
      }

      state.set({
        ...state.get(),
        phase: 'error',
        page: null,
        error:
          isMissingRestEndpoint(error) && /no such api endpoint|endpoint is likely missing/i.test(String(error))
            ? 'unavailable'
            : 'searchFailed'
      })

      return false
    }
  }

  const select = (index: number) => {
    const current = state.get()

    if (current.phase !== 'ready' || !current.page?.results[index]) {
      return null
    }

    state.set({ ...current, selected: index, error: null })

    return current.page.results[index]
  }

  const step = async (direction: 1 | -1) => {
    const current = state.get()
    const page = current.page

    if (current.phase !== 'ready' || !page?.results.length) {
      return null
    }

    const index = current.selected < 0 ? (direction === 1 ? 0 : page.results.length - 1) : current.selected + direction

    if (index >= 0 && index < page.results.length) {
      return select(index)
    }

    const offset =
      direction === 1
        ? page.pagination.next_offset
        : Math.max(0, page.pagination.offset - CONVERSATION_SEARCH_PAGE_SIZE)

    if (offset === null || (direction === -1 && page.pagination.offset === 0)) {
      return null
    }

    if (!(await search(offset))) {
      return null
    }

    return select(direction === 1 ? 0 : (state.get().page?.results.length ?? 0) - 1)
  }

  return {
    state,
    reset,
    setQuery,
    search,
    select,
    step,
    jumpFailed: () => state.set({ ...state.get(), error: 'jumpFailed' }),
    dispose: reset
  }
}
