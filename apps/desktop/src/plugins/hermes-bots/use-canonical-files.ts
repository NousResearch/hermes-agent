/** #104199 snapshot/cache coordination, scoped to one canonical dialog lifetime. */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import {
  CanonicalFilesError,
  canonicalFilesFailure,
  type FilesAuthority,
  type FilesFailure,
  listCanonicalFiles
} from './canonical-files-client'
import type { CanonicalGroupBinding } from './canonical-groups'
import {
  GROUP_FILES_MAX_QUERY_LENGTH,
  GROUP_FILES_PAGE_SIZE,
  type GroupFilesPage,
  validateGroupFilesContinuation
} from './group-files-parser'

interface CachedPage {
  data: GroupFilesPage
  cursor?: string
}
interface FilesState {
  pages: CachedPage[]
  index: number
  query: string
  loading: boolean
  failure: FilesFailure | null
  latestFileSeq: number
  reconnected: boolean
}

export function useCanonicalFiles({
  binding,
  open,
  authority,
  accessDenied = false
}: {
  binding: CanonicalGroupBinding
  open: boolean
  authority?: FilesAuthority
  accessDenied?: boolean
}) {
  const [selected] = useState(() => ({ binding: { ...binding }, authority: authority ? { ...authority } : undefined }))

  const [state, setState] = useState<FilesState>({
    pages: [],
    index: 0,
    query: '',
    loading: open && !accessDenied,
    failure: null,
    latestFileSeq: 0,
    reconnected: false
  })

  // This ref owns synchronous intent retirement, not a copy of an external store.
  const model = useRef({
    state,
    open,
    denied: accessDenied,
    generation: 0,
    controller: null as AbortController | null,
    deliveries: new AbortController()
  })

  const publish = useCallback((patch: Partial<FilesState>) => {
    const next = { ...model.current.state, ...patch }
    model.current.state = next
    setState(next)
  }, [])

  const retire = useCallback(() => {
    const current = model.current
    current.generation++
    current.controller?.abort()
    current.deliveries.abort()
  }, [])

  const invalidateAccess = useCallback(
    (failure: FilesFailure = 'access') => {
      retire()
      publish({ pages: [], index: 0, failure, loading: false, latestFileSeq: 0, reconnected: false })
    },
    [publish, retire]
  )

  const fetchPage = useCallback(
    async (mode: 'latest' | 'older' | 'retry') => {
      const current = model.current

      if (!current.open || current.denied) {
        return
      }

      const before = current.state
      const held = before.pages[before.index]

      const cursor =
        mode === 'older' ? (held?.data.nextCursor ?? undefined) : mode === 'retry' ? held?.cursor : undefined

      retire()
      const generation = current.generation
      const controller = new AbortController()
      current.controller = controller
      current.deliveries = new AbortController()
      publish({
        loading: true,
        failure: null,
        ...(mode === 'latest' ? { pages: [], index: 0, reconnected: false, latestFileSeq: 0 } : {})
      })

      try {
        const selectedAuthority =
          selected.authority ?? (mode === 'latest' ? undefined : (held?.data.authority ?? undefined))

        const data = await listCanonicalFiles(
          selected.binding,
          {
            limit: GROUP_FILES_PAGE_SIZE,
            ...(cursor ? { cursor } : {}),
            ...(before.query.trim() ? { query: before.query.trim() } : {})
          },
          selectedAuthority,
          controller.signal
        )

        if (!current.open || current.generation !== generation) {
          return
        }

        if (mode === 'older' && held) {
          try {
            validateGroupFilesContinuation(held.data, data)

            if (data.nextCursor && before.pages.some(page => page.data.nextCursor === data.nextCursor)) {
              throw new Error('Repeated cursor')
            }

            const boundary = before.pages
              .slice(0, before.index + 1)
              .reverse()
              .find(page => page.data.items.length)?.data

            if (boundary && boundary !== held.data) {
              validateGroupFilesContinuation(boundary, data)
            }

            const seen = new Set(
              before.pages.flatMap(page => page.data.items.map(item => item.attachment.attachmentId))
            )

            if (data.items.some(item => seen.has(item.attachment.attachmentId))) {
              throw new Error('Repeated file')
            }
          } catch {
            throw new CanonicalFilesError('cursor')
          }
        }

        publish({
          loading: false,
          failure: null,
          latestFileSeq: Math.max(mode === 'latest' ? 0 : before.latestFileSeq, data.latestFileSeq ?? 0),
          reconnected: before.failure === 'offline' || before.failure === 'timeout' || before.reconnected,
          ...(mode === 'retry' && held
            ? {}
            : mode === 'older'
              ? { pages: [...before.pages.slice(0, before.index + 1), { data, cursor }], index: before.index + 1 }
              : { pages: [{ data }], index: 0 })
        })
      } catch (error) {
        if (!current.open || current.generation !== generation) {
          return
        }

        const failure = canonicalFilesFailure(error)

        if (failure === 'access' || failure === 'scope' || failure === 'gone') {
          invalidateAccess(failure === 'gone' ? 'access' : failure)
        } else {
          publish({ loading: false, failure })
        }
      } finally {
        controller.abort()
      }
    },
    [selected, invalidateAccess, publish, retire]
  )

  // Retire downloads before a closed/replaced/denied dialog can deliver bytes.
  useLayoutEffect(() => {
    const current = model.current
    current.open = open
    current.denied = accessDenied

    if (accessDenied) {
      invalidateAccess()
    }

    if (!open) {
      retire()
    }

    return () => {
      current.open = false
      retire()
    }
  }, [open, accessDenied, invalidateAccess, retire])

  useEffect(() => {
    if (!open || accessDenied) {
      return
    }

    const timer = setTimeout(() => void fetchPage('latest'), state.query.trim() ? 250 : 0)

    return () => {
      clearTimeout(timer)
      retire()
    }
  }, [open, accessDenied, state.query, fetchPage, retire])

  const setQuery = (value: string) => {
    const query = [...value].slice(0, GROUP_FILES_MAX_QUERY_LENGTH).join('')

    if (query === model.current.state.query) {
      return
    }

    retire()
    publish({
      query,
      pages: [],
      index: 0,
      loading: !model.current.denied,
      failure: null,
      latestFileSeq: 0,
      reconnected: false
    })
  }

  const move = (index: number) => {
    model.current.deliveries.abort()
    model.current.deliveries = new AbortController()
    publish({ index })
  }

  const older = () => {
    const current = model.current.state

    if (current.loading) {
      return
    }

    if (current.pages[current.index + 1]) {
      move(current.index + 1)
    } else if (current.pages[current.index]?.data.hasMore) {
      void fetchPage('older')
    }
  }

  const newer = () => {
    if (!model.current.state.loading) {
      move(Math.max(0, model.current.state.index - 1))
    }
  }

  return {
    ...state,
    page: state.pages[state.index]?.data,
    setQuery,
    older,
    newer,
    deliverySignal: model.current.deliveries.signal,
    invalidateAccess,
    cancel: retire,
    retry: () => void fetchPage(state.pages.length ? 'retry' : 'latest'),
    latest: () => void fetchPage('latest')
  }
}
