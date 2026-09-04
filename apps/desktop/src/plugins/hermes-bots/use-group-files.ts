/** The dialog owns a snapshot; connectivity observations never own its page position. */

import { useCallback, useEffect, useRef, useState } from 'react'

import { $groupChats } from './group-chat'
import { groupFileFailure, type GroupFileFailure, withGroupFileDeadline } from './group-file-errors'
import {
  captureGroupFileAccess,
  groupFileAccessCurrent,
  invalidateGroupFileAccess,
  subscribeGroupFileAccess
} from './group-files-access'
import { GROUP_FILES_PAGE_SIZE, isGroupFilesCursorError, validateGroupFilesContinuation } from './group-files-client'
import type { GroupFilesListInput, GroupFilesPage } from './group-files-client'

export type GroupFilesAvailability = 'available' | 'offline' | 'unavailable'
export type GroupFilesLoader = (
  group: string,
  input?: GroupFilesListInput,
  signal?: AbortSignal
) => Promise<GroupFilesPage>
interface CachedPage {
  data: GroupFilesPage
  cursor?: string
}
interface FilesState {
  pages: CachedPage[]
  index: number
  query: string
  loading: boolean
  failure: GroupFileFailure | null
  cursorExpired: boolean
  offline: boolean
  unavailable: boolean
  reconnected: boolean
  latestFileSeq: number
}

interface Options {
  group: string
  open: boolean
  availability: GroupFilesAvailability
  observation?: unknown
  loadPage: GroupFilesLoader
}

export function useGroupFiles({ group, open, availability, observation, loadPage }: Options) {
  const [state, setState] = useState<FilesState>(() => ({
    pages: [],
    index: 0,
    query: '',
    loading: availability === 'available',
    failure: null,
    cursorExpired: false,
    offline: availability === 'offline',
    unavailable: availability === 'unavailable',
    reconnected: false,
    latestFileSeq: 0
  }))

  // This ref owns synchronous request/cache coordination, not a lagging atom mirror.
  const model = useRef({
    state,
    generation: 0,
    misses: 0,
    availability,
    open,
    controller: null as AbortController | null,
    deliveries: new AbortController(),
    access: captureGroupFileAccess($groupChats.get()[group])
  })

  const publish = useCallback((patch: Partial<FilesState>) => {
    const next = { ...model.current.state, ...patch }
    model.current.state = next
    setState(next)
  }, [])

  const denied = useCallback(() => {
    const current = model.current
    current.generation += 1
    current.controller?.abort()
    current.deliveries.abort()
    publish({
      pages: [],
      index: 0,
      loading: false,
      failure: 'access',
      offline: false,
      reconnected: false,
      latestFileSeq: 0
    })
  }, [publish])

  useEffect(() => subscribeGroupFileAccess(model.current.access, denied), [denied])

  const invalidateAccess = () => {
    const access = model.current.access
    invalidateGroupFileAccess(access ? { state: access.state, generation: access.state.generation } : null)
    denied()
  }

  const cancel = () => {
    const current = model.current
    current.generation += 1
    current.controller?.abort()
    current.deliveries.abort()
  }

  const fetchPage = useCallback(
    async (mode: 'latest' | 'older' | 'retry') => {
      const current = model.current

      if (!current.open) {
        return
      }

      const before = current.state
      const held = before.pages[before.index]
      const cursor = mode === 'older' ? held?.data.nextCursor || undefined : mode === 'retry' ? held?.cursor : undefined
      const generation = ++current.generation
      const access = captureGroupFileAccess($groupChats.get()[group])
      current.controller?.abort()
      const controller = new AbortController()
      current.controller = controller

      if (mode === 'latest') {
        current.deliveries.abort()
        current.deliveries = new AbortController()
      }

      publish({
        loading: true,
        failure: null,
        cursorExpired: false,
        ...(mode === 'latest' ? { pages: [], index: 0 } : {})
      })

      try {
        const data = await withGroupFileDeadline(
          loadPage(
            group,
            {
              limit: GROUP_FILES_PAGE_SIZE,
              ...(cursor ? { cursor } : {}),
              ...(before.query.trim() ? { query: before.query.trim() } : {})
            },
            controller.signal
          ),
          controller.signal
        )

        if (!model.current.open || generation !== model.current.generation) {
          return
        }

        if (!groupFileAccessCurrent(access)) {
          throw new Error('Files access changed during listing')
        }

        if (mode === 'older' && held) {
          validateGroupFilesContinuation(held.data, data)

          if (data.nextCursor && before.pages.some(page => page.data.nextCursor === data.nextCursor)) {
            throw Object.assign(new Error('attachment list cursor is invalid'), { code: 4143 })
          }

          const boundary = before.pages
            .slice(0, before.index + 1)
            .reverse()
            .find(page => page.data.items.length)?.data

          if (boundary && boundary !== held.data) {
            validateGroupFilesContinuation(boundary, data)
          }

          const seen = new Set(
            before.pages.flatMap(page =>
              page.data.items.map(item => item.key || `${item.eventId}:${item.attachment.attachmentId}`)
            )
          )

          if (data.items.some(item => seen.has(item.key || `${item.eventId}:${item.attachment.attachmentId}`))) {
            throw new Error('Invalid shared-files duplicate page')
          }
        }

        model.current.misses = 0
        model.current.availability = 'available'

        if (current.deliveries.signal.aborted) {
          current.deliveries = new AbortController()
        }

        publish({
          loading: false,
          offline: false,
          unavailable: false,
          reconnected: before.offline || before.reconnected,
          latestFileSeq: Math.max(before.latestFileSeq, data.latestFileSeq || 0),
          ...(mode === 'retry' && held
            ? {}
            : mode === 'older'
              ? { pages: [...before.pages.slice(0, before.index + 1), { data, cursor }], index: before.index + 1 }
              : { pages: [{ data }], index: 0 })
        })
      } catch (error) {
        if (!model.current.open || generation !== model.current.generation) {
          return
        }

        const failure = groupFileFailure(error)

        if (failure === 'access') {
          invalidateGroupFileAccess(access)
          denied()

          return
        }

        publish({
          loading: false,
          failure,
          cursorExpired: isGroupFilesCursorError(error),
          ...(failure === 'offline' || failure === 'timeout' ? { offline: true } : {})
        })
      } finally {
        controller.abort()
      }
    },
    [group, loadPage, publish, denied]
  )

  useEffect(() => {
    const current = model.current
    const previous = current.availability
    current.availability = availability
    current.open = open

    if (!open) {
      return
    }

    if (availability === 'unavailable') {
      current.generation += 1
      current.controller?.abort()
      current.deliveries.abort()
      publish({ unavailable: true, loading: false })
    } else if (availability === 'offline') {
      current.misses += 1

      if (!current.state.pages.length || current.misses >= 2) {
        publish({ offline: true, reconnected: false })
      }
    } else {
      current.misses = 0
      publish({ offline: false, unavailable: false, reconnected: current.state.offline || current.state.reconnected })

      if (previous !== 'available' && !current.state.loading) {
        if (previous === 'unavailable') {
          void fetchPage(current.state.pages.length ? 'retry' : 'latest')
        } else if (!current.state.pages.length) {
          void fetchPage('latest')
        }
      }
    }
  }, [availability, observation, open, fetchPage, publish])

  useEffect(() => {
    const current = model.current
    current.generation += 1
    current.open = open

    if (!open) {
      publish({ pages: [], index: 0, query: '', loading: false, failure: null, latestFileSeq: 0, reconnected: false })
      current.controller?.abort()
      current.deliveries.abort()

      return
    }

    if (current.state.offline || current.state.unavailable) {
      publish({ loading: false })

      return
    }

    publish({ loading: true, failure: null })
    const timer = setTimeout(() => void fetchPage('latest'), state.query.trim() ? 250 : 0)

    return () => {
      clearTimeout(timer)
      current.generation += 1
      current.controller?.abort()
      current.deliveries.abort()
    }
  }, [open, state.query, fetchPage, publish])

  useEffect(() => {
    const current = model.current

    return () => {
      current.open = false
      current.generation += 1
      current.controller?.abort()
      current.deliveries.abort()
    }
  }, [])

  const query = (value: string) => {
    const next = [...value].slice(0, 255).join('')

    if (next === model.current.state.query) {
      return
    }

    model.current.generation += 1
    model.current.controller?.abort()
    model.current.deliveries.abort()
    model.current.deliveries = new AbortController()
    publish({
      query: next,
      pages: [],
      index: 0,
      failure: null,
      loading: true,
      cursorExpired: false
    })
  }

  const older = () => {
    const current = model.current.state

    if (current.loading) {
      return
    }

    if (current.pages[current.index + 1]) {
      publish({ index: current.index + 1 })
    } else if (current.pages[current.index]?.data.hasMore) {
      void fetchPage('older')
    }
  }

  const newer = () => {
    if (!model.current.state.loading) {
      publish({ index: Math.max(0, model.current.state.index - 1) })
    }
  }

  return {
    ...state,
    page: state.pages[state.index]?.data,
    setQuery: query,
    invalidateAccess,
    cancel,
    deliverySignal: model.current.deliveries.signal,
    older,
    newer,
    retry: () => void fetchPage('retry'),
    latest: () => void fetchPage('latest')
  }
}
