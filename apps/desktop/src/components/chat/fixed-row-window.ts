import type { RefObject, UIEvent } from 'react'
import { useCallback, useLayoutEffect, useRef, useState } from 'react'

export interface LineChunk<T> {
  lines: T[]
  start: number
}

export interface TextLineChunk extends LineChunk<string> {
  text: string
}

interface FixedRowWindowOptions {
  overscanRows: number
  rowPx: number
  rowsPerChunk: number
  totalRows: number
}

export interface FixedRowWindow {
  afterRows: number
  beforeRows: number
  endChunk: number
  onScroll: (event: UIEvent<HTMLDivElement>) => void
  scrollerRef: RefObject<HTMLDivElement | null>
  startChunk: number
}

export function chunkLines<T>(lines: T[], perChunk: number): Array<LineChunk<T>> {
  if (lines.length <= perChunk) {
    return [{ lines, start: 0 }]
  }

  const chunks: Array<LineChunk<T>> = []

  for (let start = 0; start < lines.length; start += perChunk) {
    chunks.push({ lines: lines.slice(start, start + perChunk), start })
  }

  return chunks
}

export function chunkTextLines(text: string, perChunk: number): TextLineChunk[] {
  return chunkLines(text.split('\n'), perChunk).map(chunk => ({
    ...chunk,
    text: chunk.lines.join('\n')
  }))
}

type ChunkWindow = Pick<FixedRowWindow, 'afterRows' | 'beforeRows' | 'endChunk' | 'startChunk'>

export function useFixedRowWindow({
  overscanRows,
  rowPx,
  rowsPerChunk,
  totalRows
}: FixedRowWindowOptions): FixedRowWindow {
  const scrollerRef = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number | null>(null)

  // Derive the visible chunk window from a node's scroll geometry. Pure so we
  // can compare results and skip a re-render unless the window actually moved.
  const compute = useCallback(
    (node: HTMLDivElement | null): ChunkWindow => {
      const height = node?.clientHeight || 800
      const scrollTop = node?.scrollTop ?? 0
      const firstRow = Math.max(0, Math.floor(scrollTop / rowPx) - overscanRows)
      const lastRow = Math.min(totalRows, Math.ceil((scrollTop + height) / rowPx) + overscanRows)
      const startChunk = Math.floor(firstRow / rowsPerChunk)
      const endChunk = Math.max(startChunk, Math.floor(Math.max(firstRow, lastRow - 1) / rowsPerChunk))

      return {
        afterRows: Math.max(0, totalRows - Math.min(totalRows, (endChunk + 1) * rowsPerChunk)),
        beforeRows: Math.min(totalRows, startChunk * rowsPerChunk),
        endChunk,
        startChunk
      }
    },
    [overscanRows, rowPx, rowsPerChunk, totalRows]
  )

  const [win, setWin] = useState<ChunkWindow>(() => compute(null))

  // Only commit a new window when a boundary is crossed — scrolling within the
  // current chunk span (the common case, every rAF) keeps the same object and
  // re-renders nothing.
  const sync = useCallback(
    (node: HTMLDivElement | null = scrollerRef.current) => {
      if (!node) {
        return
      }

      const next = compute(node)

      setWin(prev =>
        prev.startChunk === next.startChunk &&
        prev.endChunk === next.endChunk &&
        prev.beforeRows === next.beforeRows &&
        prev.afterRows === next.afterRows
          ? prev
          : next
      )
    },
    [compute]
  )

  const cancelFrame = useCallback(() => {
    if (rafRef.current == null) {
      return
    }

    cancelAnimationFrame(rafRef.current)
    rafRef.current = null
  }, [])

  const onScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      const node = event.currentTarget

      cancelFrame()
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null
        sync(node)
      })
    },
    [cancelFrame, sync]
  )

  // Re-sync on mount, on resize, and whenever the row geometry changes (new
  // file/diff → `compute` identity changes → effect re-runs).
  useLayoutEffect(() => {
    const node = scrollerRef.current

    if (!node) {
      return
    }

    sync(node)

    if (typeof ResizeObserver === 'undefined') {
      return cancelFrame
    }

    const observer = new ResizeObserver(() => sync(node))

    observer.observe(node)

    return () => {
      observer.disconnect()
      cancelFrame()
    }
  }, [cancelFrame, sync])

  return { ...win, onScroll, scrollerRef }
}
