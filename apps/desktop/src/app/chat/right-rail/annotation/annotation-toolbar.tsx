import { useEffect, useRef, useState } from 'react'

import { X } from '@/lib/icons'
import { cn } from '@/lib/utils'

export interface AnnotationListEntry {
  commentPreview: string
  id: string
  number: number
  summary: string
}

interface AnnotationToolbarProps {
  copy: {
    cancel: string
    finish: string
    locate: string
    remove: string
    title: (count: number) => string
  }
  items: AnnotationListEntry[]
  onCancel: () => void
  onFinish: () => void
  onLocate: (id: string) => void
  onRemove: (id: string) => void
}

const CIRCLED = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']

/**
 * Floating pill shown at the top of the preview while annotation mode is
 * active. The count segment expands into the annotation list: click an entry
 * to flash its target on the page, or remove it inline.
 */
export function AnnotationToolbar({ copy, items, onCancel, onFinish, onLocate, onRemove }: AnnotationToolbarProps) {
  const [listOpen, setListOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  // Collapse the list when clicking anywhere outside the toolbar.
  useEffect(() => {
    if (!listOpen) {
      return
    }

    const onPointerDown = (event: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setListOpen(false)
      }
    }

    window.addEventListener('pointerdown', onPointerDown, true)
    return () => window.removeEventListener('pointerdown', onPointerDown, true)
  }, [listOpen])

  return (
    <div className="pointer-events-none absolute inset-x-0 top-3 z-30 flex justify-center">
      <div className="relative" ref={rootRef}>
        <div
          className={cn(
            'pointer-events-auto flex items-center gap-3 rounded-full py-1.5 pl-4 pr-1.5',
            'border border-white/10 bg-neutral-900/85 text-neutral-100 shadow-lg shadow-black/30',
            'backdrop-blur-md backdrop-saturate-150',
            'animate-in fade-in slide-in-from-top-2 duration-200'
          )}
          data-slot="annotation-toolbar"
        >
          <button
            className={cn(
              'flex items-center gap-2 rounded-full px-1.5 py-0.5 text-xs font-medium transition-colors',
              items.length > 0 && 'hover:bg-white/10',
              listOpen && 'bg-white/10'
            )}
            disabled={items.length === 0}
            onClick={() => setListOpen(open => !open)}
            type="button"
          >
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" />
            {copy.title(items.length)}
            {items.length > 0 && (
              <span className={cn('text-[0.6rem] text-neutral-400 transition-transform', listOpen && 'rotate-180')}>▾</span>
            )}
          </button>

          <div className="flex items-center gap-1">
            <button
              className="rounded-full px-3 py-1 text-xs text-neutral-400 transition-colors hover:bg-white/10 hover:text-neutral-200"
              onClick={onCancel}
              type="button"
            >
              {copy.cancel}
            </button>
            <button
              className="rounded-full bg-red-500 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={items.length === 0}
              onClick={onFinish}
              type="button"
            >
              {copy.finish}
            </button>
          </div>
        </div>

        {listOpen && items.length > 0 && (
          <div
            className={cn(
              'pointer-events-auto absolute left-1/2 top-full z-30 mt-2 w-72 -translate-x-1/2',
              'rounded-2xl border border-white/10 bg-neutral-900/90 shadow-2xl shadow-black/40',
              'backdrop-blur-xl backdrop-saturate-150',
              'animate-in fade-in slide-in-from-top-1 zoom-in-95 duration-150'
            )}
            data-testid="annotation-list"
          >
            <ul className="max-h-64 overflow-y-auto p-1.5">
              {items.map(item => (
                <li
                  className="group flex items-center gap-2 rounded-xl px-2 py-1.5 transition-colors hover:bg-white/5"
                  key={item.id}
                >
                  <button
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    onClick={() => onLocate(item.id)}
                    title={copy.locate}
                    type="button"
                  >
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red-500 text-[0.65rem] font-semibold text-white">
                      {CIRCLED[item.number - 1] || item.number}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-mono text-[0.6875rem] text-neutral-200">{item.summary}</span>
                      {item.commentPreview && (
                        <span className="block truncate text-[0.6875rem] text-neutral-500">{item.commentPreview}</span>
                      )}
                    </span>
                  </button>
                  <button
                    className="shrink-0 rounded-full p-1 text-neutral-500 opacity-0 transition-all hover:bg-white/10 hover:text-neutral-200 group-hover:opacity-100"
                    onClick={() => onRemove(item.id)}
                    title={copy.remove}
                    type="button"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
