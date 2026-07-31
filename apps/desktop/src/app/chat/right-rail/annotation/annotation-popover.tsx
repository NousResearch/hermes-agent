import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import type { PickedElement, PickedRegion } from './element-picker'

export interface AnnotationDraft {
  comment: string
  kind: 'element' | 'region'
  screenshotDataUrl?: string
  target: PickedElement | PickedRegion
}

interface AnnotationPopoverProps {
  /** 1-based badge number this annotation will get when added. */
  number: number
  onAdd: (draft: AnnotationDraft) => void
  onDiscard: () => void
  screenshotDataUrl?: string
  target: PickedElement | PickedRegion
  kind: 'element' | 'region'
}

function isPickedElement(target: PickedElement | PickedRegion): target is PickedElement {
  return 'selector' in target
}

function targetSummary(target: PickedElement | PickedRegion): string {
  if (isPickedElement(target)) {
    const text = target.text ? ` "${target.text}"` : ''
    return `<${target.tagName.toLowerCase()}>${target.id ? ` #${target.id}` : ''}${text}`
  }

  return `${Math.round(target.rect.width)}×${Math.round(target.rect.height)}px`
}

const CIRCLED = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']

/**
 * Comment card for one pending annotation in session mode. Dark Linear-style
 * glass card anchored at the bottom of the preview: number badge, target
 * summary, screenshot thumb, and add/discard actions.
 */
export function AnnotationPopover({ kind, number, onAdd, onDiscard, screenshotDataUrl, target }: AnnotationPopoverProps) {
  const { t } = useI18n()
  const copy = t.preview.web.annotation
  const [comment, setComment] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onDiscard()
      }

      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.stopPropagation()
        onAdd({ comment, kind, screenshotDataUrl, target })
      }
    }

    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [comment, kind, onAdd, onDiscard, screenshotDataUrl, target])

  const marker = CIRCLED[number - 1] || `(${number})`

  return (
    <div
      className={cn(
        'pointer-events-auto absolute bottom-4 left-1/2 z-50 w-[24rem] -translate-x-1/2',
        'rounded-2xl border border-white/10 bg-neutral-900/90 shadow-2xl shadow-black/40',
        'backdrop-blur-xl backdrop-saturate-150',
        'animate-in fade-in slide-in-from-bottom-3 zoom-in-95 duration-200'
      )}
      data-testid="annotation-popover"
    >
      <div className="flex items-center gap-2.5 px-4 pb-2 pt-3">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-red-500 text-xs font-semibold text-white">
          {marker}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-medium text-neutral-100">
            {kind === 'element' ? copy.elementTitle : copy.regionTitle}
          </div>
          <div className="truncate font-mono text-[0.6875rem] text-neutral-400">{targetSummary(target)}</div>
        </div>
        {screenshotDataUrl && (
          <img
            alt={copy.screenshotAlt}
            className="h-10 w-16 shrink-0 rounded-md border border-white/10 object-cover"
            src={screenshotDataUrl}
          />
        )}
      </div>

      <div className="px-4 pb-3">
        <textarea
          ref={textareaRef}
          className={cn(
            'min-h-20 w-full resize-none rounded-xl border border-white/10 bg-white/5 px-3 py-2',
            'text-sm text-neutral-100 placeholder:text-neutral-500',
            'outline-none transition-colors focus:border-red-500/50 focus:bg-white/[0.07]'
          )}
          onChange={event => setComment(event.target.value)}
          placeholder={copy.placeholder}
          value={comment}
        />
        <div className="mt-2.5 flex items-center justify-between">
          <span className="text-[0.6875rem] text-neutral-500">{copy.hint}</span>
          <div className="flex gap-2">
            <button
              className="rounded-full px-3 py-1.5 text-xs text-neutral-400 transition-colors hover:bg-white/10 hover:text-neutral-200"
              onClick={onDiscard}
              type="button"
            >
              {copy.discard}
            </button>
            <button
              className="rounded-full bg-red-500 px-4 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-red-400"
              onClick={() => onAdd({ comment, kind, screenshotDataUrl, target })}
              type="button"
            >
              {copy.add}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
