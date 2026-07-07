'use client'

import { type CSSProperties, useId, useMemo, useState } from 'react'

import {
  type HermesUiChecklistArtifact,
  parseHermesUiArtifact,
  stableArtifactStorageKey
} from '@/lib/hermes-ui-artifacts'
import { readKey, writeKey } from '@/lib/storage'
import { cn } from '@/lib/utils'

import type { RichFenceProps } from './types'

function readChecklistState(key: string, itemIds: ReadonlySet<string>): Record<string, boolean> {
  const raw = readKey(key)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed)
        .filter((entry): entry is [string, boolean] => itemIds.has(entry[0]) && typeof entry[1] === 'boolean')
        .map(([id, checked]) => [id, checked])
    )
  } catch {
    return {}
  }
}

function persistChecklistState(key: string, itemIds: readonly string[], state: Record<string, boolean>) {
  writeKey(
    key,
    JSON.stringify(Object.fromEntries(itemIds.map(id => [id, Boolean(state[id])] as const)))
  )
}

function hasRtlText(value: string | undefined): boolean {
  return /[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]/.test(value || '')
}

function checklistDirection(artifact: HermesUiChecklistArtifact): 'ltr' | 'rtl' {
  if (artifact.direction === 'rtl' || artifact.direction === 'ltr') {
    return artifact.direction
  }

  const sample = [artifact.title, artifact.description, ...artifact.items.flatMap(item => [item.label, item.description])]
    .filter(Boolean)
    .join('\n')

  return hasRtlText(sample) ? 'rtl' : 'ltr'
}

function splitTextBlocks(value: string): string[] {
  return value
    .split(/\n{2,}/)
    .map(part => part.trim())
    .filter(Boolean)
}

function PlainTextBlocks({ className, text }: { className?: string; text: string }) {
  const blocks = splitTextBlocks(text)

  if (blocks.length <= 1) {
    return (
      <span className={className} dir="auto">
        {text}
      </span>
    )
  }

  return (
    <span className={cn('block space-y-1.5', className)} dir="auto">
      {blocks.map((block, index) => (
        <span className="block" key={`${index}:${block.slice(0, 16)}`}>
          {block}
        </span>
      ))}
    </span>
  )
}

export function ChecklistArtifactCard({ artifact }: { artifact: HermesUiChecklistArtifact }) {
  const reactId = useId()
  const itemIds = useMemo(() => artifact.items.map(item => item.id), [artifact.items])
  const itemIdSet = useMemo(() => new Set(itemIds), [itemIds])
  const storageKey = useMemo(() => stableArtifactStorageKey(artifact), [artifact])
  const [checked, setChecked] = useState<Record<string, boolean>>(() => readChecklistState(storageKey, itemIdSet))
  const checkedCount = itemIds.reduce((total, id) => total + (checked[id] ? 1 : 0), 0)
  const percent = itemIds.length === 0 ? 0 : (checkedCount / itemIds.length) * 100
  const direction = checklistDirection(artifact)
  const isRtl = direction === 'rtl'
  const directionalStyle = { direction, textAlign: isRtl ? 'right' : 'left' } satisfies CSSProperties

  const updateChecked = (next: Record<string, boolean>) => {
    setChecked(next)
    persistChecklistState(storageKey, itemIds, next)
  }

  return (
    <section
      aria-label={artifact.title || 'Interactive checklist'}
      className={cn(
        'my-3 overflow-hidden rounded-xl border border-border/80 bg-muted/25 shadow-[0_0.0625rem_0.125rem_color-mix(in_srgb,#000_10%,transparent)]',
        isRtl ? 'text-right' : 'text-left'
      )}
      data-hermes-ui-artifact="checklist"
      dir={direction}
      style={directionalStyle}
    >
      <div className="border-b border-border/65 px-3 py-2.5">
        <div className={cn('flex min-w-0 items-start justify-between gap-3', isRtl && 'flex-row-reverse')}>
          <div className="min-w-0">
            {artifact.title && (
              <h3 className="m-0 text-[0.8125rem] leading-snug font-semibold text-foreground" dir={direction} style={directionalStyle}>
                {artifact.title}
              </h3>
            )}
            {artifact.description && (
              <p className="m-0 mt-1 text-[0.75rem] leading-relaxed text-muted-foreground" dir={direction} style={directionalStyle}>
                <PlainTextBlocks text={artifact.description} />
              </p>
            )}
          </div>
          <span className="shrink-0 rounded-md border border-border/70 bg-background/45 px-1.5 py-0.5 text-[0.6875rem] leading-none font-medium text-muted-foreground tabular-nums">
            {checkedCount} / {itemIds.length}
          </span>
        </div>
        <div
          aria-label={`${checkedCount} of ${itemIds.length} checklist items complete`}
          aria-valuemax={itemIds.length}
          aria-valuemin={0}
          aria-valuenow={checkedCount}
          className="mt-2 h-1.5 overflow-hidden rounded-full bg-background/70"
          role="progressbar"
        >
          <div className="h-full rounded-full bg-foreground/70 transition-[width]" style={{ width: `${percent}%` }} />
        </div>
      </div>
      <div className="divide-y divide-border/45">
        {artifact.items.map(item => {
          const inputId = `${reactId}-${item.id}`

          return (
            <div
              className={cn('flex gap-2.5 px-3 py-2.5', isRtl && 'flex-row-reverse')}
              dir={direction}
              key={item.id}
              style={directionalStyle}
            >
              <input
                checked={Boolean(checked[item.id])}
                className="mt-0.5 size-4 shrink-0 accent-foreground"
                id={inputId}
                onChange={event => updateChecked({ ...checked, [item.id]: event.currentTarget.checked })}
                type="checkbox"
              />
              <div className="min-w-0 flex-1">
                <label
                  className={cn(
                    'block cursor-pointer whitespace-pre-wrap text-[0.8125rem] leading-relaxed text-foreground wrap-anywhere',
                    checked[item.id] && 'text-muted-foreground line-through decoration-muted-foreground/50'
                  )}
                  dir={direction}
                  htmlFor={inputId}
                  style={directionalStyle}
                >
                  <PlainTextBlocks text={item.label} />
                </label>
                {item.description && (
                  <p
                    className="m-0 mt-1 text-[0.75rem] leading-relaxed text-muted-foreground wrap-anywhere"
                    dir={direction}
                    style={directionalStyle}
                  >
                    <PlainTextBlocks text={item.description} />
                  </p>
                )}
              </div>
            </div>
          )
        })}
      </div>
      <div className={cn('flex items-center gap-2 border-t border-border/65 px-3 py-2', isRtl && 'justify-end')}>
        <button
          className="rounded-md border border-border/80 bg-background/45 px-2 py-1 text-[0.75rem] font-medium text-foreground hover:bg-muted/70"
          onClick={() => updateChecked(Object.fromEntries(itemIds.map(id => [id, true] as const)))}
          type="button"
        >
          {isRtl ? 'סמן הכול' : 'Mark all'}
        </button>
        <button
          className="rounded-md border border-border/80 bg-transparent px-2 py-1 text-[0.75rem] font-medium text-muted-foreground hover:bg-muted/60 hover:text-foreground"
          onClick={() => updateChecked(Object.fromEntries(itemIds.map(id => [id, false] as const)))}
          type="button"
        >
          {isRtl ? 'נקה' : 'Clear'}
        </button>
      </div>
    </section>
  )
}

export default function HermesUiArtifactRenderer({ code }: RichFenceProps) {
  const result = parseHermesUiArtifact(code)

  if (!result.ok) {
    return null
  }

  if (result.artifact.type === 'checklist') {
    return <ChecklistArtifactCard artifact={result.artifact} />
  }

  return null
}
