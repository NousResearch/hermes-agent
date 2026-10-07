import './butterbar.css'

import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $butterbarItems, type ButterbarItem, dismissButterbar, isButterbarCloseable } from '@/store/butterbar'

const DEFAULT_DWELL_MS = 3000

/** Full-width notice strip above the statusbar. Reads the butterbar store, so
 *  any module can put a notice here with `registerButterbar` — no props. One
 *  item shows bare; several rotate with dots and pause while hovered. */
export function Butterbar() {
  const items = useStore($butterbarItems)
  const [activeId, setActiveId] = useState<null | string>(null)
  const [paused, setPaused] = useState(false)
  const index = Math.max(0, items.findIndex(item => item.id === activeId))
  const item = items[index]
  const multiple = items.length > 1

  useEffect(() => {
    if (!multiple || paused || !item) {
      return
    }

    const timer = window.setTimeout(
      () => setActiveId(items[(index + 1) % items.length].id),
      item.durationMs ?? DEFAULT_DWELL_MS
    )

    return () => window.clearTimeout(timer)
  }, [index, item, items, multiple, paused])

  if (!item) {
    return null
  }

  return (
    <div
      className="grid h-6 shrink-0 grid-cols-[1fr_minmax(0,max-content)_1fr] items-center gap-2 px-2.5 text-[0.6875rem] leading-4 [-webkit-app-region:no-drag]"
      data-slot="butterbar"
      data-tone={item.tone ?? 'accent'}
      data-variant={item.variant ?? 'soft'}
      onBlur={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onPointerEnter={() => setPaused(true)}
      onPointerLeave={() => setPaused(false)}
      role="region"
      style={item.color ? ({ '--bb-tint': item.color } as CSSProperties) : undefined}
    >
      <span aria-hidden />
      <div
        aria-live={multiple ? 'off' : 'polite'}
        className="butterbar-slide flex min-w-0 items-center justify-center gap-1.5"
        key={multiple ? item.id : undefined}
      >
        {item.icon && <span className="flex shrink-0 items-center opacity-80">{item.icon}</span>}
        <div className="min-w-0 truncate">{item.node}</div>
      </div>
      <div className="flex min-w-0 items-center justify-end gap-1">
        {multiple && <ButterbarDots activeIndex={index} items={items} onSelect={setActiveId} />}
        {isButterbarCloseable(item) && <ButterbarClose item={item} />}
      </div>
    </div>
  )
}

function ButterbarDots({
  activeIndex,
  items,
  onSelect
}: {
  activeIndex: number
  items: readonly ButterbarItem[]
  onSelect: (id: string) => void
}) {
  const copy = useI18n().t.butterbar

  return (
    <div className="flex shrink-0 items-center">
      {items.map((entry, i) => (
        <button
          aria-current={i === activeIndex}
          aria-label={copy.goTo(i + 1, items.length)}
          className="butterbar-dot grid h-6 cursor-pointer place-items-center px-[0.1875rem]"
          key={entry.id}
          onClick={() => onSelect(entry.id)}
          type="button"
        >
          <span />
        </button>
      ))}
    </div>
  )
}

function ButterbarClose({ item }: { item: ButterbarItem }) {
  const { t } = useI18n()

  return (
    <Button
      aria-label={t.common.close}
      className="-mr-1 size-5 text-(--bb-fg) hover:bg-(--bb-hover) hover:text-(--bb-link)"
      onClick={() => dismissButterbar(item)}
      size="icon-xs"
      type="button"
      variant="ghost"
    >
      <Codicon name="close" size="0.75rem" />
    </Button>
  )
}
