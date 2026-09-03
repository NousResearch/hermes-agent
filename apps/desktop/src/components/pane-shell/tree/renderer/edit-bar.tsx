/**
 * Edit-mode palette — the floating, draggable "Layouts" card shown while
 * layout edit mode is on. It hosts the layout picker and the reset/done
 * actions; its header doubles as the drag handle. Position survives edit-mode
 * toggles within a session.
 */

import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useCallback, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { $bindings, bindingsFor } from '@/store/keybinds'

import { $layoutEditMode } from '../../edit-mode'
import { resetLayoutTree } from '../store'

import { LayoutPicker } from './layout-picker'

// Palette position survives edit-mode toggles within a session; null = centered.
let lastPalettePos: { x: number; y: number } | null = null

export function TreeEditBar() {
  const { t } = useI18n()
  const editMode = useStore($layoutEditMode)
  const bindings = useStore($bindings)
  const [pos, setPos] = useState(lastPalettePos)
  const cardRef = useRef<HTMLDivElement>(null)
  // The toggle is a `keybinds` contribution (`layout.editMode`) — show the
  // user's LIVE binding, not a hardcoded chord.
  const toggleCombo = bindingsFor('layout.editMode', bindings)[0]

  const startMove = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    if (e.button !== 0) {
      return
    }

    const card = cardRef.current
    const parent = card?.parentElement

    if (!card || !parent) {
      return
    }

    e.preventDefault()

    const cardRect = card.getBoundingClientRect()
    const dx = e.clientX - cardRect.x
    const dy = e.clientY - cardRect.y

    const onMove = (ev: PointerEvent) => {
      const parentRect = parent.getBoundingClientRect()

      const next = {
        x: Math.max(4, Math.min(parentRect.width - cardRect.width - 4, ev.clientX - parentRect.x - dx)),
        y: Math.max(4, Math.min(parentRect.height - 40, ev.clientY - parentRect.y - dy))
      }

      lastPalettePos = next
      setPos(next)
    }

    const onUp = () => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
  }, [])

  if (!editMode) {
    return null
  }

  return (
    <div
      className="absolute z-50 flex w-[26rem] max-w-[calc(100%-2rem)] flex-col rounded-xl border border-(--ui-stroke-secondary) bg-popover text-popover-foreground shadow-2xl [-webkit-app-region:no-drag]"
      ref={cardRef}
      style={pos ? { left: pos.x, top: pos.y } : { left: '50%', top: '50%', transform: 'translate(-50%, -50%)' }}
    >
      {/* Header doubles as the drag handle (Panel-style title + actions). */}
      <header
        className="flex shrink-0 cursor-grab select-none items-start justify-between gap-3 px-4 pb-2 pt-3 active:cursor-grabbing"
        onPointerDown={startMove}
      >
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">{t.zones.editTitle}</h2>
          <p className="text-xs text-muted-foreground/80">
            {t.zones.editHint}{' '}
            {toggleCombo && (
              <kbd className="rounded border border-(--ui-stroke-secondary) bg-foreground/5 px-1 font-mono text-[10px]">
                {formatCombo(toggleCombo)}
              </kbd>
            )}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5" onPointerDown={e => e.stopPropagation()}>
          <Button onClick={resetLayoutTree} size="sm" variant="ghost">
            {t.zones.reset}
          </Button>
          <Button onClick={() => $layoutEditMode.set(false)} size="sm" variant="outline">
            {t.common.done}
          </Button>
        </div>
      </header>
      <div className="px-4 pb-4">
        <LayoutPicker />
      </div>
    </div>
  )
}
