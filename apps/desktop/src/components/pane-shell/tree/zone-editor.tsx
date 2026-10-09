/**
 * Zone editor — the FancyZones grid editor experience, ported from PowerToys'
 * `GridEditor.xaml.cs` interactions onto the verbatim model port (grid-model.ts):
 *
 *  - A full-screen canvas of numbered translucent zones.
 *  - A splitter line follows the cursor inside the hovered zone; CLICK splits
 *    at that position. Hold SHIFT to flip the line horizontal (row split).
 *  - DRAG across zones to rubber-band select; the selection expands to its
 *    rectangular closure (ComputeClosure) and a Merge button appears.
 *  - Shared zone edges are draggable resizer thumbs (multi-zone edges move
 *    together, min-size clamped) — GridData.Drag semantics.
 *  - Templates: Columns / Rows / Grid / Priority with a zone-count stepper.
 *
 * Saving converts the grid to our runtime layout tree (guillotine cuts) and
 * registers it as a user preset; non-guillotine arrangements (pinwheels)
 * disable Save with an explanation.
 */

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { registry } from '@/contrib/registry'
import { useI18n } from '@/i18n'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { cn } from '@/lib/utils'

import {
  canSplit,
  doMerge,
  dragResizer,
  type GridLayout,
  type GridResizer,
  type GridZone,
  initColumns,
  initGrid,
  initPriorityGrid,
  initRows,
  mergeClosureIndices,
  modelToResizers,
  modelToZones,
  MULTIPLIER,
  splitZone
} from './grid-model'
import { gridIsTreeExpressible, gridToTree, type PanePlacementHint } from './grid-to-tree'
import { allPaneIds } from './model'
import { applyLayoutPreset, saveLayoutPresetTree } from './presets'
import { $layoutTree } from './store'

export const $zoneEditorOpen = atom(false)

const SPLIT_SNAP = 50 // model units (0.5%)

const pct = (v: number) => `${(v / MULTIPLIER) * 100}%`

interface SplitPreview {
  zoneIndex: number
  orientation: 'horizontal' | 'vertical'
  position: number
}

interface SelectBox {
  x0: number
  y0: number
  x1: number
  y1: number
}

/** Resizer screen geometry derived from its zones (model units). */
function resizerGeometry(resizer: GridResizer, zones: GridZone[]) {
  const all = [...resizer.negativeSideIndices, ...resizer.positiveSideIndices].map(i => zones[i])

  if (resizer.orientation === 'horizontal') {
    return {
      at: zones[resizer.positiveSideIndices[0]].top,
      from: Math.min(...all.map(z => z.left)),
      to: Math.max(...all.map(z => z.right))
    }
  }

  return {
    at: zones[resizer.positiveSideIndices[0]].left,
    from: Math.min(...all.map(z => z.top)),
    to: Math.max(...all.map(z => z.bottom))
  }
}

export function ZoneEditor() {
  const { t } = useI18n()
  const open = useStore($zoneEditorOpen)
  const [model, setModel] = useState<GridLayout>(() => initPriorityGrid(3))
  const [templateCount, setTemplateCount] = useState(3)
  const [shift, setShift] = useState(false)
  const [splitPreview, setSplitPreview] = useState<SplitPreview | null>(null)
  const [selectBox, setSelectBox] = useState<SelectBox | null>(null)
  const [selection, setSelection] = useState<number[]>([])
  const [mergeAt, setMergeAt] = useState<{ x: number; y: number } | null>(null)
  const [name, setName] = useState('')
  const canvasRef = useRef<HTMLDivElement>(null)

  const zones = modelToZones(model) ?? []
  const resizers = modelToResizers(model)
  const treeExpressible = gridIsTreeExpressible(model)

  // Shift flips the splitter orientation (FancyZones behavior); Esc closes.
  useEffect(() => {
    if (!open) {
      return
    }

    const releaseLayer = pushEscapeLayer(ESCAPE_PRIORITY.zoneEditor)

    const down = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        setShift(true)
      }

      // Skip when a nested field (the name Input) already handled Escape, or a
      // higher layer owns it.
      if (e.key === 'Escape' && !e.defaultPrevented && isTopEscapeLayer(ESCAPE_PRIORITY.zoneEditor)) {
        e.preventDefault()
        $zoneEditorOpen.set(false)
      }
    }

    const up = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        setShift(false)
      }
    }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)

    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      releaseLayer()
    }
  }, [open])

  const toModelPoint = useCallback((clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect()

    return {
      x: Math.round(((clientX - rect.x) / rect.width) * MULTIPLIER),
      y: Math.round(((clientY - rect.y) / rect.height) * MULTIPLIER)
    }
  }, [])

  if (!open) {
    return null
  }

  const zoneAt = (x: number, y: number) =>
    zones.find(z => x >= z.left && x < z.right && y >= z.top && y < z.bottom) ?? null

  const updateSplitPreview = (clientX: number, clientY: number) => {
    const p = toModelPoint(clientX, clientY)
    const zone = zoneAt(p.x, p.y)

    if (!zone) {
      setSplitPreview(null)

      return
    }

    const orientation = shift ? 'horizontal' : 'vertical'
    const raw = orientation === 'horizontal' ? p.y : p.x
    const position = Math.round(raw / SPLIT_SNAP) * SPLIT_SNAP

    setSplitPreview(
      canSplit(model, zone.index, position, orientation) ? { zoneIndex: zone.index, orientation, position } : null
    )
  }

  const onCanvasPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || (e.target as HTMLElement).dataset.resizer !== undefined) {
      return
    }

    e.preventDefault()
    setMergeAt(null)
    setSelection([])

    const start = toModelPoint(e.clientX, e.clientY)
    let dragged = false

    const onMove = (ev: PointerEvent) => {
      const cur = toModelPoint(ev.clientX, ev.clientY)

      if (!dragged && Math.hypot(cur.x - start.x, cur.y - start.y) < 60) {
        return
      }

      dragged = true

      const box = {
        x0: Math.min(start.x, cur.x),
        y0: Math.min(start.y, cur.y),
        x1: Math.max(start.x, cur.x),
        y1: Math.max(start.y, cur.y)
      }

      setSelectBox(box)

      // Zones intersecting the rubber band, expanded to rectangular closure.
      const picked = zones
        .filter(z => z.left < box.x1 && z.right > box.x0 && z.top < box.y1 && z.bottom > box.y0)
        .map(z => z.index)

      setSelection(mergeClosureIndices(model, picked))
    }

    const onUp = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
      setSelectBox(null)

      if (!dragged) {
        // Plain click: split at the previewed line.
        if (splitPreview) {
          setModel(m => splitZone(m, splitPreview.zoneIndex, splitPreview.position, splitPreview.orientation))
          setSplitPreview(null)
        }

        setSelection([])

        return
      }

      const rect = canvasRef.current!.getBoundingClientRect()
      setMergeAt({ x: ev.clientX - rect.x, y: ev.clientY - rect.y })
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
  }

  const startResizerDrag = (index: number, e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()

    const rect = canvasRef.current!.getBoundingClientRect()
    const resizer = resizers[index]
    const horizontal = resizer.orientation === 'horizontal'
    const start = horizontal ? e.clientY : e.clientX
    let applied = 0

    const onMove = (ev: PointerEvent) => {
      const px = (horizontal ? ev.clientY : ev.clientX) - start
      const total = Math.round((px / (horizontal ? rect.height : rect.width)) * MULTIPLIER)
      const step = total - applied

      if (step !== 0) {
        setModel(m => {
          const next = dragResizer(m, index, step)

          if (next !== m) {
            applied = total
          }

          return next
        })
      }
    }

    const onUp = () => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
  }

  const merge = () => {
    setModel(m => doMerge(m, selection))
    setSelection([])
    setMergeAt(null)
  }

  const save = () => {
    const paneIds = allPaneIds($layoutTree.get() ?? { type: 'group', id: 'tmp', panes: [], active: '' })
    // Placement hints ride on the pane contributions (`data.placement`), so
    // zones are assigned by ROLE (main/left/right/bottom), not index order.
    const contributions = registry.getArea('panes')

    const placed = paneIds.map(id => ({
      id,
      placement: (contributions.find(c => c.id === id)?.data as { placement?: PanePlacementHint } | undefined)
        ?.placement
    }))

    const tree = gridToTree(model, placed)

    if (!tree) {
      return
    }

    const id = saveLayoutPresetTree(name || t.zones.customZoneName(zones.length), tree)

    if (id) {
      applyLayoutPreset(id, tree)
    }

    $zoneEditorOpen.set(false)
  }

  const templates = [
    { label: t.zones.templateColumns, make: initColumns },
    { label: t.zones.templateRows, make: initRows },
    { label: t.zones.templateGrid, make: initGrid },
    { label: t.zones.templatePriority, make: initPriorityGrid }
  ]

  return (
    <div
      className="absolute inset-0 z-[70] flex flex-col gap-3 p-6 [-webkit-app-region:no-drag]"
      style={{ background: 'color-mix(in srgb, var(--ui-bg-chrome) 88%, transparent)', backdropFilter: 'blur(6px)' }}
    >
      {/* Toolbar — Panel-style title + hint, template chooser on the right. */}
      <div className="flex items-end justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">{t.zones.zoneEditorTitle}</h2>
          <p className="text-xs text-muted-foreground/80">
            {t.zones.editorHintPre}
            <kbd className="rounded border border-(--ui-stroke-secondary) bg-foreground/5 px-1 font-mono text-[10px]">
              ⇧
            </kbd>
            {t.zones.editorHintPost}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {templates.map(t => (
            <Button
              key={t.label}
              onClick={() => {
                setSelection([])
                setMergeAt(null)
                setModel(t.make(templateCount))
              }}
              size="sm"
              variant="outline"
            >
              {t.label}
            </Button>
          ))}
          <Input
            className="h-7 w-14 text-center text-xs"
            max={8}
            min={1}
            onChange={e => setTemplateCount(Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
            type="number"
            value={templateCount}
          />
        </div>
      </div>

      {/* Canvas */}
      <div
        className="relative min-h-0 flex-1 cursor-crosshair overflow-hidden rounded-lg border border-(--ui-stroke-secondary)"
        onPointerDown={onCanvasPointerDown}
        onPointerLeave={() => setSplitPreview(null)}
        onPointerMove={e => updateSplitPreview(e.clientX, e.clientY)}
        ref={canvasRef}
      >
        {zones.map(zone => {
          const selected = selection.includes(zone.index)

          return (
            <div
              className="absolute flex items-center justify-center rounded-[3px] border transition-colors"
              key={zone.index}
              style={{
                left: pct(zone.left),
                top: pct(zone.top),
                width: pct(zone.right - zone.left),
                height: pct(zone.bottom - zone.top),
                // Accent over an elevated base — zones stay legible on dark
                // themes where a bare accent wash sinks into the canvas.
                background: `color-mix(in srgb, var(--ui-accent) ${selected ? 34 : 12}%, var(--ui-bg-elevated))`,
                borderColor: `color-mix(in srgb, var(--ui-accent) ${selected ? 90 : 40}%, transparent)`
              }}
            >
              {/* Quiet zone tag — the app's small-caps label voice, not a
                  billboard number. */}
              <span className="select-none text-[0.64rem] font-semibold uppercase tracking-[0.16em] text-(--ui-text-tertiary)">
                {t.zones.zoneTag(zone.index + 1)}
              </span>
            </div>
          )
        })}

        {/* Splitter preview line following the cursor. */}
        {splitPreview &&
          (() => {
            const zone = zones[splitPreview.zoneIndex]

            if (!zone) {
              return null
            }

            const horizontal = splitPreview.orientation === 'horizontal'

            return (
              <div
                className="pointer-events-none absolute"
                style={{
                  left: horizontal ? pct(zone.left) : pct(splitPreview.position),
                  top: horizontal ? pct(splitPreview.position) : pct(zone.top),
                  width: horizontal ? pct(zone.right - zone.left) : 2,
                  height: horizontal ? 2 : pct(zone.bottom - zone.top),
                  background: 'var(--ui-accent)'
                }}
              />
            )
          })()}

        {/* Rubber-band selection box. */}
        {selectBox && (
          <div
            className="pointer-events-none absolute border"
            style={{
              left: pct(selectBox.x0),
              top: pct(selectBox.y0),
              width: pct(selectBox.x1 - selectBox.x0),
              height: pct(selectBox.y1 - selectBox.y0),
              background: 'color-mix(in srgb, var(--ui-accent) 10%, transparent)',
              borderColor: 'var(--ui-accent)'
            }}
          />
        )}

        {/* Resizer thumbs on shared edges. */}
        {resizers.map((resizer, i) => {
          const geo = resizerGeometry(resizer, zones)
          const horizontal = resizer.orientation === 'horizontal'

          return (
            <div
              className={cn(
                'absolute z-10 flex items-center justify-center',
                horizontal
                  ? 'h-[10px] -translate-y-1/2 cursor-row-resize'
                  : 'w-[10px] -translate-x-1/2 cursor-col-resize'
              )}
              data-resizer={i}
              key={`r-${i}`}
              onPointerDown={e => startResizerDrag(i, e)}
              style={
                horizontal
                  ? { top: pct(geo.at), left: pct(geo.from), width: pct(geo.to - geo.from) }
                  : { left: pct(geo.at), top: pct(geo.from), height: pct(geo.to - geo.from) }
              }
            >
              <span
                className={cn('pointer-events-none rounded-full', horizontal ? 'h-[4px] w-10' : 'h-10 w-[4px]')}
                style={{ background: 'color-mix(in srgb, var(--ui-text-primary) 55%, transparent)' }}
              />
            </div>
          )
        })}

        {/* Merge affordance at drag-release point. */}
        {mergeAt && selection.length > 1 && (
          <div className="absolute z-20 flex gap-1" style={{ left: mergeAt.x, top: mergeAt.y }}>
            <Button
              className="shadow-lg"
              onClick={merge}
              onPointerDown={e => e.stopPropagation()}
              size="sm"
              variant="outline"
            >
              {t.zones.mergeZones(selection.length)}
            </Button>
          </div>
        )}
      </div>

      {/* Footer: save / cancel. */}
      <div className="flex items-center gap-1.5">
        <Input
          className="h-7 w-64 text-xs"
          onChange={e => setName(e.target.value)}
          // Escape while typing clears the field and yields — it must not bubble
          // up to close the whole editor and lose the name.
          onKeyDown={e => {
            if (e.key === 'Escape') {
              e.preventDefault()
              e.stopPropagation()
              setName('')
              e.currentTarget.blur()
            }
          }}
          placeholder={t.zones.layoutNamePlaceholder(t.zones.customZoneName(zones.length))}
          value={name}
        />
        <Button disabled={!treeExpressible} onClick={save} size="sm" variant="outline">
          {t.zones.saveApply}
        </Button>
        <Button onClick={() => $zoneEditorOpen.set(false)} size="sm" variant="ghost">
          {t.common.cancel}
        </Button>
        {!treeExpressible && <span className="text-xs text-muted-foreground/80">{t.zones.notExpressible}</span>}
        <span className="ml-auto text-xs text-muted-foreground/60">{t.zones.zoneCount(zones.length)}</span>
      </div>
    </div>
  )
}
