/**
 * Layout picker — the preset card grid inside the edit palette. Thumbnails
 * are a miniature render of each preset's layout tree; clicking a card
 * applies it, and "Save current arrangement" captures the live tree as a
 * user preset. The "New grid layout" button opens the zone editor.
 */

import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import type { LayoutNode } from '../model'
import { isLayoutNode } from '../model'
import { applyLayoutPreset, deleteUserPreset, isUserPreset, LAYOUTS_AREA, saveCurrentLayoutAs } from '../presets'
import { $activePresetId } from '../store'
import { $zoneEditorOpen } from '../zone-editor'

/** Miniature render of a layout tree — the preset card thumbnail. */
function TreeThumbnail({ node }: { node: LayoutNode }) {
  if (node.type === 'group') {
    return (
      // currentColor-derived fill: light zones on dark themes, dark zones on
      // light — legible everywhere without leaning on the accent.
      <div
        className="min-h-0 min-w-0 flex-1 rounded-[2px]"
        style={{ background: 'color-mix(in srgb, currentColor 16%, transparent)' }}
      />
    )
  }

  return (
    <div className={cn('flex min-h-0 min-w-0 flex-1 gap-px', node.orientation === 'row' ? 'flex-row' : 'flex-col')}>
      {node.children.map((child, i) => (
        <div
          className="flex min-h-0 min-w-0"
          key={child.id}
          style={{ flex: `${node.weights[i]} ${node.weights[i]} 0px` }}
        >
          <TreeThumbnail node={child} />
        </div>
      ))}
    </div>
  )
}

/** Small-caps section heading — the app's SidebarPanelLabel voice. */
function PickerSectionLabel({ children }: { children: ReactNode }) {
  return (
    <span className="text-[0.6rem] font-semibold uppercase tracking-[0.16em] text-(--ui-text-quaternary)">
      {children}
    </span>
  )
}

function PresetCard({ preset }: { preset: Contribution }) {
  const { t } = useI18n()
  const activeId = useStore($activePresetId)
  const tree = isLayoutNode(preset.data) ? preset.data : null

  if (!tree) {
    return null
  }

  const active = preset.id === activeId

  return (
    <div className="group/preset relative">
      <button
        className={cn(
          'flex w-full flex-col gap-1.5 rounded-lg border p-1.5 text-left transition-colors',
          active
            ? 'border-(--ui-accent) bg-(--ui-row-active-background)'
            : 'border-(--ui-stroke-secondary) hover:border-(--ui-stroke-primary) hover:bg-(--ui-row-hover-background)'
        )}
        onClick={() => applyLayoutPreset(preset.id, tree)}
        type="button"
      >
        <div className="flex h-12 w-full">
          <TreeThumbnail node={tree} />
        </div>
        <span
          className={cn('truncate text-[0.68rem] font-medium', active ? 'text-foreground' : 'text-muted-foreground/80')}
        >
          {preset.title ?? preset.id}
        </span>
      </button>
      {isUserPreset(preset.id) && (
        <button
          aria-label={t.zones.deletePreset(preset.title ?? preset.id)}
          // Hover-reveal (opacity, not display) — stays laid out + clickable,
          // appears on card hover or keyboard focus.
          className="absolute right-1 top-1 z-10 grid size-5 place-items-center rounded-md bg-(--ui-bg-elevated) text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 group-hover/preset:opacity-100"
          onClick={() => deleteUserPreset(preset.id)}
          onPointerDown={e => e.stopPropagation()}
          type="button"
        >
          <Codicon name="close" size="0.7rem" />
        </button>
      )}
    </div>
  )
}

export function LayoutPicker() {
  const { t } = useI18n()
  const presets = useContributions(LAYOUTS_AREA)
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)

  const templates = presets.filter(p => !isUserPreset(p.id) && isLayoutNode(p.data))
  const custom = presets.filter(p => isUserPreset(p.id) && isLayoutNode(p.data))

  const commitSave = () => {
    if (!name.trim()) {
      return
    }

    saveCurrentLayoutAs(name)
    setName('')
    setSaving(false)
  }

  return (
    <div className="flex flex-col gap-4">
      <section className="flex flex-col gap-2">
        <PickerSectionLabel>{t.zones.templates}</PickerSectionLabel>
        <div className="grid grid-cols-4 gap-2">
          {templates.map(preset => (
            <PresetCard key={`${preset.source ?? 'core'}:${preset.id}`} preset={preset} />
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <PickerSectionLabel>{t.zones.custom}</PickerSectionLabel>
        {custom.length > 0 && (
          <div className="grid grid-cols-4 gap-2">
            {custom.map(preset => (
              <PresetCard key={`${preset.source ?? 'core'}:${preset.id}`} preset={preset} />
            ))}
          </div>
        )}
        <Button
          className="h-8 w-full justify-center gap-1.5 border border-dashed border-(--ui-stroke-secondary) text-muted-foreground hover:border-(--ui-stroke-primary) hover:text-foreground"
          onClick={() => $zoneEditorOpen.set(true)}
          size="sm"
          variant="ghost"
        >
          <Codicon name="add" size="0.875rem" />
          {t.zones.newGridLayout}
        </Button>
      </section>

      {/* Save-current lives behind a reveal so the raw input doesn't clash
          with the card grid until it's actually needed. */}
      {saving ? (
        <form
          className="flex items-center gap-1.5"
          onSubmit={e => {
            e.preventDefault()
            commitSave()
          }}
        >
          <Input
            autoFocus
            className="h-7 flex-1 text-xs"
            onChange={e => setName(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Escape') {
                setSaving(false)
                setName('')
              }
            }}
            placeholder={t.zones.nameLayoutPlaceholder}
            value={name}
          />
          <Button disabled={!name.trim()} size="sm" type="submit" variant="outline">
            {t.common.save}
          </Button>
          <Button onClick={() => setSaving(false)} size="sm" variant="ghost">
            {t.common.cancel}
          </Button>
        </form>
      ) : (
        <button
          className="flex items-center gap-1.5 self-start text-xs text-muted-foreground/80 transition-colors hover:text-foreground"
          onClick={() => setSaving(true)}
          type="button"
        >
          <Codicon name="save" size="0.8125rem" />
          {t.zones.saveCurrentAs}
        </button>
      )}
    </div>
  )
}
