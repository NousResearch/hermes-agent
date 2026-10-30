'use client'

import { useStore } from '@nanostores/react'

import { isLayoutNode } from '@/components/pane-shell/tree/model'
import { layoutPresetResting, LAYOUTS_AREA } from '@/components/pane-shell/tree/presets'
import { TreeThumbnail } from '@/components/pane-shell/tree/renderer/layout-picker'
import { $activePresetId } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { layoutPickOpens } from '@/store/pane-focus'

import { CARD_ACTION_CLASS, CARD_DETAIL_CLASS, CARD_OPTION_CLASS } from '../setup-beat/card-type'

import { useLookPreview } from './look-preview'

export interface LayoutPickerCardProps {
  /** Answers with the staged preset id; true once sent. */
  onContinue: (preset: string) => boolean
}

/** Basic is the Simple shell's sidebar-left; Elite is the Advanced terminal deck. */
export const FIRST_RUN_LAYOUTS = [
  { id: 'sidebar-left', name: 'basic', detail: 'basicDetail' },
  { id: 'terminal-deck', name: 'elite', detail: 'eliteDetail' }
] as const

/** A thumbnail draws the panes a pick opens as open, so Elite's terminal and files show. */
function pickResting(id: string): ReadonlySet<string> {
  const opens = layoutPickOpens(id)

  return new Set([...layoutPresetResting(id)].filter(pane => !opens.includes(pane)))
}

/** The two first-run layouts as the layout picker's own preset cards; a click previews the layout, Continue answers with it. */
export function LayoutPickerCard({ onContinue }: LayoutPickerCardProps) {
  const { t } = useI18n()
  const copy = t.assistant.themePicker
  const look = useLookPreview('layout', false)
  const staged = look.staged?.value ?? null
  const presets = useContributions(LAYOUTS_AREA)
  const activeId = useStore($activePresetId)

  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-2 gap-3 p-1" role="group">
        {FIRST_RUN_LAYOUTS.map(choice => {
          const tree = presets.find(preset => preset.id === choice.id)?.data
          const active = choice.id === (staged ?? activeId)

          return (
            <button
              aria-pressed={active}
              className={cn(selectableCardClass({ active, prominent: true }), 'flex flex-col gap-2 p-2 text-left')}
              key={choice.id}
              onClick={() => look.stage(choice.id)}
              type="button"
            >
              <span className="flex aspect-[10/7] w-full">
                {isLayoutNode(tree) && <TreeThumbnail node={tree} resting={pickResting(choice.id)} />}
              </span>
              <span className="flex flex-col gap-0.5">
                <span
                  className={cn(
                    CARD_OPTION_CLASS,
                    'font-medium',
                    active ? 'text-foreground' : 'text-(--ui-text-secondary)'
                  )}
                >
                  {copy[choice.name]}
                </span>
                <span className={cn(CARD_DETAIL_CLASS, 'text-(--ui-text-tertiary)')}>{copy[choice.detail]}</span>
              </span>
            </button>
          )
        })}
      </div>
      <div>
        <Button
          className={CARD_ACTION_CLASS}
          disabled={!staged}
          onClick={() => look.commit(value => value !== null && onContinue(value))}
          size="sm"
        >
          {t.common.continue}
        </Button>
      </div>
    </div>
  )
}
