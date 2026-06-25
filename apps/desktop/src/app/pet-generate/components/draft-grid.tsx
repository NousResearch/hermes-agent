import { PixelEggSprite } from '@/components/pet/pixel-egg-sprite'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { PawPrint } from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'

const VARIANT_COUNT = 4

interface DraftGridProps {
  drafts: { index: number; dataUri: string }[]
  generating: boolean
  hasDrafts: boolean
  onCancel: () => void
  onHatch: () => void
  onRemix: (draft: { index: number; dataUri: string }) => void
  onSelect: (index: number) => void
  selected: number | null
}

export function DraftGrid({
  drafts,
  generating,
  hasDrafts,
  onCancel,
  onHatch,
  onRemix,
  onSelect,
  selected
}: DraftGridProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet

  const slots = generating
    ? Array.from({ length: VARIANT_COUNT }, (_, i) => drafts.find(draft => draft.index === i) ?? null)
    : drafts

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        <span className={cn(generating && 'shimmer shimmer-color-primary opacity-40', !generating && 'invisible')}>
          {copy.generating}
        </span>
        <span className="tabular-nums">
          {Math.min(drafts.length, VARIANT_COUNT)}/{VARIANT_COUNT}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {slots.map((draft, i) => {
          // A streamed draft is selectable immediately — even mid-generation —
          // so the user can commit to one without waiting for the rest.
          const isSelected = draft != null && selected === draft.index

          return (
            <div className="group relative aspect-[192/208]" key={draft ? `draft-${draft.index}` : `slot-${i}`}>
              <button
                className={cn(
                  'absolute inset-0 flex items-center justify-center overflow-hidden',
                  selectableCardClass({ active: isSelected, prominent: true })
                )}
                disabled={draft == null}
                onClick={() => draft != null && onSelect(draft.index)}
                type="button"
              >
                {draft != null ? (
                  // Hatches into place as each draft streams back.
                  <img
                    alt=""
                    className="pet-reveal size-full object-contain p-1.5"
                    draggable={false}
                    src={draft.dataUri}
                  />
                ) : (
                  // Incubating: a creme egg bouncing on its contact shadow.
                  <div className="relative z-10 flex flex-col items-center">
                    <PixelEggSprite index={i} mode="bounce" size={48} />
                    <span className="pet-egg-shadow pet-egg-shadow--sm" style={{ marginTop: '-0.3rem' }} />
                  </div>
                )}
              </button>

              {/* Branch off this look — reuses the draft as the next reference so
                  the user can explore variations without leaving the grid. Hidden
                  until hover/focus to keep the grid clean. */}
              {draft != null && !generating && (
                <Tip label={copy.remix}>
                  <Button
                    aria-label={copy.remix}
                    className={cn(
                      'absolute right-1 top-1 z-20',
                      'text-(--ui-text-tertiary) opacity-10 transition',
                      'hover:bg-transparent hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100'
                    )}
                    onClick={event => {
                      event.stopPropagation()
                      onRemix(draft)
                    }}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Codicon name="git-branch" size={12} />
                  </Button>
                </Tip>
              )}
            </div>
          )
        })}
      </div>

      {/* Same abort/go-back text link in both states (sits right under the grid);
          once drafts land, the full-width Hatch drops in below it. */}
      <Button className="self-center" onClick={onCancel} size="xs" variant="text">
        {t.common.cancel}
      </Button>
      {hasDrafts && (
        <Button className="w-full" disabled={selected === null} onClick={onHatch}>
          <PawPrint />
          {copy.hatch}
        </Button>
      )}
    </div>
  )
}
