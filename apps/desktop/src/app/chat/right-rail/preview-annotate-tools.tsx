/**
 * Markup tools for comment mode: freehand, arrow, and the two shapes, plus undo
 * and clear. They arm a tool in the guest page (see `in-page.ts`), so a stroke
 * is drawn ON the page and therefore appears in the crop the agent receives —
 * that is the whole point of circling something.
 *
 * Draw first, then box or pick with the pointer: the crop is taken when the
 * target is picked, so anything drawn before that point is in the picture.
 * Saving or cancelling a comment clears the strokes.
 *
 * The picker is the default: with no tool armed a click picks an element and a
 * drag boxes a comment area.
 */

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ArrowUpRight, CircleIcon, Pencil, Restore, Square, Trash2 } from '@/lib/icons'
import type { AnnotateMarkupTool } from '@/lib/preview-annotate'
import { cn } from '@/lib/utils'

interface PreviewAnnotateToolsProps {
  onClear: () => void
  onPick: (tool: AnnotateMarkupTool) => void
  onPicker: () => void
  onUndo: () => void
  tool: AnnotateMarkupTool | null
}

export function PreviewAnnotateTools({ onClear, onPick, onPicker, onUndo, tool }: PreviewAnnotateToolsProps) {
  const { t } = useI18n()
  const copy = t.preview.web

  const glyph = (active: boolean) =>
    cn(
      'grid size-7 place-items-center rounded-md transition-colors [&_svg]:size-3.5',
      active
        ? 'bg-(--ui-control-active-background) text-foreground'
        : 'text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground'
    )

  return (
    <div
      aria-label={copy.annotateTools}
      className="absolute left-1/2 top-2 z-20 flex -translate-x-1/2 items-center gap-0.5 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-1 shadow-lg"
      data-annotate-tools=""
      role="toolbar"
    >
      <Tip label={copy.annotatePicker}>
        <Button
          aria-label={copy.annotatePicker}
          aria-pressed={tool === null}
          className={glyph(tool === null)}
          onClick={onPicker}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <span aria-hidden className="text-[0.8rem] leading-none">
            ⤢
          </span>
        </Button>
      </Tip>
      <Tip label={copy.annotateDraw}>
        <Button
          aria-label={copy.annotateDraw}
          aria-pressed={tool === 'draw'}
          className={glyph(tool === 'draw')}
          onClick={() => onPick('draw')}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Pencil />
        </Button>
      </Tip>
      <Tip label={copy.annotateArrow}>
        <Button
          aria-label={copy.annotateArrow}
          aria-pressed={tool === 'arrow'}
          className={glyph(tool === 'arrow')}
          onClick={() => onPick('arrow')}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <ArrowUpRight />
        </Button>
      </Tip>
      <Tip label={copy.annotateBox}>
        <Button
          aria-label={copy.annotateBox}
          aria-pressed={tool === 'rect'}
          className={glyph(tool === 'rect')}
          onClick={() => onPick('rect')}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Square />
        </Button>
      </Tip>
      <Tip label={copy.annotateEllipse}>
        <Button
          aria-label={copy.annotateEllipse}
          aria-pressed={tool === 'ellipse'}
          className={glyph(tool === 'ellipse')}
          onClick={() => onPick('ellipse')}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <CircleIcon />
        </Button>
      </Tip>
      <span aria-hidden className="mx-0.5 h-4 w-px bg-(--ui-stroke-tertiary)" />
      <Tip label={copy.annotateUndo}>
        <Button
          aria-label={copy.annotateUndo}
          className={glyph(false)}
          onClick={onUndo}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Restore />
        </Button>
      </Tip>
      <Tip label={copy.annotateClear}>
        <Button
          aria-label={copy.annotateClear}
          className={glyph(false)}
          onClick={onClear}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Trash2 />
        </Button>
      </Tip>
    </div>
  )
}
