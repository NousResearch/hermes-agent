import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import type { ComposerPredictions } from './predictions'

interface ComposerPredictionPreviewProps {
  prediction: Pick<ComposerPredictions, 'suggestions' | 'index'>
  stacked: boolean
}

/** A grid overlay sibling, NEVER a descendant of the contentEditable. The
 * preview sizes the shared cell so long messages wrap fully instead of clipping
 * or overflowing the controls. It cannot receive focus, selection or clicks. */
export function ComposerPredictionPreview({ prediction, stacked }: ComposerPredictionPreviewProps) {
  const { t } = useI18n()

  return (
    <>
      <div
        aria-hidden
        className={cn(
          'pointer-events-none min-w-0 select-none whitespace-pre-wrap [overflow-wrap:anywhere] pb-1 pr-1 pt-1 leading-normal text-muted-foreground/60 text-[length:var(--conversation-text-font-size)] [grid-area:1/1]',
          stacked && 'pl-3'
        )}
        data-slot="composer-prediction-text"
      >
        {prediction.suggestions[prediction.index]}
      </div>
      <div
        aria-hidden
        className={cn(
          'pointer-events-none select-none text-xs text-muted-foreground [grid-area:2/1]',
          stacked && 'pl-3'
        )}
        data-slot="composer-prediction-hint"
      >
        <span>{t.composer.predictionHint}</span>{' '}
        <span>
          {prediction.index + 1}/{prediction.suggestions.length}
        </span>
      </div>
      <span aria-atomic="true" className="sr-only" role="status">
        {t.composer.predictionAnnouncement(
          prediction.index + 1,
          prediction.suggestions.length,
          prediction.suggestions[prediction.index]
        )}{' '}
        {t.composer.predictionHint}
      </span>
    </>
  )
}
