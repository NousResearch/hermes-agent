import { Button } from '@/components/ui/button'
import { WisdomReviewTables } from '@/components/wisdom-checks'
import type { WisdomConsentAction, WisdomConsentInteraction } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronLeft, ChevronRight } from '@/lib/icons'

interface WisdomConsentDetailsProps {
  interaction: WisdomConsentInteraction
  review: WisdomConsentInteraction['inspection']
  expanded: boolean
  busy: boolean
  active: boolean
  onToggle: (open: boolean) => void
  onAction: (action: WisdomConsentAction) => void
}

export function WisdomConsentDetails({
  interaction,
  review,
  expanded,
  busy,
  active,
  onToggle,
  onAction
}: WisdomConsentDetailsProps) {
  const { t } = useI18n()
  const copy = t.skills.collective

  return (
    <details className="mt-2" onToggle={event => onToggle(event.currentTarget.open)} open={expanded}>
      <summary className="cursor-pointer text-xs">{copy.reviewExact}</summary>
      <WisdomReviewTables
        professionalism={interaction.facts.professionalism_check}
        security={interaction.facts.security_check}
      />
      {interaction.facts.file_names?.map(name => (
        <p className="break-words font-mono text-xs" key={name}>
          {name}
        </p>
      ))}
      {review && (
        <div className="mt-3 min-w-0">
          <p className="whitespace-pre-wrap break-words text-xs">{review.description}</p>
          <p className="mt-2 break-words font-mono text-xs">{review.path}</p>
          <pre className="my-2 max-h-80 overflow-auto whitespace-pre-wrap break-words border border-(--ui-stroke-tertiary) p-2 text-xs">
            {review.content}
          </pre>
          {active && (
            <div className="flex items-center gap-2">
              <Button
                aria-label={copy.reviewPreviousPage}
                disabled={busy || review.page === 0}
                onClick={() => onAction(`inspect.${review.page - 1}`)}
                size="icon"
                variant="outline"
              >
                <ChevronLeft aria-hidden />
              </Button>
              <span className="text-xs tabular-nums">
                {review.page + 1}/{review.page_count}
              </span>
              <Button
                aria-label={copy.reviewNextPage}
                disabled={busy || review.page + 1 >= review.page_count}
                onClick={() => onAction(`inspect.${review.page + 1}`)}
                size="icon"
                variant="outline"
              >
                <ChevronRight aria-hidden />
              </Button>
            </div>
          )}
        </div>
      )}
    </details>
  )
}
