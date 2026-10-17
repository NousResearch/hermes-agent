import { Badge } from '@/components/ui/badge'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { ModelPricing } from '@/types/hermes'

/** The gateway owns prices, units and sale calculations; this leaf only lays them out. */
export function NousModelPrice({
  price,
  showUnit = false,
  selected = false
}: {
  price?: ModelPricing
  showUnit?: boolean
  selected?: boolean
}) {
  const { t } = useI18n()
  const copy = t.shell.modelMenu

  if (!price || (!price.input && !price.output)) {
    return null
  }

  const displayPrice = (value: string) => (value === 'free' ? t.modelPicker.free : value)

  return (
    <span
      className={cn(
        'flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.625rem] tabular-nums text-(--ui-text-tertiary)',
        selected && 'text-primary-foreground/80'
      )}
    >
      {price.free ? (
        <Badge size="xs" variant="success">
          {t.modelPicker.free}
        </Badge>
      ) : (
        <>
          {price.input && (
            <span>
              {copy.inputPrice} {displayPrice(price.input)}
            </span>
          )}
          {price.output && (
            <span>
              {copy.outputPrice} {displayPrice(price.output)}
            </span>
          )}
          {price.cache && (
            <span>
              {copy.cachePrice} {displayPrice(price.cache)}
            </span>
          )}
        </>
      )}
      {typeof price.discount_percent === 'number' && (
        <Badge size="xs" variant="warn">
          -{price.discount_percent}%
        </Badge>
      )}
      {(price.was_input || price.was_output) && (
        <span className="line-through decoration-from-font">
          {t.modelPicker.wasPrice} {price.was_input || '—'} / {price.was_output || '—'}
        </span>
      )}
      {showUnit && <span>{copy.priceUnit}</span>}
    </span>
  )
}
