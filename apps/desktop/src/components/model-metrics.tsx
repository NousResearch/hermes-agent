import type { ModelCapabilities, ModelPricing } from '@hermes/shared'
import type { ReactElement } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { ModelMenuTranslations } from '@/i18n/types_model_menu'
import { cn } from '@/lib/utils'
import type { ModelPriceUnit } from '@/store/model-pricing'

const compactTokens = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })

/** A backend `$/Mtok` string (`"$0.15"`) as the same price per 1K tokens (`"$0.00015"`); null
 *  for anything that is not a plain dollar figure ("free", "?", ""). */
export function perThousand(perMillion?: null | string): null | string {
  const match = perMillion?.match(/^\$(\d+(?:\.\d+)?)$/)

  if (!match) {
    return null
  }

  return `$${(Number(match[1]) / 1000).toFixed(8).replace(/\.?0+$/, '')}`
}

/** A backend `$/Mtok` figure in the user's chosen unit; non-dollar values ("free", "?") pass through. */
export function displayPrice(perMillion: null | string | undefined, unit: ModelPriceUnit): null | string {
  if (!perMillion) {
    return null
  }

  return unit === '1k' ? (perThousand(perMillion) ?? perMillion) : perMillion
}

/** The price chip's tooltip: its own line, the same price per 1K tokens (derived from the figure
 *  the chip shows, so the two units cannot disagree), and a line naming a models.dev list price —
 *  the provider did not report one, and a reseller may charge differently. */
export function modelPriceTitle(
  title: string,
  price: Pick<ModelPricing, 'input' | 'output' | 'source'>,
  copy: Pick<ModelMenuTranslations, 'catalogPrice' | 'perThousandTitle'>
): string {
  const input = perThousand(price.input)
  const output = perThousand(price.output)

  return [
    title,
    input || output ? copy.perThousandTitle(input ?? '—', output ?? '—') : null,
    price.source === 'catalog' ? copy.catalogPrice : null
  ]
    .filter(Boolean)
    .join('\n')
}

function CapabilityMark({ icon, label }: { icon: string; label: string }): ReactElement {
  return (
    <span aria-label={label} className="inline-flex" role="img" title={label}>
      <Codicon name={icon} size="0.66rem" />
    </span>
  )
}

/** What a model can take and give, beside its name: context window, longest reply, image input and
 *  tool calling. Both model pickers render this one component so they can never disagree. It shows
 *  only what the models.dev catalog answered: a miss reads as unknown, never as "cannot" (#112649),
 *  so a model the catalog does not know gets no marks rather than a text-only one. */
export function ModelMetrics({
  caps,
  className
}: {
  caps?: ModelCapabilities | null
  className?: string
}): null | ReactElement {
  const { t } = useI18n()
  const copy = t.shell.modelMenu
  const context = caps?.context_window ? compactTokens.format(caps.context_window) : null
  const maxOutput = caps?.max_output ? compactTokens.format(caps.max_output) : null
  const vision = caps?.supports_vision === true
  const tools = caps?.supports_tools === true

  if (!context && !maxOutput && !vision && !tools) {
    return null
  }

  return (
    <span
      className={cn(
        'flex shrink-0 items-center gap-1 text-[0.625rem] tabular-nums',
        className ?? 'text-(--ui-text-tertiary)'
      )}
      data-model-metrics=""
    >
      {context ? <span title={copy.contextTitle(context)}>{context}</span> : null}
      {maxOutput ? <span title={copy.maxOutputTitle(maxOutput)}>{copy.maxOutputLabel(maxOutput)}</span> : null}
      {vision ? <CapabilityMark icon="eye" label={copy.vision} /> : null}
      {tools ? <CapabilityMark icon="tools" label={copy.tools} /> : null}
    </span>
  )
}
