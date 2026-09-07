import {
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSubContent
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import type { ModelPickerMetadata, ModelPricing } from '@/types/hermes'

export function compactTokens(value?: number): string | null {
  if (!value || value <= 0) {
    return null
  }

  const unit = value >= 1_000_000 ? 1_000_000 : value >= 1_000 ? 1_000 : 1
  const suffix = unit === 1_000_000 ? 'M' : unit === 1_000 ? 'K' : ''
  const amount = value / unit

  return `${Number.isInteger(amount) ? amount : amount.toFixed(1)}${suffix}`
}

/** Models whose id ends in the `-free` suffix are free-tier entries in the
 *  catalogs that name them that way (muse-spark-1.2-contributor-free,
 *  hy3-free, …). Pure display signal — mirrors what OpenCode derives. */
export function isFreeTierModelId(modelId: string): boolean {
  return /-free$/i.test(modelId.trim())
}

export function metadataSummary(metadata?: ModelPickerMetadata, freeBadge?: string): string {
  if (!metadata) {
    return ''
  }

  return [
    compactTokens(metadata.context_window),
    metadata.supports_vision ? 'Vision' : null,
    metadata.supports_tools ? 'Tools' : null,
    freeBadge
  ]
    .filter(Boolean)
    .join(' · ')
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 px-2 py-0.5 text-xs">
      <span className="text-(--ui-text-tertiary)">{label}</span>
      <span className="text-right font-medium text-foreground">{value}</span>
    </div>
  )
}

/** Model facts popover, modeled on OpenCode's per-model tooltip: label/value
 *  rows for model, provider, accepted inputs (modalities), reasoning, and
 *  context — plus output limits and pricing when the registry knows them. */
export function ModelDetailsSubmenu({
  metadata,
  model,
  modelName,
  pricing,
  providerName
}: {
  metadata?: ModelPickerMetadata
  model: string
  modelName: string
  pricing?: ModelPricing
  providerName: string
}) {
  const copy = useI18n().t.shell.modelOptions
  const context = compactTokens(metadata?.context_window)
  const maxOutput = compactTokens(metadata?.max_output_tokens)

  // Modalities in registry order, each rendered with its translated label.
  const modalityLabels: Record<string, string> = {
    audio: copy.modalityAudio,
    image: copy.modalityImage,
    pdf: copy.modalityPdf,
    text: copy.modalityText,
    video: copy.modalityVideo
  }

  const inputs = (metadata?.input_modalities ?? [])
    .map(modality => modalityLabels[modality] ?? modality)
    .join(', ')

  // Registry order can be noisy; keep the summary chips as the fallback so an
  // unknown-modalities model still shows what it accepts.
  const fallbackInputs = [
    copy.modalityText,
    metadata?.supports_vision ? copy.modalityImage : null,
    metadata?.supports_audio_input ? copy.modalityAudio : null,
    metadata?.supports_pdf ? copy.modalityPdf : null
  ]
    .filter(Boolean)
    .join(', ')

  return (
    <DropdownMenuSubContent className="w-64 p-1.5">
      <DropdownMenuLabel className="px-2 py-1 text-xs font-semibold">{copy.detailsTitle}</DropdownMenuLabel>
      <Row label={copy.model} value={modelName} />
      <Row label={copy.provider} value={providerName} />
      <Row label={copy.inputs} value={inputs || fallbackInputs || '—'} />
      {context ? <Row label={copy.contextWindow} value={copy.tokens(context)} /> : null}
      {maxOutput ? <Row label={copy.maxOutput} value={copy.tokens(maxOutput)} /> : null}
      {pricing ? (
        <>
          <DropdownMenuSeparator className="mx-1" />
          {pricing.free ? (
            <Row label={copy.pricing} value={copy.pricingFree} />
          ) : (
            <>
              {pricing.input ? <Row label={copy.pricingInput} value={copy.pricingPerMtok(pricing.input)} /> : null}
              {pricing.output ? <Row label={copy.pricingOutput} value={copy.pricingPerMtok(pricing.output)} /> : null}
              {pricing.cache ? <Row label={copy.cachedInput} value={copy.pricingPerMtok(pricing.cache)} /> : null}
            </>
          )}
        </>
      ) : isFreeTierModelId(model) ? (
        <>
          <DropdownMenuSeparator className="mx-1" />
          <Row label={copy.pricing} value={copy.pricingFree} />
        </>
      ) : null}
      {!metadata && !pricing ? (
        <div className="px-2 py-1 text-xs text-(--ui-text-tertiary)">{copy.detailsUnavailable}</div>
      ) : null}
    </DropdownMenuSubContent>
  )
}
