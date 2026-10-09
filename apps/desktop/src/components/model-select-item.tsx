import type { ComponentProps } from 'react'

import { NousModelPrice } from '@/components/nous-model-price'
import { SelectItem } from '@/components/ui/select'
import type { ModelOptionProvider } from '@/types/hermes'

/** Keep the selected trigger and typeahead label as the model name; pricing
 * belongs to the option's description and only to the signed-in Nous row. */
export function ModelSelectItem({
  model,
  provider,
  value = model,
  ...props
}: Omit<ComponentProps<typeof SelectItem>, 'children' | 'value'> & {
  model: string
  provider?: ModelOptionProvider
  value?: string
}) {
  const price = provider?.slug === 'nous' && !provider.free_tier_row ? provider.pricing?.[model] : undefined

  return (
    <SelectItem
      {...props}
      description={price ? <NousModelPrice price={price} showUnit /> : undefined}
      textValue={model}
      title={model}
      value={value}
    >
      {model}
    </SelectItem>
  )
}
