import { useStore } from '@nanostores/react'
import type { ReactElement } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { $modelPriceUnit, $showModelPricing, setModelPriceUnit, setShowModelPricing } from '@/store/model-pricing'

import { ListRow, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

const ids = SETTING_IDS.appearance

/** Model Pricing: whether the pickers show prices, then — only while they do — the unit they show
 *  them in. A unit for prices the picker does not show would be noise. */
export function ModelPricingRows(): ReactElement {
  const { t } = useI18n()
  const a = t.settings.appearance
  const copy = t.shell.modelMenu
  const showModelPricing = useStore($showModelPricing)
  const priceUnit = useStore($modelPriceUnit)

  return (
    <>
      <ToggleRow
        checked={showModelPricing}
        description={a.modelPricingDesc}
        id={settingElementId(ids.modelPricing)}
        label={a.modelPricingTitle}
        onChange={setShowModelPricing}
      />
      {showModelPricing ? (
        <ListRow
          action={
            <SegmentedControl
              onChange={setModelPriceUnit}
              options={[
                { id: 'mtok', label: copy.priceUnitPerMillion },
                { id: '1k', label: copy.priceUnitPerThousand }
              ]}
              value={priceUnit}
            />
          }
          description={copy.priceUnitDesc}
          id={settingElementId(ids.modelPriceUnit)}
          title={copy.priceUnitTitle}
        />
      ) : null}
    </>
  )
}
