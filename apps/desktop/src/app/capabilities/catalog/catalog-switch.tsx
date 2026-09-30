import { type ComponentProps, useId } from 'react'

import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'

/** Visible action copy stays attached to the actual controlled switch. Both
 *  labels share one grid cell so the control never shifts when it flips. */
export function CatalogSwitch(props: ComponentProps<typeof Switch>) {
  const id = useId()
  const { t } = useI18n()

  return <label className="catalog-switch relative inline-flex shrink-0 items-center gap-2 text-xs text-(--ui-text-secondary)" htmlFor={id}>
    <span className="grid justify-items-end [&>*]:col-start-1 [&>*]:row-start-1">
      <span className={props.checked ? undefined : 'invisible'}>{t.settings.plugins.disable}</span>
      <span className={props.checked ? 'invisible' : undefined}>{t.settings.plugins.enable}</span>
    </span>
    <Switch size="xs" {...props} id={id} />
  </label>
}
