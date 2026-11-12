import { type ComponentProps, useId } from 'react'

import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'

/** The caption names the switch's current STATE (On/Off), never the action a
 *  click would take: a verb ("Disable") beside a lit switch reads as "this is
 *  disabled". Both captions share one grid cell so the control never shifts. */
export function CatalogSwitch(props: ComponentProps<typeof Switch>) {
  const id = useId()
  const { t } = useI18n()

  return <label className="catalog-switch relative inline-flex shrink-0 items-center gap-2 text-xs text-(--ui-text-secondary)" htmlFor={id}>
    <span className="grid justify-items-end [&>*]:col-start-1 [&>*]:row-start-1">
      <span className={props.checked ? undefined : 'invisible'}>{t.common.on}</span>
      <span className={props.checked ? 'invisible' : undefined}>{t.common.off}</span>
    </span>
    <Switch size="xs" {...props} id={id} />
  </label>
}
