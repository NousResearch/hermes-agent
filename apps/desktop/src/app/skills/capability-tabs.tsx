import type { ReactNode } from 'react'

import { TextTab } from '@/components/ui/text-tab'
import { useI18n } from '@/i18n'

export type CapabilityView = 'installed' | 'browse'

interface CapabilityTabsProps {
  value: CapabilityView
  onChange: (value: CapabilityView) => void
  actions?: ReactNode
}

/** Shared Installed / Browse navigation for Skills and Plugins. */
export function CapabilityTabs({ value, onChange, actions }: CapabilityTabsProps) {
  const { t } = useI18n()

  return (
    <div className="flex shrink-0 items-center gap-4 px-3 py-1" data-capability-tabs>
      <TextTab active={value === 'installed'} aria-pressed={value === 'installed'} onClick={() => onChange('installed')}>
        {t.catalog.installed}
      </TextTab>
      <TextTab active={value === 'browse'} aria-pressed={value === 'browse'} onClick={() => onChange('browse')}>
        {t.catalog.browse}
      </TextTab>
      {actions && <div className="-my-0.5 -mr-2 ml-auto flex h-6 items-center gap-2">{actions}</div>}
    </div>
  )
}
