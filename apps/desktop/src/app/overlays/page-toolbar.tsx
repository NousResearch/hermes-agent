import type { ReactNode } from 'react'

import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

export function PageToolbar({
  actions,
  children,
  onBack
}: {
  actions?: ReactNode
  children?: ReactNode
  onBack: () => void
}) {
  const { t } = useI18n()

  return (
    <header className="flex h-full min-h-0 shrink-0 items-center gap-3 px-3 py-1">
      <Button className="[-webkit-app-region:no-drag]" onClick={onBack} size="sm" variant="ghost">
        <TitlebarIcon name="arrow-left" />
        {t.common.back}
      </Button>
      <div className="flex min-w-0 flex-1 items-center justify-center gap-2 [-webkit-app-region:no-drag]">{children}</div>
      {actions && <div className="flex shrink-0 items-center gap-1.5 [-webkit-app-region:no-drag]">{actions}</div>}
    </header>
  )
}
