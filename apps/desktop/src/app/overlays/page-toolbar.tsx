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
    <header className="flex min-h-10 shrink-0 items-center gap-3 px-3 py-1">
      <Button onClick={onBack} size="sm" variant="ghost">
        <TitlebarIcon name="arrow-left" />
        {t.common.back}
      </Button>
      <div className="flex min-w-0 flex-1 items-center justify-center gap-2">{children}</div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </header>
  )
}
