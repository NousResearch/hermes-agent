import type { ReactNode } from 'react'

import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

/** The redesigned page's titlebar band (Figma 856:2713): a 30px Back chrome button 20px from the
 *  rail, the page's own controls centered, its actions at the end. Only redesigned pages mount it. */

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
    <header className="flex h-full min-h-0 shrink-0 items-center gap-3 px-5">
      <Tip label={t.common.back} placement="toolbar">
        <Button
          aria-label={t.common.back}
          className="portal-chrome-button [-webkit-app-region:no-drag]"
          onClick={onBack}
          size="icon"
          variant="ghost"
        >
          <TitlebarIcon name="arrow-left" />
        </Button>
      </Tip>
      <div className="flex min-w-0 flex-1 items-center justify-center gap-2 [-webkit-app-region:no-drag]">
        {children}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1.5 [-webkit-app-region:no-drag]">{actions}</div>}
    </header>
  )
}
