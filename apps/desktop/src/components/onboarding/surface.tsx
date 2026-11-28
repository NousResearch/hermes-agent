import './surface.css'

import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { closeManualOnboarding } from '@/store/onboarding'
import { usePortalShell } from '@/store/shell-variant'

interface OnboardingSurfaceProps {
  bare: boolean
  children: ReactNode
  leaving: boolean
  manual: boolean
}

export function OnboardingSurface({ bare, children, leaving, manual }: OnboardingSurfaceProps) {
  const { t } = useI18n()
  const simplified = usePortalShell()
  // Capture before a key form's native autofocus runs in the portal.
  const [opener] = useState(() => document.activeElement)

  if (simplified && manual) {
    return (
      <Dialog onOpenChange={open => !open && closeManualOnboarding()} open>
        <DialogContent
          aria-describedby={undefined}
          bodyClassName="gap-0 p-0"
          className={cn('onboarding-v2 z-(--z-onboarding) w-[calc(100vw_-_2rem)] max-w-[40rem]', leaving && 'pointer-events-none opacity-0')}
          onCloseAutoFocus={event => {
            if (opener instanceof HTMLElement && opener.isConnected) {
              event.preventDefault()
              opener.focus({ preventScroll: true })
            }
          }}
          onPointerDownOutside={event => event.preventDefault()}
          overlayClassName="z-[calc(var(--z-onboarding)-1)]"
        >
          <DialogTitle className="sr-only">{t.onboarding.headerTitle}</DialogTitle>
          {children}
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <div
      className={cn(
        'fixed inset-0 z-(--z-onboarding) flex items-center justify-center bg-(--ui-chat-surface-background) p-6 transition-opacity duration-[520ms] ease-out',
        simplified && 'onboarding-v2',
        bare && leaving ? '[transition-delay:660ms]' : '',
        leaving ? 'pointer-events-none opacity-0' : 'opacity-100'
      )}
      data-glass-opaque=""
    >
      <div
        className={cn(
          'onboarding-card relative w-full max-w-[45rem] transition-all duration-500 ease-out',
          bare ? '' : 'overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) shadow-nous',
          leaving && !bare ? '-translate-y-1 scale-[0.985] opacity-0 blur-[2px]' : 'translate-y-0 scale-100 opacity-100 blur-0'
        )}
      >
        {manual && (
          <Button
            aria-label={t.common.close}
            className="absolute right-3 top-3 z-10 text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
            onClick={closeManualOnboarding}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="close" size="1rem" />
          </Button>
        )}
        {children}
      </div>
    </div>
  )
}
