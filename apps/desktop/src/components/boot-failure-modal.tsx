import { Dialog as DialogPrimitive } from 'radix-ui'
import { type ReactNode, useState } from 'react'

import { DialogPortalContainerContext } from '@/components/ui/dialog-portal-context'

// Recovery surface for a hard boot failure (gateway never came up, backend
// exited during startup, bootstrap latched, …). Without this the app shell
// renders dead — "gateway offline", no composer, only a toast — with no way
// to retry, repair the install, switch the gateway, or find the logs.
export function BootFailureModal({
  children,
  onDismiss,
  title
}: {
  children: ReactNode
  onDismiss: () => void
  title?: string
}) {
  const [contentNode, setContentNode] = useState<HTMLDivElement | null>(null)

  return (
    <DialogPrimitive.Root
      onOpenChange={open => {
        if (!open) {
          onDismiss()
        }
      }}
      open
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Content aria-describedby={undefined} aria-modal="true" asChild ref={setContentNode}>
          <div
            // contain: layout makes this overlay the containing block of the poppers
            // portaled into it, so Floating UI scales them by the Webapp's root zoom.
            className="fixed inset-0 z-(--z-setup) flex items-center justify-center bg-(--ui-chat-surface-background) p-6 [contain:layout]"
            // Masks the whole app on boot failure — must stay filled under window
            // glass. Contract: `[data-glass-opaque]` in styles.css.
            data-glass-opaque=""
          >
            <DialogPortalContainerContext.Provider value={contentNode}>
              {title ? <DialogPrimitive.Title className="sr-only">{title}</DialogPrimitive.Title> : null}
              {children}
            </DialogPortalContainerContext.Provider>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
