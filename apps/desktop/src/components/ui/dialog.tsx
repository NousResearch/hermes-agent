import { Dialog as DialogPrimitive } from 'radix-ui'
import * as React from 'react'

import { DialogPortalContainerContext } from '@/components/ui/dialog-portal-context'
import { SurfaceCloseButton, SurfaceCloseHeader } from '@/components/ui/surface-close'
import { cn } from '@/lib/utils'

function Dialog({ ...props }: React.ComponentProps<typeof DialogPrimitive.Root>) {
  return <DialogPrimitive.Root data-slot="dialog" {...props} />
}

function DialogTrigger({ ...props }: React.ComponentProps<typeof DialogPrimitive.Trigger>) {
  return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />
}

function DialogPortal({ ...props }: React.ComponentProps<typeof DialogPrimitive.Portal>) {
  return <DialogPrimitive.Portal data-slot="dialog-portal" {...props} />
}

function DialogClose({ ...props }: React.ComponentProps<typeof DialogPrimitive.Close>) {
  return <DialogPrimitive.Close data-slot="dialog-close" {...props} />
}

function DialogOverlay({
  className,
  blur = true,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Overlay> & {
  blur?: boolean
}) {
  return (
    <DialogPrimitive.Overlay
      className={cn(
        'fixed inset-0 z-(--z-modal-backdrop) pointer-events-auto bg-black/22 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0',
        blur && 'backdrop-blur-[0.125rem]',
        className
      )}
      data-slot="dialog-overlay"
      {...props}
    />
  )
}

type DialogBannerTone = 'error' | 'warn' | 'info'

// Tinted, edge-to-edge bottom banner per tone. Error/warn keep their semantic
// destructive/primary tokens; info derives from the dialog's own bubble
// background so it reads as part of the themed dialog — lifted 30% toward white
// in light mode, deepened 20% toward black in dark mode.
const DIALOG_BANNER_TONES: Record<DialogBannerTone, string> = {
  error: 'bg-destructive/12 text-destructive',
  warn: 'bg-primary/12 text-primary',
  info: 'bg-[color-mix(in_srgb,var(--ui-chat-bubble-background),white_30%)] text-[color-mix(in_srgb,var(--ui-chat-bubble-background),black_60%)] dark:bg-[color-mix(in_srgb,var(--ui-chat-bubble-background),black_20%)] dark:text-[color-mix(in_srgb,var(--ui-chat-bubble-background),white_60%)]'
}

// Opt in only for dialogs without an input; otherwise Radix's autofocus is useful.
export function preventCloseButtonAutoFocus(event: Event) {
  event.preventDefault()
}

// Radix owns dismissal; the shared header reserves the entire close row.
function DialogCloseButton() {
  return (
    <SurfaceCloseHeader
      className="order-first"
      close={
        <DialogPrimitive.Close asChild data-slot="dialog-close-button">
          <SurfaceCloseButton />
        </DialogPrimitive.Close>
      }
    />
  )
}

function DialogContent({
  className,
  bodyClassName,
  children,
  showCloseButton = true,
  fitContent = false,
  blurBackdrop = true,
  banner,
  bannerTone = 'error',
  chrome,
  overlayClassName,
  onOpenAutoFocus,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & {
  showCloseButton?: boolean
  // Backdrop skin, e.g. a heavier scrim for media viewers.
  overlayClassName?: string
  // Controls pinned to the shell rather than the scrolling body (e.g. prev/next
  // pagers). The shell doesn't clip, so these may sit past its edges.
  chrome?: React.ReactNode
  // Keep the underlying task readable for context-sensitive prompts.
  blurBackdrop?: boolean
  // Size the dialog to its content (capped at the viewport) instead of the
  // default fixed `max-w-lg`. For content that has no intrinsic width (grids,
  // full-width inputs) pair it with a `min-w-*` in `className`.
  fitContent?: boolean
  // Layout and scroll classes for the inner body box: padding, gap, display,
  // overflow. `className` styles the OUTER shell: position, size, border, and
  // background. The note on the shell below explains this split.
  bodyClassName?: string
  // A dialog-level notice rendered as a banner flush to the bottom edge (tinted,
  // inherited bottom radius) so it reads as part of the dialog, not a floating
  // alert. Falsy → no banner. Tone picks the colour.
  banner?: React.ReactNode
  bannerTone?: DialogBannerTone
}) {
  const widthClass = fitContent ? 'w-auto max-w-[92vw]' : 'w-full max-w-lg'

  // Popovers portal into the dialog so focus stays inside when they dismiss.
  const [contentNode, setContentNode] = React.useState<HTMLElement | null>(null)
  const nested = React.useContext(DialogPortalContainerContext) !== null

  const overlay = (
    <DialogOverlay blur={blurBackdrop} className={cn(nested && 'z-[calc(var(--z-modal)+1)]', overlayClassName)} />
  )

  const layerClass = nested && 'z-[calc(var(--z-modal)+2)]'
  const closeButton = showCloseButton ? <DialogCloseButton /> : null

  // With a banner the border lives on the body, not around the tinted footer.
  if (banner) {
    return (
      <DialogPortal>
        {overlay}
        <DialogPrimitive.Content
          className={cn(
            'fixed left-1/2 top-1/2 z-(--z-modal) pointer-events-auto flex max-h-[85vh] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl bg-(--ui-chat-bubble-background) text-[length:var(--conversation-text-font-size)] text-foreground shadow-nous duration-200 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
            widthClass,
            layerClass,
            className,
            'gap-0'
          )}
          data-slot="dialog-content"
          onOpenAutoFocus={onOpenAutoFocus}
          ref={setContentNode}
          {...props}
        >
          <DialogPortalContainerContext.Provider value={contentNode}>
            <div className="relative z-10 flex min-h-0 flex-col overflow-hidden rounded-xl border border-b-0 border-(--stroke-nous) bg-(--ui-chat-bubble-background)">
              <div
                className={cn(
                  'grid max-h-[calc(85vh-5rem)] min-h-0 min-w-0 grid-cols-[minmax(0,1fr)] gap-3 overflow-y-auto p-4',
                  bodyClassName
                )}
              >
                {children}
              </div>
              {closeButton}
            </div>
            <div
              className={cn(
                'relative z-0 -mt-[var(--radius-xl)] overflow-hidden rounded-b-xl px-4 pb-2.5 pt-[calc(var(--radius-xl)+0.625rem)] text-center text-[length:var(--conversation-tool-font-size)] leading-relaxed shadow-[inset_0_7px_7px_-4px_rgb(0_0_0/0.28)]',
                DIALOG_BANNER_TONES[bannerTone]
              )}
              data-slot="dialog-banner"
              role={bannerTone === 'error' ? 'alert' : 'status'}
            >
              {banner}
            </div>
            {chrome}
          </DialogPortalContainerContext.Provider>
        </DialogPrimitive.Content>
      </DialogPortal>
    )
  }

  return (
    <DialogPortal>
      {overlay}
      <DialogPrimitive.Content
        className={cn(
          // The shell cannot clip: it also hosts portaled popovers. Only the body scrolls.
          'fixed left-1/2 top-1/2 z-(--z-modal) pointer-events-auto flex max-h-[85vh] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) text-[length:var(--conversation-text-font-size)] text-foreground shadow-nous duration-200 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
          widthClass,
          layerClass,
          className
        )}
        data-slot="dialog-content"
        onOpenAutoFocus={onOpenAutoFocus}
        ref={setContentNode}
        {...props}
      >
        <DialogPortalContainerContext.Provider value={contentNode}>
          <div
            className={cn(
              'grid min-h-0 min-w-0 grid-cols-[minmax(0,1fr)] gap-3 overflow-y-auto rounded-[inherit] p-4',
              bodyClassName
            )}
          >
            {children}
          </div>
          {chrome}
          {closeButton}
        </DialogPortalContainerContext.Provider>
      </DialogPrimitive.Content>
    </DialogPortal>
  )
}

function DialogHeader({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn('flex flex-col gap-1 text-center sm:text-start', className)}
      data-slot="dialog-header"
      {...props}
    />
  )
}

function DialogFooter({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)}
      data-slot="dialog-footer"
      {...props}
    />
  )
}

function DialogTitle({
  className,
  icon: Icon,
  children,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Title> & {
  // Canonical dialog-header glyph, plain and primary-tinted.
  icon?: React.ComponentType<{ className?: string }>
}) {
  return (
    <DialogPrimitive.Title
      className={cn(
        'text-[0.9375rem] font-semibold tracking-tight text-foreground',
        Icon && 'flex items-center gap-2',
        className
      )}
      data-slot="dialog-title"
      {...props}
    >
      {Icon ? <Icon className="size-4 shrink-0 text-primary" /> : null}
      {children}
    </DialogPrimitive.Title>
  )
}

function DialogDescription({ className, ...props }: React.ComponentProps<typeof DialogPrimitive.Description>) {
  return (
    <DialogPrimitive.Description
      className={cn(
        'text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)',
        className
      )}
      data-slot="dialog-description"
      {...props}
    />
  )
}

export {
  Dialog,
  DialogClose,
  DialogCloseButton,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
  DialogTrigger
}
