import { SurfaceCloseButton } from '@/components/ui/surface-close'

interface StatusDismissButtonProps {
  label: string
  onDismiss: () => void
}

/** The leading close control shared by artifacts, background work and queued prompts. */
export function StatusDismissButton({ label, onDismiss }: StatusDismissButtonProps) {
  return (
    <SurfaceCloseButton
      aria-label={label}
      className="status-dismiss-button text-muted-foreground/60 hover:text-foreground/90"
      data-slot="status-dismiss"
      onClick={event => {
        event.stopPropagation()
        onDismiss()
      }}
      onKeyDown={event => event.stopPropagation()}
    />
  )
}
