import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'

interface StatusDismissButtonProps {
  label: string
  onDismiss: () => void
}

/** The leading close control shared by artifacts, background work and queued prompts. */
export function StatusDismissButton({ label, onDismiss }: StatusDismissButtonProps) {
  return (
    <Button
      aria-label={label}
      className="status-dismiss-button text-muted-foreground/60 hover:text-foreground/90"
      data-slot="status-dismiss"
      onClick={event => {
        event.stopPropagation()
        onDismiss()
      }}
      onKeyDown={event => event.stopPropagation()}
      size="icon-xs"
      type="button"
      variant="ghost"
    >
      <Codicon name="close" size="0.8rem" />
    </Button>
  )
}
