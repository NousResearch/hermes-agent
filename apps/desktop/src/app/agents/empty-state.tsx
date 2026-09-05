import { Codicon } from '@/components/ui/codicon'

export function EmptyState({ icon, text }: { icon: string; text: string }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
      <Codicon className="text-muted-foreground/50" name={icon} size="1.25rem" />
      <p className="text-xs text-muted-foreground/70">{text}</p>
    </div>
  )
}
