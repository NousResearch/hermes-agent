import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

/**
 * Full-bleed affordance shown while files are dragged over the chat area. Always
 * `pointer-events-none` so the drop lands on the real element underneath and the
 * drop-zone handler claims it — the overlay is purely visual. Mirrors the
 * composer surface so the two read as one family.
 */
export function ChatDropOverlay({ active }: { active: boolean }) {
  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute inset-0 z-40 flex items-center justify-center p-4 transition-opacity duration-150 ease-out',
        active ? 'opacity-100' : 'opacity-0'
      )}
      data-slot="chat-drop-overlay"
    >
      <div className="absolute inset-2 rounded-2xl border-2 border-dashed border-[color-mix(in_srgb,var(--dt-composer-ring)_55%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_55%,transparent)] backdrop-blur-[2px] [-webkit-backdrop-filter:blur(2px)]" />
      <div className="relative flex items-center gap-2 rounded-full border border-[color-mix(in_srgb,var(--dt-composer-ring)_45%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-4 py-2 text-[0.8125rem] font-medium text-foreground shadow-composer">
        <Codicon className="text-(--ui-accent)" name="cloud-upload" size="1rem" />
        Drop files to attach
      </div>
    </div>
  )
}
