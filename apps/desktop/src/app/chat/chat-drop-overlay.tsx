import { useRef } from 'react'

import type { DragKind } from '@/app/chat/hooks/use-file-drop-zone'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

const COPY: Record<'files' | 'session', { icon: string; label: string }> = {
  files: { icon: 'cloud-upload', label: 'Drop files to attach' },
  session: { icon: 'comment-discussion', label: 'Drop to link this chat' }
}

/**
 * Full-bleed affordance shown while files or a session are dragged over the chat
 * area. Always `pointer-events-none` so the drop lands on the real element
 * underneath and the drop-zone handler claims it — the overlay is purely visual.
 * Copy adapts to whatever is being dragged; the last kind is held through the
 * fade-out so the label doesn't blank.
 */
export function ChatDropOverlay({ kind }: { kind: DragKind }) {
  const lastKind = useRef<'files' | 'session'>('files')

  if (kind) {
    lastKind.current = kind
  }

  const { icon, label } = COPY[kind ?? lastKind.current]

  return (
    <div
      aria-hidden
      className={cn(
        'pointer-events-none absolute inset-0 z-40 flex items-center justify-center p-4 transition-opacity duration-150 ease-out',
        kind ? 'opacity-100' : 'opacity-0'
      )}
      data-slot="chat-drop-overlay"
    >
      <div className="absolute inset-2 rounded-2xl border-2 border-dashed border-[color-mix(in_srgb,var(--dt-composer-ring)_55%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_55%,transparent)] backdrop-blur-[2px] [-webkit-backdrop-filter:blur(2px)]" />
      <div className="relative flex items-center gap-2 rounded-full border border-[color-mix(in_srgb,var(--dt-composer-ring)_45%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-4 py-2 text-[0.8125rem] font-medium text-foreground shadow-composer">
        <Codicon className="text-(--ui-accent)" name={icon} size="1rem" />
        {label}
      </div>
    </div>
  )
}
