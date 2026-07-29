import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { DesktopManagedUpdateSnapshot } from '@/global'
import { cn } from '@/lib/utils'
import { $managedUpdate } from '@/store/updates'

interface ManagedUpdatePresentation {
  detail: string
  icon: string
  label: string
  spinning: boolean
  tone: 'error' | 'muted' | 'ready'
}

export function managedUpdatePresentation(
  snapshot: DesktopManagedUpdateSnapshot | null
): ManagedUpdatePresentation | null {
  if (!snapshot || snapshot.stage === 'disabled' || snapshot.stage === 'idle') {
    return null
  }

  if (snapshot.stage === 'checking') {
    return {
      detail: 'Hermes is checking for a verified update.',
      icon: 'sync',
      label: 'Checking for update…',
      spinning: true,
      tone: 'muted'
    }
  }

  if (snapshot.stage === 'available') {
    return {
      detail: 'A verified update was found. Download begins automatically.',
      icon: 'cloud-download',
      label: 'Update found…',
      spinning: true,
      tone: 'muted'
    }
  }

  if (snapshot.stage === 'downloading') {
    const percent = snapshot.percent === null ? null : Math.round(snapshot.percent)

    return {
      detail: 'Hermes is downloading and verifying the update in the background.',
      icon: 'cloud-download',
      label: percent === null ? 'Downloading…' : `Downloading ${percent}%`,
      spinning: false,
      tone: 'muted'
    }
  }

  if (snapshot.stage === 'downloaded') {
    return {
      detail: 'Verified and ready. The update installs automatically when Hermes closes normally.',
      icon: 'check',
      label: 'Update downloaded',
      spinning: false,
      tone: 'ready'
    }
  }

  return {
    detail: 'The background update could not be downloaded. Hermes will try again later.',
    icon: 'error',
    label: 'Update download failed',
    spinning: false,
    tone: 'error'
  }
}

export function ManagedUpdateIndicator() {
  const snapshot = useStore($managedUpdate)
  const presentation = managedUpdatePresentation(snapshot)

  if (!presentation) {
    return null
  }

  const progress = snapshot?.stage === 'downloading' ? snapshot.percent : null

  return (
    <Tip label={presentation.detail} side="top">
      <div
        aria-label={presentation.label}
        aria-live="polite"
        className={cn(
          'relative flex h-7 max-w-36 shrink-0 items-center gap-1.5 overflow-hidden rounded-md border px-2 text-[10px] font-medium',
          presentation.tone === 'muted' && 'border-border/55 bg-muted/45 text-muted-foreground',
          presentation.tone === 'ready' && 'border-emerald-500/25 bg-emerald-500/10 text-emerald-600',
          presentation.tone === 'error' && 'border-destructive/25 bg-destructive/10 text-destructive'
        )}
        data-stage={snapshot?.stage}
        role="status"
      >
        <Codicon className="shrink-0" name={presentation.icon} size="0.75rem" spinning={presentation.spinning} />
        <span className="truncate">{presentation.label}</span>
        {progress !== null && (
          <span aria-hidden className="absolute inset-x-0 bottom-0 h-0.5 bg-foreground/10">
            <span
              className="block h-full bg-primary transition-[width] duration-300 ease-out"
              style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
            />
          </span>
        )}
      </div>
    </Tip>
  )
}
