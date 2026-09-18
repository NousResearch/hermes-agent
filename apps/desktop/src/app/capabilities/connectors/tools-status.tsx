// Everything the right column shows instead of a tool list. Each state is its own
// honest copy and its own way out — a failed fetch, a connector that left the
// catalog, an expired sign-in and a lost race are four different problems, and
// one spinner for all of them tells the reader nothing.

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import type { ConflictDifference } from './types'

/** Cold start. A static wash, not a shimmer loop: the page does not repaint by
 *  itself, so an animation would promise progress that is not being made. */
export function ToolsWash({ label, rows = 8 }: { label?: string; rows?: number }) {
  const { t } = useI18n()

  return (
    <div aria-busy className="grid gap-2 px-3.5 py-3" role="status">
      <span className="sr-only">{label ?? t.connectorsPage.tools.loading}</span>
      {Array.from({ length: rows }, (_, index) => (
        <span
          aria-hidden
          className="h-3 rounded-sm bg-(--ui-bg-quaternary)"
          key={index}
          // A wash, not a bar chart: the widths only stop the block reading as a
          // solid rectangle.
          style={{ width: `${68 - (index % 4) * 9}%` }}
        />
      ))}
    </div>
  )
}

export function ToolsUnavailable({ onRetry }: { onRetry: () => void }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <PanelEmpty
      action={
        <Button onClick={onRetry} size="xs" variant="secondary">
          {copy.retry}
        </Button>
      }
      description={copy.unavailableBody}
      icon="warning"
      title={copy.unavailableTitle}
    />
  )
}

export function ToolsGone({ connectorName, onRemove }: { connectorName: string; onRemove: () => void }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <PanelEmpty
      action={
        <Button onClick={onRemove} size="xs" variant="secondary">
          {copy.remove}
        </Button>
      }
      description={copy.goneBody}
      icon="circle-slash"
      title={copy.goneTitle(connectorName)}
    />
  )
}

export function ToolsSignedOut({ onSignIn }: { onSignIn: () => void }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <PanelEmpty
      action={
        <Button onClick={onSignIn} size="xs">
          {copy.signIn}
        </Button>
      }
      description={copy.signedOutBody}
      icon="sign-in"
      title={copy.signedOutTitle}
    />
  )
}

/** The lost race. Nothing merges and nothing was written, so both ways out are
 *  offered in full words and neither is the default. */
export function ToolsConflict({
  difference,
  onKeepMine,
  onReload
}: {
  difference: ConflictDifference
  onKeepMine: () => void
  onReload: () => void
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <PanelEmpty
      action={
        <div className="flex items-center gap-2">
          <Button onClick={onReload} size="xs" variant="secondary">
            {copy.conflictReload}
          </Button>
          <Button onClick={onKeepMine} size="xs">
            {copy.conflictSave}
          </Button>
        </div>
      }
      description={copy.conflictBody(difference.theyOff, difference.theyOn)}
      icon="git-merge"
      title={copy.conflictTitle}
    />
  )
}
