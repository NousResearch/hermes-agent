import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'

import type { ConflictDifference, ToolsEditorPhase } from './types'

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
          style={{ width: `${68 - (index % 4) * 9}%` }}
        />
      ))}
    </div>
  )
}

type PageCopy = Translations['connectorsPage']
type ToolsCopy = PageCopy['tools']

export type ToolsStatusAction = 'keepMine' | 'reload' | 'remove' | 'signIn'
export type ToolsStatusPhase = 'conflict' | 'gone' | 'needsAuth' | 'off' | 'signedOut'

interface StatusContext {
  connectorName: string
  difference: ConflictDifference
}

interface StatusView {
  actions: { id: ToolsStatusAction; label: (copy: PageCopy) => string; variant?: 'secondary' }[]
  body: (copy: ToolsCopy, context: StatusContext) => string
  icon: string
  title: (copy: ToolsCopy, context: StatusContext) => string
}

type StatusTable = { readonly [phase in ToolsStatusPhase]: StatusView }

const STATUS: StatusTable = {
  conflict: {
    actions: [
      { id: 'reload', label: copy => copy.tools.conflictReload, variant: 'secondary' },
      { id: 'keepMine', label: copy => copy.tools.conflictSave }
    ],
    body: (copy, { difference }) => copy.conflictBody(difference.theyOff, difference.theyOn),
    icon: 'git-merge',
    title: copy => copy.conflictTitle
  },
  gone: {
    actions: [{ id: 'remove', label: copy => copy.tools.remove, variant: 'secondary' }],
    body: copy => copy.goneBody,
    icon: 'circle-slash',
    title: (copy, { connectorName }) => copy.goneTitle(connectorName)
  },
  needsAuth: {
    actions: [],
    body: copy => copy.needsAuthBody,
    icon: 'key',
    title: (copy, { connectorName }) => copy.needsAuthTitle(connectorName)
  },
  off: {
    actions: [],
    body: copy => copy.offBody,
    icon: 'plug',
    title: (copy, { connectorName }) => copy.offTitle(connectorName)
  },
  signedOut: {
    actions: [{ id: 'signIn', label: copy => copy.page.signIn }],
    body: copy => copy.signedOutBody,
    icon: 'sign-in',
    title: copy => copy.signedOutTitle
  }
}

export function isToolsStatusPhase(phase: ToolsEditorPhase): phase is ToolsStatusPhase {
  return phase in STATUS
}

export function ToolsStatus({
  connectorName,
  difference,
  onAction,
  phase
}: {
  connectorName: string
  difference: ConflictDifference
  onAction: (id: ToolsStatusAction) => void
  phase: ToolsStatusPhase
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools
  const view = STATUS[phase]
  const context = { connectorName, difference }

  return (
    <PanelEmpty
      action={
        view.actions.length > 0 ? (
          <div className="flex items-center gap-2">
            {view.actions.map(action => (
              <Button key={action.id} onClick={() => onAction(action.id)} size="xs" variant={action.variant}>
                {action.label(t.connectorsPage)}
              </Button>
            ))}
          </div>
        ) : undefined
      }
      description={view.body(copy, context)}
      icon={view.icon}
      title={view.title(copy, context)}
    />
  )
}
