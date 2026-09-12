import {
  createContext,
  lazy,
  type PropsWithChildren,
  Suspense,
  useCallback,
  useContext,
  useEffect,
  useState
} from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

import type { ToolPart } from './fallback-model/types'

const InspectionDialog = lazy(() => import('./tool-inspection-dialog'))

interface InspectionTarget {
  part: ToolPart
  inlineDiff: string
  trigger: HTMLButtonElement
}

const InspectionContext = createContext<((target: InspectionTarget) => void) | null>(null)

/** A thread-local host survives ticker/list virtualization. No payload is
 * duplicated into a global store, persisted, or shared between session tiles. */
export function ToolInspectionProvider({ children, scope }: PropsWithChildren<{ scope: unknown }>) {
  const [inspection, setInspection] = useState<(InspectionTarget & { scope: unknown }) | null>(null)
  const open = useCallback((target: InspectionTarget) => setInspection({ ...target, scope }), [scope])
  const close = useCallback(() => setInspection(null), [])

  useEffect(() => setInspection(null), [scope])

  return (
    <InspectionContext.Provider value={open}>
      {children}
      {inspection && inspection.scope === scope && (
        <Suspense fallback={null}>
          <InspectionDialog
            inlineDiff={inspection.inlineDiff}
            onClose={close}
            part={inspection.part}
            trigger={inspection.trigger}
          />
        </Suspense>
      )}
    </InspectionContext.Provider>
  )
}

export function ToolInspectionButton({ part, inlineDiff }: { part: ToolPart; inlineDiff: string }) {
  const open = useContext(InspectionContext)
  const { t } = useI18n()
  const label = t.assistant.tool.inspector.open

  if (!open) {
    return null
  }

  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        onClick={event => {
          event.stopPropagation()
          open({ part, inlineDiff, trigger: event.currentTarget })
        }}
        size="icon-xs"
        variant="ghost"
      >
        <Codicon name="inspect" size="0.75rem" />
      </Button>
    </Tip>
  )
}
