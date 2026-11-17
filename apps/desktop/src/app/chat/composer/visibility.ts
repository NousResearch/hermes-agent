import { createContext, useContext } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'

/** Editor visibility is narrower than chat visibility while a shared float is
 * active. Background queues and explicit session actions still belong to the pane. */
export const ComposerVisibleContext = createContext<boolean | null>(null)

export function useComposerVisible(): boolean {
  const composer = useContext(ComposerVisibleContext)
  const pane = usePaneVisible()

  return composer ?? pane
}
