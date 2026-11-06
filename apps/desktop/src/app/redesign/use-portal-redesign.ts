import { useStore } from '@nanostores/react'

import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'

/** Shell layout is independent of the profile's skin and light/dark mode. */
export function usePortalRedesign() {
  return useStore($simplifiedAgentMode)
}
