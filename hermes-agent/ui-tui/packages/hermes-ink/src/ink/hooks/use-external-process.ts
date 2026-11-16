import { useCallback } from 'react'

import instances from '../instances.js'

export type RunExternalProcess = () => Promise<void>

export async function withInkSuspended(run: RunExternalProcess): Promise<void> {
  const ink = instances.get(process.stdout)

  if (!ink) {
    await run()

    return
  }

  ink.enterAlternateScreen()

  try {
    await run()
  } finally {
    ink.exitAlternateScreen()
  }
}

export function useExternalProcess(): (run: RunExternalProcess) => Promise<void> {
  return useCallback((run: RunExternalProcess) => withInkSuspended(run), [])
}
