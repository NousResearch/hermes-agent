import { useEffect, useState } from 'react'

import type { HermesLocalFilesPolicy } from '@/global'

const LOCAL_FILES_ALLOWED: HermesLocalFilesPolicy = { disabled: false, reason: null }

// Remote-only mode is fixed for the lifetime of the process (it comes from the
// launch environment), so a single read on mount is enough. Older bridges that
// predate the policy handler simply report "allowed".
export function useLocalFilesPolicy(): HermesLocalFilesPolicy {
  const [policy, setPolicy] = useState<HermesLocalFilesPolicy>(LOCAL_FILES_ALLOWED)

  useEffect(() => {
    let cancelled = false
    const read = window.hermesDesktop?.localFilesPolicy

    if (!read) {
      return
    }

    read()
      .then(next => {
        if (!cancelled && next) {
          setPolicy(next)
        }
      })
      .catch(() => {
        // Treat a failed probe as "allowed" — the main-process IPC guard is the
        // real boundary, so a missing banner never weakens enforcement.
      })

    return () => {
      cancelled = true
    }
  }, [])

  return policy
}
