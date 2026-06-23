import { useCallback, useEffect, useRef, useState } from 'react'

import type { HermesGitStatusResult } from '@/global'

const EMPTY_STATUS: HermesGitStatusResult = {
  branch: { ahead: 0, behind: 0, detached: false, name: '', oid: '', upstream: '' },
  entries: [],
  root: null
}

const cache = new Map<string, HermesGitStatusResult>()

export function useGitStatus(cwd: string, active: boolean) {
  const [status, setStatus] = useState(() => cache.get(cwd) ?? EMPTY_STATUS)
  const [loading, setLoading] = useState(false)
  const requestId = useRef(0)

  const refresh = useCallback(async () => {
    if (!cwd || !window.hermesDesktop?.gitStatus) {
      setStatus(EMPTY_STATUS)

      return
    }

    const currentRequest = ++requestId.current
    setLoading(true)

    try {
      const next = await window.hermesDesktop.gitStatus(cwd)

      if (currentRequest === requestId.current) {
        cache.set(cwd, next)
        setStatus(next)
      }
    } finally {
      if (currentRequest === requestId.current) {
        setLoading(false)
      }
    }
  }, [cwd])

  useEffect(() => {
    setStatus(cache.get(cwd) ?? EMPTY_STATUS)

    if (active) {
      void refresh()
    }
  }, [active, cwd, refresh])

  useEffect(() => {
    if (!active || !cwd) {
      return
    }

    const onFocus = () => void refresh()

    const interval = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh()
      }
    }, 3000)

    window.addEventListener('focus', onFocus)

    return () => {
      window.clearInterval(interval)
      window.removeEventListener('focus', onFocus)
    }
  }, [active, cwd, refresh])

  return { loading, refresh, status }
}
