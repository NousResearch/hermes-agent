import { useCallback, useEffect, useState } from 'react'

import { broadcastDesktopStateChange, onDesktopStateSync } from '@/lib/desktop-state-sync'

export function useMenuBarCompanion() {
  const [enabled, setEnabledState] = useState(false)

  useEffect(() => {
    let cancelled = false

    const unsubscribe = onDesktopStateSync(message => {
      if (message.type === 'changed' && message.domain === 'menu-bar-companion' && typeof message.value === 'boolean') {
        setEnabledState(message.value)
      }
    })

    void window.hermesDesktop?.settings
      ?.getMenuBarCompanionEnabled?.()
      .then(result => {
        if (!cancelled) {
          setEnabledState(result.enabled)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setEnabledState(false)
        }
      })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  const setEnabled = useCallback(
    async (next: boolean): Promise<boolean> => {
      const previous = enabled
      const save = window.hermesDesktop?.settings?.setMenuBarCompanionEnabled

      setEnabledState(next)
      // Broadcast before the IPC call because turning this off destroys the
      // companion renderer before its promise can settle.
      broadcastDesktopStateChange('menu-bar-companion', { value: next })

      if (!save) {
        setEnabledState(previous)
        broadcastDesktopStateChange('menu-bar-companion', { value: previous })
        throw new Error('Companion preference control unavailable')
      }

      try {
        const result = await save(next)
        setEnabledState(result.enabled)

        if (result.enabled !== next) {
          broadcastDesktopStateChange('menu-bar-companion', { value: result.enabled })
        }

        return result.enabled
      } catch (error) {
        setEnabledState(previous)
        broadcastDesktopStateChange('menu-bar-companion', { value: previous })
        throw error
      }
    },
    [enabled]
  )

  return { enabled, setEnabled }
}
