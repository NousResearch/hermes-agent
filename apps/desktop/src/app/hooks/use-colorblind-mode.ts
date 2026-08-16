import { useEffect } from 'react'

import { useHermesConfigRecord } from './use-config-record'

/**
 * Bridges the `desktop.colorblind_mode` config toggle to a data attribute on
 * <html>. When enabled, `html[data-colorblind='true']` switches diff colors
 * to a colorblind-safe blue (add) / orange (remove) pair — see the matching
 * rules in styles.css. Off by default, so the default green/red diff palette
 * stays untouched until the user opts in.
 */
export function useColorblindMode() {
  const { data: config } = useHermesConfigRecord()
  const desktopCfg = (config?.desktop ?? {}) as { colorblind_mode?: boolean }
  const enabled = desktopCfg.colorblind_mode ?? false

  useEffect(() => {
    const root = document.documentElement
    if (enabled) {
      root.dataset.colorblind = 'true'
    } else {
      delete root.dataset.colorblind
    }
  }, [enabled])
}
