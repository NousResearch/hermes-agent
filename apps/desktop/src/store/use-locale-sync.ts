import { useEffect, useState } from 'react'
import { onLocaleChange } from './i18n'

/**
 * Forces the component to re-render whenever the locale changes.
 * Returns a version number that can be used as a useMemo/useCallback dependency.
 */
export function useLocaleSync(): number {
  const [version, setVersion] = useState(0)
  useEffect(() => onLocaleChange(() => setVersion(n => n + 1)), [])
  return version
}
