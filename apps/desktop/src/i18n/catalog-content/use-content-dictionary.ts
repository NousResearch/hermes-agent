import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $contentDictionaryVersion, loadContentDictionary } from './index'

/**
 * Ensure the content dictionary for `locale` is loading and re-render when it arrives. Returns the
 * dictionary version so memoized localization can depend on it.
 */
export function useContentDictionary(locale: string): number {
  const version = useStore($contentDictionaryVersion)

  useEffect(() => {
    void loadContentDictionary(locale)
  }, [locale])

  return version
}
