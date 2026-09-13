import { useI18n } from '@hermes/plugin-sdk'

import { useBots } from './i18n'

/** Hosted group copy stays plugin-owned; generic controls reuse the core catalog. */
export function useCanonicalGroupLabels() {
  const { t } = useI18n()
  const { canonical } = useBots()

  return {
    ...canonical,
    back: t.common.back,
    cancel: t.common.cancel,
    refresh: t.common.refresh,
    retry: t.common.retry,
    send: t.common.send,
    stop: t.composer.stop,
    discard: t.composer.queueLostDiscard,
    download: t.fileMenu.download
  }
}
