import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionFallbackRoute } from '@/lib/session-effective-route'

/** The card footer's model: the model that really answers. While a provider fallback stands in
 *  for the picked model it names the fallback, tinted, with the requested model in the tooltip. */
export function SessionCardModel({ model, session }: { model: string; session: SessionInfo }) {
  const { t } = useI18n()
  const fallback = sessionFallbackRoute(session)

  if (!fallback) {
    return <span className="min-w-0 truncate">{model}</span>
  }

  return (
    <span
      className="min-w-0 truncate text-amber-600 dark:text-amber-400"
      data-effective-route="fallback"
      title={t.shell.modelMenu.effectiveFallback(fallback.effective, fallback.requested)}
    >
      {fallback.effective}
    </span>
  )
}
