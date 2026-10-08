import { useI18n } from '@/i18n'

const NOTE = 'text-[length:var(--conversation-caption-font-size)]'

/** Install-dialog copy for a memory provider: installs and admits it, never selects it. */
export function MemoryInstallConsent() {
  const { t } = useI18n()

  return <p className={`${NOTE} text-(--ui-text-tertiary)`}>{t.memoryDiscovery.installConsent}</p>
}

/** What happened after the install, or why it cannot run from the current owner. */
export function MemoryInstallStatus({
  ownerMatches,
  result
}: {
  ownerMatches: boolean
  result: 'discovered' | 'missing' | null
}) {
  const { t } = useI18n()
  const c = t.memoryDiscovery

  const message =
    result === 'discovered' ? c.installedNotice : result ? c.notDiscovered : ownerMatches ? null : c.ownerChanged

  return message ? (
    <p className={`${NOTE} text-(--ui-text-secondary)`} role="status">
      {message}
    </p>
  ) : null
}
