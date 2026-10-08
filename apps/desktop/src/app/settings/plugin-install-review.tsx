import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { AlertTriangle } from '@/lib/icons'
import type { PluginSourceLinks } from '@/lib/plugin-source-urls'
import type { PluginInstallRequest } from '@/store/plugin-install-request'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)]'

/** Repository identity, catalog pin and where to read the code before installing it. */
export function PluginSourceReview({
  request,
  sourceLinks
}: {
  request: PluginInstallRequest & { repo: string }
  sourceLinks: PluginSourceLinks | null
}) {
  const { t } = useI18n()
  const m = t.settings.plugins.installModal

  return (
    <>
      <div>
        <div className={`mb-1 ${CAPTION} font-medium text-foreground`}>{m.repoLabel}</div>
        <div
          className={`rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2 font-mono ${CAPTION} break-all text-foreground`}
        >
          {request.repo}
        </div>
        {request.catalogName && (
          <p className={`mt-1 ${CAPTION} text-(--ui-text-tertiary)`}>
            {m.catalogPinned(request.catalogName, request.sha?.slice(0, 8) ?? '')}
          </p>
        )}
      </div>

      <div className="space-y-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2.5">
        <div className={`space-y-2 ${CAPTION}`}>
          <div className="font-medium text-foreground">
            {request.catalogName ? m.reviewedHeading : m.securityHeading}
          </div>
          <p className="text-(--ui-text-secondary)">{request.catalogName ? m.reviewedIntro : m.securityIntro}</p>
        </div>

        {sourceLinks && (
          <div className="space-y-2 border-t border-(--ui-stroke-tertiary) pt-3">
            <div className="font-medium text-foreground">{m.sourceHeading}</div>
            {sourceLinks.browseUrl && (
              <ExternalLink className={CAPTION} href={sourceLinks.browseUrl} showExternalIcon>
                {sourceLinks.subdir ? m.viewPluginFiles : m.viewRepository}
              </ExternalLink>
            )}
            <div>
              <div className="mb-1 text-(--ui-text-tertiary)">{m.gitCloneLabel}</div>
              <div className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-primary) px-2.5 py-1.5 font-mono break-all text-foreground">
                {sourceLinks.gitUrl}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

/** Probe warnings (insecure transport, packaging notes), de-duplicated into one callout. */
export function ProbeWarnings({ insecure, warnings }: { insecure?: boolean; warnings?: string[] }) {
  const { t } = useI18n()

  const text = [...new Set([...(warnings ?? []), insecure ? t.settings.plugins.installModal.insecureWarning : ''])]
    .filter(Boolean)
    .join(' ')

  return text ? (
    <div
      className={`flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 ${CAPTION} text-foreground`}
    >
      <AlertTriangle aria-hidden className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
      <span>{text}</span>
    </div>
  ) : null
}
