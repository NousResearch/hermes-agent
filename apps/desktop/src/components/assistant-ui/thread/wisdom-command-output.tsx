import { useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { WisdomCheckBadge, WisdomReviewTables } from '@/components/wisdom-checks'
import {
  getWisdomSkill,
  getWisdomVersion,
  type WisdomReviewCheck,
  type WisdomSkillDetail,
  type WisdomVersionDetail
} from '@/hermes'
import { useI18n } from '@/i18n'
import { LinkifiedText, openExternalLink } from '@/lib/external-link'
import { ChevronLeft, ChevronRight, Clock, Download, ExternalLink, Eye, Loader2Icon } from '@/lib/icons'

const WISDOM_VIEW_RE = /^View:\s+\/wisdom\s+show\s+(\S+)\s*$/i

type PreviewState =
  | { detail: null; error: null; status: 'idle' | 'loading' }
  | { detail: null; error: string; status: 'error' }
  | { detail: WisdomSkillDetail; error: null; status: 'ready' }

type VersionState =
  | { detail: null; error: null; status: 'idle' | 'loading' }
  | { detail: null; error: string; status: 'error' }
  | { detail: WisdomVersionDetail; error: null; status: 'ready' }

interface VersionSummary {
  authorDescription: string
  explanation: string
  publishedAt: string
  scanVerdict: string
  securityCheck?: WisdomReviewCheck
  professionalismCheck?: WisdomReviewCheck
  version: number
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function asText(...values: unknown[]): string {
  return String(values.find(value => typeof value === 'string' && value.trim()) ?? '').trim()
}

function versionNumber(value: unknown): null | number {
  const parsed = Number(value)

  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function labelToken(value: string): string {
  return value.replaceAll('_', ' ').replaceAll('-', ' ')
}

function formatPublishedAt(value: string): string {
  if (!value) {
    return ''
  }

  const date = new Date(value)

  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function parseVersionSummary(value: Record<string, unknown>): null | VersionSummary {
  const version = versionNumber(value.version)

  if (!version) {
    return null
  }

  const verifiedFacts = asRecord(value.verified_facts)
  const scan = asRecord(value.scan)

  return {
    authorDescription: asText(value.author_description, value.authorDescription),
    explanation: asText(value.explanation),
    publishedAt: asText(value.published_at, value.created_at),
    scanVerdict: asText(verifiedFacts.scan_verdict, scan.verdict),
    securityCheck: value.security_check as WisdomReviewCheck | undefined,
    professionalismCheck: value.professionalism_check as WisdomReviewCheck | undefined,
    version
  }
}

function requirementsFrom(specification: Record<string, unknown>): string[] {
  const requirements = asRecord(specification.requirements)
  const source = Object.keys(requirements).length ? requirements : specification
  const hermes = asRecord(source.hermes)
  const runtime = asRecord(source.runtime)
  const minimumVersion = asText(hermes.minimum_version)

  const list = (key: string) =>
    Array.isArray(source[key])
      ? source[key].filter((item): item is string => typeof item === 'string' && Boolean(item.trim()))
      : []

  return [
    ...(minimumVersion ? [`Hermes ≥ ${minimumVersion}`] : []),
    ...list('platforms'),
    ...list('architectures'),
    ...Object.entries(runtime)
      .filter(([, enabled]) => enabled === true)
      .map(([name]) => labelToken(name))
  ]
}

function MetadataBadges({
  compatibility,
  scanVerdict,
  version
}: {
  compatibility: string
  scanVerdict: string
  version: null | number
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {version ? <Badge variant="outline">v{version}</Badge> : null}
      {scanVerdict ? <Badge>{labelToken(scanVerdict)}</Badge> : null}
      {compatibility ? <Badge variant="muted">{labelToken(compatibility)}</Badge> : null}
    </div>
  )
}

function Requirements({ label, specification }: { label: string; specification: Record<string, unknown> }) {
  const requirements = requirementsFrom(specification)

  if (!requirements.length && !Object.keys(specification).length) {
    return null
  }

  return (
    <section aria-label={label}>
      <h3 className="text-xs font-medium">{label}</h3>
      {requirements.length ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {requirements.map(requirement => (
            <Badge key={requirement} variant="outline">
              {requirement}
            </Badge>
          ))}
        </div>
      ) : (
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded-md bg-muted/40 p-2 text-[0.67rem] text-muted-foreground">
          {JSON.stringify(specification, null, 2)}
        </pre>
      )}
    </section>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex min-h-32 items-center justify-center gap-2 text-xs text-muted-foreground" role="status">
      <Loader2Icon className="size-4 animate-spin" />
      {label}
    </div>
  )
}

function ErrorState({ error }: { error: string }) {
  return (
    <div
      className="rounded-md border border-destructive/50 bg-destructive/8 px-3 py-2 text-xs text-destructive"
      role="alert"
    >
      {error}
    </div>
  )
}

function WisdomSkillPreview({ onClose, skillId }: { onClose: () => void; skillId: string }) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [state, setState] = useState<PreviewState>({ detail: null, error: null, status: 'idle' })
  const [showVersions, setShowVersions] = useState(false)
  const [selectedVersion, setSelectedVersion] = useState<null | number>(null)
  const [versionState, setVersionState] = useState<VersionState>({ detail: null, error: null, status: 'idle' })

  useEffect(() => {
    let current = true

    setState({ detail: null, error: null, status: 'loading' })
    void getWisdomSkill(skillId)
      .then(detail => {
        if (current) {
          setState({ detail, error: null, status: 'ready' })
        }
      })
      .catch(error => {
        if (current) {
          setState({
            detail: null,
            error: error instanceof Error ? error.message : copy.unavailable,
            status: 'error'
          })
        }
      })

    return () => {
      current = false
    }
  }, [copy.unavailable, skillId])

  const preview = useMemo(() => {
    if (state.status !== 'ready') {
      return null
    }

    const skill = asRecord(state.detail.skill)
    const latest = asRecord(state.detail.latest_version_detail)
    const version = asRecord(latest.version)
    const compatibility = asRecord(state.detail.local_compatibility)
    const installation = asRecord(state.detail.local_installation)
    const scan = asRecord(version.scan)
    const verifiedFacts = asRecord(version.verified_facts)
    const systemSpec = asRecord(version.system_spec)

    const versions = state.detail.versions
      .map(item => parseVersionSummary(asRecord(item)))
      .filter((item): item is VersionSummary => item !== null)
      .sort((a, b) => b.version - a.version)

    const latestVersion = versionNumber(version.version) ?? versions[0]?.version ?? null

    return {
      compatibility: asText(compatibility.outcome),
      description: asText(
        skill.authorDescription,
        skill.author_description,
        version.authorDescription,
        version.author_description
      ),
      installId: asText(skill.id, skillId) || skillId,
      installedVersion: versionNumber(installation.version),
      installedUpdateMode: asText(installation.update_mode),
      latestVersion,
      portalUrl: asText(state.detail.portal_url),
      scanVerdict: asText(scan.verdict, verifiedFacts.scan_verdict),
      securityCheck: version.security_check as WisdomReviewCheck | undefined,
      professionalismCheck: version.professionalism_check as WisdomReviewCheck | undefined,
      slug: asText(skill.slug, skill.id, skillId) || skillId,
      systemSpec,
      versions
    }
  }, [skillId, state])

  useEffect(() => {
    if (!selectedVersion || !preview) {
      setVersionState({ detail: null, error: null, status: 'idle' })

      return
    }

    let current = true

    setVersionState({ detail: null, error: null, status: 'loading' })
    void getWisdomVersion(preview.installId, selectedVersion)
      .then(detail => {
        if (current) {
          setVersionState({ detail, error: null, status: 'ready' })
        }
      })
      .catch(error => {
        if (current) {
          setVersionState({
            detail: null,
            error: error instanceof Error ? error.message : copy.unavailable,
            status: 'error'
          })
        }
      })

    return () => {
      current = false
    }
  }, [copy.unavailable, preview, selectedVersion])

  const versionPreview = useMemo(() => {
    if (versionState.status !== 'ready') {
      return null
    }

    const version = asRecord(versionState.detail.version)
    const compatibility = asRecord(versionState.detail.local_compatibility)
    const scan = asRecord(version.scan)
    const verifiedFacts = asRecord(version.verified_facts)

    return {
      authorDescription: asText(version.author_description, version.authorDescription),
      commit: asText(version.commit),
      compatibility: asText(compatibility.outcome),
      contentHash: asText(version.content_hash),
      explanation: asText(version.explanation),
      packageManifestHash: asText(version.package_manifest_hash),
      portalUrl: asText(versionState.detail.portal_url),
      publishedAt: asText(version.published_at),
      scan,
      scanVerdict: asText(scan.verdict, verifiedFacts.scan_verdict),
      securityCheck: version.security_check as WisdomReviewCheck | undefined,
      professionalismCheck: version.professionalism_check as WisdomReviewCheck | undefined,
      systemSpec: asRecord(version.system_spec),
      verifiedFacts,
      version: versionNumber(version.version)
    }
  }, [versionState])

  const startInstall = (version?: number) => {
    if (!preview) {
      return
    }

    const params = new URLSearchParams({
      tab: 'collective',
      wisdomAction: 'install',
      wisdomSkillId: version ? `${preview.installId}@v${version}` : preview.installId
    })

    onClose()
    window.location.hash = `#/skills?${params.toString()}`
  }

  const returnToOverview = () => {
    setSelectedVersion(null)
    setShowVersions(false)
  }

  const returnToVersions = () => {
    setSelectedVersion(null)
    setShowVersions(true)
  }

  const screen = selectedVersion ? 'version' : showVersions ? 'versions' : 'overview'
  const portalUrl = screen === 'version' ? versionPreview?.portalUrl : preview?.portalUrl

  const dialogBadge =
    screen === 'version' && selectedVersion
      ? `v${selectedVersion}`
      : screen === 'versions'
        ? copy.versions
        : copy.preview

  const dialogDescription =
    screen === 'version' ? versionPreview?.authorDescription : screen === 'overview' ? preview?.description : ''

  return (
    <Dialog onOpenChange={open => !open && onClose()} open>
      <DialogContent className="max-w-[42rem]" onOpenAutoFocus={event => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex min-w-0 items-center gap-2 pr-6">
            <span className="min-w-0 truncate font-mono">{preview?.slug || skillId}</span>
            <Badge className="shrink-0" variant="muted">
              {dialogBadge}
            </Badge>
          </DialogTitle>
          {dialogDescription ? (
            <DialogDescription className="text-left leading-5">{dialogDescription}</DialogDescription>
          ) : null}
        </DialogHeader>

        <div className="max-h-[min(65vh,38rem)] overflow-y-auto pr-1">
          {state.status === 'loading' || state.status === 'idle' ? <LoadingState label={copy.loading} /> : null}
          {state.status === 'error' ? <ErrorState error={state.error} /> : null}

          {preview && screen === 'overview' ? (
            <div className="grid gap-4">
              <MetadataBadges
                compatibility={preview.compatibility}
                scanVerdict={preview.scanVerdict}
                version={preview.latestVersion}
              />
              <div className="flex flex-wrap gap-1.5">
                <WisdomCheckBadge label="Security" value={preview.securityCheck} />
                <WisdomCheckBadge label="Professionalism" value={preview.professionalismCheck} />
              </div>
              <WisdomReviewTables professionalism={preview.professionalismCheck} security={preview.securityCheck} />
              <Requirements label={copy.systemSpecification} specification={preview.systemSpec} />
              {preview.versions.length ? (
                <section aria-label={copy.versionHistory}>
                  <h3 className="text-xs font-medium">{copy.versionHistory}</h3>
                  <p className="mt-1 text-[0.68rem] text-muted-foreground">
                    {preview.versions.map(item => `v${item.version}`).join(' · ')}
                  </p>
                </section>
              ) : null}
            </div>
          ) : null}

          {preview && screen === 'versions' ? (
            <section aria-label={copy.versionHistory}>
              <div className="divide-y divide-(--ui-stroke-tertiary) border-y border-(--ui-stroke-tertiary)">
                {preview.versions.map(item => (
                  <button
                    className="row-hover flex w-full items-start justify-between gap-4 px-1 py-3 text-left focus-visible:outline focus-visible:outline-2"
                    key={item.version}
                    onClick={() => setSelectedVersion(item.version)}
                    type="button"
                  >
                    <span className="min-w-0">
                      <span className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-sm">v{item.version}</span>
                        {item.scanVerdict ? <Badge>{labelToken(item.scanVerdict)}</Badge> : null}
                        <WisdomCheckBadge label="Security" value={item.securityCheck} />
                        <WisdomCheckBadge label="Professionalism" value={item.professionalismCheck} />
                      </span>
                      <span className="mt-1 block text-xs leading-5 text-(--ui-text-secondary)">
                        {item.authorDescription || item.explanation || copy.immutableVersion}
                      </span>
                      {item.publishedAt ? (
                        <time className="mt-1 block text-[0.65rem] text-muted-foreground" dateTime={item.publishedAt}>
                          {copy.published(formatPublishedAt(item.publishedAt))}
                        </time>
                      ) : null}
                    </span>
                    <ChevronRight className="mt-1 size-4 shrink-0 text-muted-foreground" />
                  </button>
                ))}
              </div>
            </section>
          ) : null}

          {screen === 'version' && (versionState.status === 'idle' || versionState.status === 'loading') ? (
            <LoadingState label={copy.loading} />
          ) : null}
          {screen === 'version' && versionState.status === 'error' ? <ErrorState error={versionState.error} /> : null}

          {screen === 'version' && versionPreview ? (
            <div className="grid gap-4">
              <MetadataBadges
                compatibility={versionPreview.compatibility}
                scanVerdict={versionPreview.scanVerdict}
                version={versionPreview.version}
              />
              <WisdomReviewTables
                professionalism={versionPreview.professionalismCheck}
                security={versionPreview.securityCheck}
              />
              {versionPreview.publishedAt ? (
                <time className="text-xs text-muted-foreground" dateTime={versionPreview.publishedAt}>
                  {copy.published(formatPublishedAt(versionPreview.publishedAt))}
                </time>
              ) : null}
              <Requirements label={copy.systemSpecification} specification={versionPreview.systemSpec} />
              {versionPreview.explanation ? (
                <section aria-label={copy.releaseExplanation}>
                  <h3 className="text-xs font-medium">{copy.releaseExplanation}</h3>
                  <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{versionPreview.explanation}</p>
                </section>
              ) : null}
              {Object.keys(versionPreview.scan).length || Object.keys(versionPreview.verifiedFacts).length ? (
                <section aria-label={copy.serverFactsLabel}>
                  <h3 className="text-xs font-medium">{copy.serverFactsLabel}</h3>
                  <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-muted/40 p-2 text-[0.67rem] text-muted-foreground">
                    {JSON.stringify(
                      { scan: versionPreview.scan, verified_facts: versionPreview.verifiedFacts },
                      null,
                      2
                    )}
                  </pre>
                </section>
              ) : null}
              {versionPreview.commit || versionPreview.contentHash || versionPreview.packageManifestHash ? (
                <section aria-label={copy.immutableVersion}>
                  <h3 className="text-xs font-medium">{copy.immutableVersion}</h3>
                  <dl className="mt-2 grid gap-2 text-[0.67rem]">
                    {versionPreview.commit ? (
                      <div>
                        <dt className="text-muted-foreground">commit</dt>
                        <dd className="break-all font-mono">{versionPreview.commit}</dd>
                      </div>
                    ) : null}
                    {versionPreview.contentHash ? (
                      <div>
                        <dt className="text-muted-foreground">{copy.contentHash}</dt>
                        <dd className="break-all font-mono">{versionPreview.contentHash}</dd>
                      </div>
                    ) : null}
                    {versionPreview.packageManifestHash ? (
                      <div>
                        <dt className="text-muted-foreground">{copy.packageManifestHash}</dt>
                        <dd className="break-all font-mono">{versionPreview.packageManifestHash}</dd>
                      </div>
                    ) : null}
                  </dl>
                </section>
              ) : null}
            </div>
          ) : null}
        </div>

        <DialogFooter className="sm:justify-between">
          <div className="mr-auto flex items-center gap-2">
            {screen === 'versions' ? (
              <Button onClick={returnToOverview} size="sm" variant="ghost">
                <ChevronLeft className="size-3.5" />
                {copy.backToSkill}
              </Button>
            ) : null}
            {screen === 'version' ? (
              <Button onClick={returnToVersions} size="sm" variant="ghost">
                <ChevronLeft className="size-3.5" />
                {copy.backToVersions}
              </Button>
            ) : null}
            {screen === 'overview' && preview?.installedVersion ? (
              <span className="self-center text-[0.68rem] text-muted-foreground">
                {copy.installed(preview.installedVersion, preview.installedUpdateMode || 'MANUAL')}
              </span>
            ) : null}
          </div>
          <div className="flex flex-wrap justify-end gap-2">
            <Button onClick={onClose} size="sm" variant="outline">
              {copy.close}
            </Button>
            {screen === 'overview' && preview?.versions.length ? (
              <Button onClick={() => setShowVersions(true)} size="sm" variant="outline">
                <Clock className="size-3.5" />
                {copy.versions}
              </Button>
            ) : null}
            {portalUrl ? (
              <Button onClick={() => openExternalLink(portalUrl)} size="sm" variant="outline">
                <ExternalLink className="size-3.5" />
                {copy.viewInPortal}
              </Button>
            ) : null}
            {screen === 'overview' && preview && !preview.installedVersion ? (
              <Button onClick={() => startInstall()} size="sm">
                <Download className="size-3.5" />
                {copy.install}
              </Button>
            ) : null}
            {screen === 'version' && preview && selectedVersion ? (
              <Button onClick={() => startInstall(selectedVersion)} size="sm">
                <Download className="size-3.5" />
                {copy.install}
              </Button>
            ) : null}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function WisdomCommandOutput({ detail, headline }: { detail: string; headline: string }) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [previewSkillId, setPreviewSkillId] = useState<null | string>(null)
  const lines = detail.split('\n')

  return (
    <>
      <div className="mt-1.5 block">
        <LinkifiedText className="block font-medium text-foreground/90" explicitOnly pretty={false} text={headline} />
        {detail ? (
          <div className="mt-1 whitespace-pre-wrap">
            {lines.map((line, index) => {
              const view = line.match(WISDOM_VIEW_RE)

              return (
                <span key={`${index}:${line}`}>
                  {view ? (
                    <Button
                      aria-label={`${copy.preview}: ${view[1]}`}
                      onClick={() => setPreviewSkillId(view[1])}
                      size="inline"
                      variant="link"
                    >
                      <Eye className="size-3" />
                      {copy.preview}
                    </Button>
                  ) : (
                    <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={line} />
                  )}
                  {index < lines.length - 1 ? '\n' : null}
                </span>
              )
            })}
          </div>
        ) : null}
      </div>
      {previewSkillId ? <WisdomSkillPreview onClose={() => setPreviewSkillId(null)} skillId={previewSkillId} /> : null}
    </>
  )
}
