import type { DesktopVersionInfo } from '@/global'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'

/**
 * Shared build-provenance display. Reads from `$desktopVersion`
 * (populated from the build stamp / `hermes:version` IPC), so every
 * surface — the About settings page, the updates overlay — shows the
 * same version, branch, commit, and dirty flag from one source of truth.
 */
export function VersionDetails({ version }: { version: DesktopVersionInfo }) {
  const { t } = useI18n()
  const u = t.updates
  const unknownDistance = version.dirty && version.distance == null
  const source = version.source === 'ci' ? 'CI' : version.source ? version.source[0].toUpperCase() + version.source.slice(1) : null
  const distribution = version.distribution === 'nix' ? 'Nix' : version.distribution === 'docker' ? 'Docker' : null

  return (
    <dl className="grid gap-2 rounded-lg border border-border/70 bg-muted/20 px-3 py-3 text-sm">
      <div className="flex justify-between gap-4">
        <dt className="text-muted-foreground">{u.versionDetailsVersion}</dt>
        <dd>v{version.appVersion}</dd>
      </div>
      {version.baseVersion && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsBaseVersion}</dt>
          <dd>{version.baseVersion}</dd>
        </div>
      )}
      <div className="flex justify-between gap-4">
        <dt className="text-muted-foreground">{u.versionDetailsBranch}</dt>
        <dd className="break-all text-right">{version.branch ?? u.versionDetailsNoBranchInfo}</dd>
      </div>
      {version.commit && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsCommit}</dt>
          <ExternalLink
            className="break-all font-mono text-xs"
            href={`https://github.com/NousResearch/hermes-agent/commit/${version.commit}`}
          >
            {version.commit.slice(0, 14)}
          </ExternalLink>
        </div>
      )}
      {source && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsSource}</dt>
          <dd>{source}</dd>
        </div>
      )}
      {distribution && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsDistribution}</dt>
          <dd>{distribution}</dd>
        </div>
      )}
      {version.dirty && (
        <div className="text-warning">{unknownDistance ? u.versionDetailsDirtyUnknown : u.versionDetailsDirty}</div>
      )}
    </dl>
  )
}
