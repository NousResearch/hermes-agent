import type { DesktopVersionInfo, RuntimeSource } from '@/global'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'

/**
 * Human label for an external build's runtime source: the resolution rung
 * plus the location it resolved from, when there is one.
 */
function runtimeSourceLabel(source: RuntimeSource): string {
  const where = 'root' in source && source.root ? source.root : 'command' in source ? source.command : null

  return where ? `${source.type} (${where})` : source.type
}

/**
 * Shared build-provenance display. Reads from `$desktopVersion`
 * (populated from the build stamp / `hermes:version` IPC), so every
 * surface — the About settings page, the updates overlay — shows the
 * same version, branch, commit, distribution, runtime, and install id
 * from one source of truth.
 */
export function VersionDetails({ version }: { version: DesktopVersionInfo }) {
  const { t } = useI18n()
  const u = t.updates

  const source =
    version.source === 'ci' ? 'CI' : version.source ? version.source[0].toUpperCase() + version.source.slice(1) : null

  const distribution =
    version.distribution === 'nix'
      ? 'Nix'
      : version.distribution === 'docker'
        ? 'Docker'
        : version.distribution === 'desktop-app'
          ? u.versionDetailsDistributionDesktop
          : null

  const runtime =
    version.hermesRuntime?.type === 'embedded'
      ? u.versionDetailsRuntimeEmbedded
      : version.hermesRuntime?.type === 'external' && version.hermesRuntime.source
        ? runtimeSourceLabel(version.hermesRuntime.source)
        : version.hermesRuntime?.type === 'external'
          ? u.versionDetailsRuntimeExternal
          : null

  return (
    <dl className="grid gap-2 rounded-lg border border-border/70 bg-muted/20 px-3 py-3 text-sm">
      <div className="flex justify-between gap-4">
        <dt className="text-muted-foreground">{u.versionDetailsVersion}</dt>
        <dd>
          v{version.appVersion}
          {version.dirty && <span className="text-warning"> (!)</span>}
        </dd>
      </div>
      {version.commit && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsCommit}</dt>
          <dd className="break-all text-right">
            <ExternalLink
              className="break-all font-mono text-xs"
              href={`https://github.com/NousResearch/hermes-agent/commit/${version.commit}`}
            >
              {version.commit.slice(0, 14)}
            </ExternalLink>
            {version.dirty && <span className="text-warning"> {u.versionDetailsUncommittedChanges}</span>}
          </dd>
        </div>
      )}
      {source && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsBuildOrigin}</dt>
          <dd>
            {source}
            {version.branch && version.branch !== 'main' ? ` (${version.branch})` : ''}
          </dd>
        </div>
      )}
      {distribution && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsDistribution}</dt>
          <dd>{distribution}</dd>
        </div>
      )}
      {runtime && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsRuntime}</dt>
          <dd className="break-all text-right">{runtime}</dd>
        </div>
      )}
      {version.installId && (
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{u.versionDetailsInstallId}</dt>
          <dd className="break-all text-right font-mono text-xs">
            {version.installId} ({version.hermesRoot})
          </dd>
        </div>
      )}
    </dl>
  )
}
