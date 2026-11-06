import type { ReactNode } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ExternalLink } from '@/lib/external-link'
import { resolvePluginSourceLinks } from '@/lib/plugin-source-urls'
import { cn } from '@/lib/utils'
import type { PluginInstallRequest } from '@/store/plugin-install-request'
import { normalizeProfileKey, profileLabel } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import {
  desktopHalfFromPackage,
  installComponents,
  type InstallModalCopy,
  type ProbeResult,
  skipsDesktopHalf
} from './plugin-install-run'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)]'

export interface PinState {
  value: string
  invalid: boolean
  onChange: (value: string) => void
}

export interface ProfileState {
  value: string
  label: string
  options: Pick<ProfileInfo, 'display_name' | 'name'>[]
  onChange: (value: string) => void
}

interface PluginInstallDetailsProps {
  copy: InstallModalCopy
  request: PluginInstallRequest
  probe: null | ProbeResult
  remote: boolean
  busy: boolean
  profile: ProfileState
  pin: PinState
  force: boolean
  onForceChange: (value: boolean) => void
  /** Offer Force for a catalog pick too: a partial install proceeds only by replacing both halves. */
  offerForce?: boolean
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid gap-1">
      <dt className={cn(CAPTION, 'text-(--ui-text-tertiary)')}>{label}</dt>
      <dd className={cn(CAPTION, 'min-w-0 text-foreground')}>{children}</dd>
    </div>
  )
}

function SourceRow({ copy, repo }: { copy: InstallModalCopy; repo: string }) {
  const links = resolvePluginSourceLinks(repo)

  return (
    <Row label={copy.sourceHeading}>
      <span className="block font-mono break-all">{repo}</span>
      {links?.browseUrl && (
        <ExternalLink href={links.browseUrl} showExternalIcon>
          {links.subdir ? copy.viewPluginFiles : copy.viewRepository}
        </ExternalLink>
      )}
    </Row>
  )
}

function VersionRow({
  copy,
  request,
  pin,
  agent,
  busy
}: Pick<PluginInstallDetailsProps, 'busy' | 'copy' | 'pin' | 'request'> & { agent: boolean }) {
  if (request.catalogName) {
    return <Row label={copy.versionLabel}>{copy.reviewedPin(request.sha?.slice(0, 8) ?? '')}</Row>
  }

  if (!agent) {
    return <Row label={copy.versionLabel}>{copy.latestCommit}</Row>
  }

  return (
    <Row label={copy.pinToCommit}>
      <Input
        aria-invalid={pin.invalid || undefined}
        aria-label={copy.pinToCommit}
        disabled={busy}
        onChange={event => pin.onChange(event.target.value)}
        placeholder={copy.pinToCommitPlaceholder}
        spellCheck={false}
        value={pin.value}
      />
      <span className={cn('mt-1 block', pin.invalid ? 'text-destructive' : 'text-(--ui-text-tertiary)')}>
        {pin.invalid ? copy.pinToCommitInvalid : copy.pinToCommitHint}
      </span>
    </Row>
  )
}

function ContentsRow({
  copy,
  request,
  probe,
  remote,
  profile
}: Pick<PluginInstallDetailsProps, 'copy' | 'probe' | 'profile' | 'remote' | 'request'>) {
  if (!probe?.ok) {
    return <Row label={copy.includesHeading}>{probe ? '' : copy.probing}</Row>
  }

  const components = installComponents(request, probe)
  const fromPackage = desktopHalfFromPackage(components, remote)

  const agentTarget = remote
    ? copy.agentTargetRemote(profile.label)
    : copy.agentTargetLocal(
        profile.label,
        profile.value === 'default' ? '~/.hermes/plugins/' : `~/.hermes/profiles/${profile.value}/plugins/`
      )

  const lines = [
    components.agent && {
      title: [copy.agentLabel, probe.agentName].filter(Boolean).join(' · '),
      hint: agentTarget
    },
    components.desktop && {
      title: [copy.desktopLabel, fromPackage ? null : probe.desktopName].filter(Boolean).join(' · '),
      hint: fromPackage ? copy.desktopTargetFromPackage : copy.desktopTarget
    },
    skipsDesktopHalf(request, probe) && {
      title: [copy.desktopLabel, probe.desktopName].filter(Boolean).join(' · '),
      hint: copy.desktopUnavailable
    }
  ].filter(line => line !== false)

  return (
    <Row label={copy.includesHeading}>
      <ul className="grid gap-1.5">
        {lines.map(line => (
          <li key={line.title}>
            <span className="block font-medium">{line.title}</span>
            <span className="block text-(--ui-text-tertiary)">{line.hint}</span>
          </li>
        ))}
      </ul>
      {probe.desktop && !probe.agent && <p className="mt-1 text-(--ui-text-tertiary)">{copy.desktopOnlyNote}</p>}
    </Row>
  )
}

/** The install's particulars behind Advanced: where it comes from, which version, what it puts down, and
 *  the profile it lands in. A git URL that is not in the reviewed catalog also offers Force and a pin. */
export function PluginInstallDetails(props: PluginInstallDetailsProps) {
  const { busy, copy, force, onForceChange, probe, profile, request } = props
  const agent = !probe?.ok || installComponents(request, probe).agent

  return (
    <dl className="grid gap-3 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2.5">
      <SourceRow copy={copy} repo={request.repo} />
      <VersionRow agent={agent} busy={busy} copy={copy} pin={props.pin} request={request} />
      <ContentsRow copy={copy} probe={probe} profile={profile} remote={props.remote} request={request} />
      <Row label={copy.profileLabel}>
        <Select disabled={busy || !agent} onValueChange={profile.onChange} value={profile.value}>
          <SelectTrigger aria-label={copy.profileLabel} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {profile.options.map(option => (
              <SelectItem key={option.name} value={normalizeProfileKey(option.name)}>
                {profileLabel(option)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Row>
      {(!request.catalogName || props.offerForce) && (
        <div>
          <label className={cn(CAPTION, 'flex items-center justify-between gap-3 text-foreground')}>
            <span>{copy.forceReinstall}</span>
            <Switch checked={force} disabled={busy} onCheckedChange={onForceChange} />
          </label>
        </div>
      )}
    </dl>
  )
}
