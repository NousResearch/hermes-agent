import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { type SetupField, SetupFieldList } from '@/components/ui/setup-field-list'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { cn } from '@/lib/utils'
import { COMMIT_SHA_RE } from '@/store/agent-plugins'
import type { CatalogEntry } from '@/store/connection-request'
import { $profiles, normalizeProfileKey, profileLabel } from '@/store/profile'

interface CatalogAdvancedDialogProps {
  entry: CatalogEntry
  fields: SetupField[]
  /** A skill has no agent/desktop halves, no enable step and no commit pin; it keeps profile and force. */
  kind: 'plugin' | 'skill'
  onCancel: () => void
  /** The `connection.respond` env map (CATALOG-ROW-CONTRACT.md): every value a string. */
  onInstall: (env: Record<string, string>) => void
  open: boolean
}

const CAPTION = 'text-[length:var(--conversation-caption-font-size)]'
const flag = (value: boolean) => (value ? '1' : '0')

/** Files view of a GitHub catalog entry at its pin; other hosts get the repository link only. */
function pluginFilesUrl({ repo, sha, subdir }: CatalogEntry): null | string {
  const base = repo?.replace(/\.git$/, '').replace(/\/+$/, '')

  return base?.startsWith('https://github.com/') && subdir ? `${base}/tree/${sha || 'HEAD'}/${subdir}` : null
}

/** The row's Install with every Plugins-tab control exposed. Its own Install sends the values; the row's
 *  Install never reads them, so closing the modal leaves nothing behind. */
export function CatalogAdvancedDialog(props: CatalogAdvancedDialogProps) {
  // Mounted only while open, so every opening starts from the entry's defaults.
  return (
    <Dialog onOpenChange={next => !next && props.onCancel()} open={props.open}>
      {props.open ? <CatalogAdvancedForm {...props} /> : null}
    </Dialog>
  )
}

function CatalogAdvancedForm({ entry, fields, kind, onCancel, onInstall }: CatalogAdvancedDialogProps) {
  const { t } = useI18n()
  const m = t.settings.plugins.installModal
  const copy = t.assistant.catalogInstall
  const profiles = useStore($profiles)
  const [agentHalf, setAgentHalf] = useState(true)
  const [desktopHalf, setDesktopHalf] = useState(entry.hasDesktopHalf)
  const [targetProfile, setTargetProfile] = useState(normalizeProfileKey(entry.targetProfile))
  const [enable, setEnable] = useState(true)
  const [force, setForce] = useState(false)
  const [pin, setPin] = useState(entry.sha ?? '')

  const [credentials, setCredentials] = useState<Record<string, string>>(() =>
    Object.fromEntries(fields.map(field => [field.name, field.secret ? '' : field.default]))
  )

  const pinTrimmed = pin.trim().toLowerCase()
  const pinInvalid = pinTrimmed !== '' && !COMMIT_SHA_RE.test(pinTrimmed)
  const missingCredential = fields.some(field => field.required && !credentials[field.name]?.trim())
  const nothingSelected = !agentHalf && !desktopHalf
  const known = profiles.some(profile => normalizeProfileKey(profile.name) === targetProfile)
  const profileOptions = known ? profiles : [...profiles, { name: targetProfile }]
  const filesUrl = pluginFilesUrl(entry)

  const plugin = kind === 'plugin'

  const install = () =>
    onInstall(
      plugin
        ? {
            ...credentials,
            agent_half: flag(agentHalf),
            ...(entry.hasDesktopHalf ? { desktop_half: flag(desktopHalf) } : {}),
            enable: flag(enable),
            force: flag(force),
            ...(pinTrimmed ? { ref: pinTrimmed } : {}),
            target_profile: targetProfile
          }
        : { ...credentials, force: flag(force), target_profile: targetProfile }
    )

  const profileSelect = (
    <Select disabled={!agentHalf} onValueChange={setTargetProfile} value={targetProfile}>
      <SelectTrigger aria-label={m.profileLabel} className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {profileOptions.map(profile => (
          <SelectItem key={profile.name} value={normalizeProfileKey(profile.name)}>
            {profileLabel(profile)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  return (
    <DialogContent className="w-[min(32rem,calc(100vw-2rem))]" data-slot="catalog-advanced">
      <DialogHeader>
        <DialogTitle className="truncate">{entry.display}</DialogTitle>
        {entry.description ? <DialogDescription>{entry.description}</DialogDescription> : null}
      </DialogHeader>

      <div className="grid max-h-[60vh] gap-4 overflow-y-auto">
        {plugin ? (
          <Section title={m.includesHeading}>
            <div className="grid gap-2 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2">
              <label className="flex items-start gap-3">
                <Checkbox checked={agentHalf} onCheckedChange={value => setAgentHalf(value === true)} />
                <span className="font-medium text-foreground">{m.agentLabel}</span>
              </label>
              <label className="grid gap-1 pl-7">
                <span className={cn(CAPTION, 'text-foreground')}>{m.profileLabel}</span>
                {profileSelect}
              </label>
            </div>
            {entry.hasDesktopHalf ? (
              <label className="flex items-start gap-3 rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2">
                <Checkbox checked={desktopHalf} onCheckedChange={value => setDesktopHalf(value === true)} />
                <span className="font-medium text-foreground">{m.desktopLabel}</span>
              </label>
            ) : null}
            {nothingSelected ? <p className={cn(CAPTION, 'text-destructive')}>{m.selectComponent}</p> : null}
          </Section>
        ) : (
          <label className="grid gap-1">
            <span className={cn(CAPTION, 'text-foreground')}>{m.profileLabel}</span>
            {profileSelect}
          </label>
        )}

        <div className="grid gap-3">
          {plugin ? (
            <label className="flex items-center justify-between gap-3">
              <span className={cn(CAPTION, 'text-foreground')}>{m.enableAgent}</span>
              <Switch checked={enable} disabled={!agentHalf} onCheckedChange={setEnable} />
            </label>
          ) : null}
          <label className="flex items-center justify-between gap-3">
            <span className={cn(CAPTION, 'text-foreground')}>{m.forceReinstall}</span>
            <Switch checked={force} onCheckedChange={setForce} />
          </label>
          {plugin ? (
            <label className="grid gap-1">
              <span className={cn(CAPTION, 'text-foreground')}>{m.pinToCommit}</span>
              <Input
                aria-invalid={pinInvalid || undefined}
                aria-label={m.pinToCommit}
                className="font-mono"
                disabled={!agentHalf}
                onChange={event => setPin(event.target.value)}
                placeholder={m.pinToCommitPlaceholder}
                spellCheck={false}
                value={pin}
              />
              <span className={cn(CAPTION, pinInvalid ? 'text-destructive' : 'text-(--ui-text-tertiary)')}>
                {pinInvalid ? m.pinToCommitInvalid : m.pinToCommitHint}
              </span>
            </label>
          ) : null}
        </div>

        {entry.repo ? (
          <Section title={m.sourceHeading}>
            <Facts
              rows={[
                [m.repoLabel, entry.repo],
                [copy.commitLabel, entry.sha?.slice(0, 12)],
                [copy.subdirLabel, entry.subdir]
              ]}
            />
            <div className={cn(CAPTION, 'flex flex-wrap gap-x-4 gap-y-1')}>
              <ExternalLink href={entry.repo} showExternalIcon>
                {m.viewRepository}
              </ExternalLink>
              {filesUrl ? (
                <ExternalLink href={filesUrl} showExternalIcon>
                  {m.viewPluginFiles}
                </ExternalLink>
              ) : null}
            </div>
          </Section>
        ) : null}

        {entry.scan || entry.requirements.length > 0 ? (
          <Section title={copy.securityHeading}>
            {entry.scan ? (
              <p className={cn(CAPTION, SCAN_TONE[entry.scan.status])}>
                {copy.scan[entry.scan.status]}
                {entry.scan.summary ? ` · ${entry.scan.summary}` : ''}
              </p>
            ) : null}
            {entry.requirements.length > 0 ? (
              <Facts mono={false} rows={[[copy.requirementsLabel, entry.requirements.join(', ')]]} />
            ) : null}
          </Section>
        ) : null}

        {fields.length > 0 ? (
          <Section title={copy.credentialsHeading}>
            <SetupFieldList
              draft={credentials}
              fields={fields}
              onChange={(name, value) => setCredentials(current => ({ ...current, [name]: value }))}
            />
          </Section>
        ) : null}
      </div>

      <DialogFooter>
        <Button onClick={onCancel} variant="outline">
          {t.common.cancel}
        </Button>
        <Button disabled={pinInvalid || nothingSelected || missingCredential} onClick={install}>
          {copy.install}
        </Button>
      </DialogFooter>
    </DialogContent>
  )
}

const SCAN_TONE = {
  failed: 'text-destructive',
  passed: 'text-emerald-600 dark:text-emerald-400',
  warnings: 'text-amber-600 dark:text-amber-400'
} as const

function Section({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="grid gap-2">
      <h3 className={cn(CAPTION, 'font-medium text-foreground')}>{title}</h3>
      {children}
    </section>
  )
}

/** Read-only provenance as a two-column table; empty values are left out. The label column is fixed so
 *  Source and Security line up. */
function Facts({ mono = true, rows }: { mono?: boolean; rows: [string, null | string | undefined][] }) {
  return (
    <dl className={cn(CAPTION, 'grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 gap-y-1')}>
      {rows
        .filter((row): row is [string, string] => Boolean(row[1]))
        .map(([label, value]) => (
          <div className="contents" key={label}>
            <dt className="text-(--ui-text-tertiary)">{label}</dt>
            <dd className={cn('text-foreground', mono ? 'break-all font-mono' : 'wrap-anywhere')}>{value}</dd>
          </div>
        ))}
    </dl>
  )
}
