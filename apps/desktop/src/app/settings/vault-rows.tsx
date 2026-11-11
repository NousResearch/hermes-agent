import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { KeyRound, Lock, Plus, ShieldLock, Trash2 } from '@/lib/icons'
import { usePortalShell } from '@/store/shell-variant'

import { ListRow, Pill, SectionHeading, SettingsContent } from './primitives'
import { SettingsProfileScope } from './profile-scope'
import { SettingsAddRow, SettingsCard, SettingsGroup, SettingsOverview } from './settings-card'
import { useSettingsPagesCopy } from './settings-pages-copy'
import type { VaultItem, VaultKind, VaultSource } from './vault-settings'

interface VaultCredentialRowProps {
  item: VaultItem
  /** The detected password managers, to name the one a non-local item lives in. */
  sources: VaultSource[]
  onDelete: () => void
}

function formatCreated(iso: string): string {
  const parsed = new Date(iso)

  return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleDateString()
}

/** One saved login, card or address: label + kind, identifier · origin · date, delete or its source. */
export function VaultCredentialRow({ item, sources, onDelete }: VaultCredentialRowProps) {
  const { t } = useI18n()
  const v = t.settings.vault
  const kindLabel = v.kinds[item.kind as VaultKind] ?? item.kind
  const sourceLabel = sources.find(source => source.name === item.backend)?.display_name ?? item.backend
  const createdOn = v.createdOn(formatCreated(item.created_at))

  return (
    <ListRow
      action={
        item.backend && item.backend !== 'local' ? (
          <Pill tone="muted">{sourceLabel}</Pill>
        ) : (
          <Button
            aria-label={v.deleteAction}
            className="text-(--ui-text-tertiary) hover:text-destructive"
            onClick={onDelete}
            size="icon-sm"
            type="button"
            variant="ghost"
          >
            <Trash2 className="size-3.5" />
          </Button>
        )
      }
      description={
        // identifier · origin · date, separated so the row scans as three facts; the origin is
        // omitted when the label already IS the host (save-on-page items are labelled by host).
        <span className="flex flex-wrap items-center gap-x-2">
          {item.identifier && <span className="truncate">{v.identifierShown(item.identifier)}</span>}
          {item.origin && item.origin.replace(/^https?:\/\//, '') !== item.label && (
            <>
              {item.identifier && (
                <span aria-hidden className="text-(--ui-text-tertiary)">
                  ·
                </span>
              )}
              <span className="truncate">{item.origin}</span>
            </>
          )}
          <span aria-hidden className="text-(--ui-text-tertiary)">
            ·
          </span>
          <span>{createdOn}</span>
        </span>
      }
      key={item.id}
      title={
        <span className="flex items-center gap-2">
          <span className="truncate">{item.label}</span>
          <Pill tone={item.kind === 'login' ? 'primary' : 'muted'}>{kindLabel}</Pill>
          {item.has_otp && <Pill tone="muted">{v.twoFactorBadge}</Pill>}
        </span>
      }
    />
  )
}

interface VaultSourceRowProps {
  source: VaultSource
  lockPending: boolean
  togglePending: boolean
  onLock: () => void
  onUnlock: () => void
  onToggle: (enabled: boolean) => void
}

/** One password manager: detection/lock state, lock or unlock, and the enable switch. */
export function VaultSourceRow({
  source,
  lockPending,
  togglePending,
  onLock,
  onUnlock,
  onToggle
}: VaultSourceRowProps) {
  const { t } = useI18n()
  const v = t.settings.vault

  return (
    <ListRow
      action={
        <span className="flex items-center justify-end gap-2">
          {source.enabled &&
            (source.unlocked ? (
              <Button
                className="gap-1.5"
                disabled={lockPending}
                onClick={onLock}
                size="sm"
                type="button"
                variant="ghost"
              >
                <Lock className="size-3.5" />
                {v.sources.lock}
              </Button>
            ) : (
              <Button className="gap-1.5" onClick={onUnlock} size="sm" type="button" variant="outline">
                <KeyRound className="size-3.5" />
                {v.sources.unlock}
              </Button>
            ))}
          {source.installed && (
            <Switch
              aria-label={source.display_name}
              checked={source.enabled}
              disabled={togglePending}
              onCheckedChange={enabled => {
                triggerHaptic('selection')
                onToggle(enabled)
              }}
            />
          )}
        </span>
      }
      description={
        !source.installed
          ? v.sources.notInstalled(source.display_name)
          : source.enabled
            ? source.unlocked
              ? v.sources.unlockedDesc
              : v.sources.lockedDesc
            : v.sources.disabledDesc
      }
      key={source.name}
      title={
        <span className="flex items-center gap-2">
          <span>{source.display_name}</span>
          <Pill tone={source.enabled && source.unlocked ? 'primary' : 'muted'}>
            {!source.installed
              ? v.sources.statusNotDetected
              : !source.enabled
                ? v.sources.statusOff
                : source.unlocked
                  ? v.sources.statusUnlocked
                  : v.sources.statusLocked}
          </Pill>
        </span>
      }
    />
  )
}

interface VaultLayoutProps {
  credentialRows: ReactNode
  sourceRows: ReactNode
  dialogs: ReactNode
  /** Null while the first read is in flight. */
  itemCount: null | number
  sourceCount: number
  /** `credentials` or `sources` shows just that section (a deep link); none shows both. */
  subpage?: string
  onAdd: () => void
}

/** The vault page: classic's headed lists, or the redesign's titled cards (895:20189) — saved
 *  credentials ending in "+ Add", then password managers. */
export function VaultLayout(props: VaultLayoutProps) {
  return usePortalShell() ? <RedesignVault {...props} /> : <ClassicVault {...props} />
}

function ClassicVault({ credentialRows, sourceRows, dialogs, itemCount, subpage, onAdd }: VaultLayoutProps) {
  const { t } = useI18n()
  const v = t.settings.vault

  return (
    <SettingsContent>
      {/* The vault is per profile (its RPCs ride the (connection, profile) owner above), so the
          page must SAY which profile it is showing — without the chip a non-default selection
          looked like the default vault with entries missing (#121966). */}
      <SettingsProfileScope className="mb-5" />
      {(subpage === undefined || subpage === 'credentials') && (
        <>
          <SectionHeading
            aside={
              <Button className="gap-1.5" onClick={onAdd} size="sm" type="button" variant="outline">
                <Plus className="size-3.5" />
                {v.add}
              </Button>
            }
            icon={ShieldLock}
            meta={itemCount ? v.count(itemCount) : undefined}
            page
            title={v.title}
          />
          <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {v.blurb}
          </p>

          {itemCount === 0 && <EmptyState description={v.emptyDesc} title={v.empty} />}

          {credentialRows}
        </>
      )}

      {(subpage === undefined || subpage === 'sources') && (
        <>
          <div className={subpage === undefined ? 'mt-6' : undefined}>
            <SectionHeading icon={KeyRound} page title={v.sources.title} />
          </div>
          <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {v.sources.blurb}
          </p>
          {sourceRows}
        </>
      )}

      {dialogs}
    </SettingsContent>
  )
}

function RedesignVault({
  credentialRows,
  sourceRows,
  dialogs,
  itemCount,
  sourceCount,
  subpage,
  onAdd
}: VaultLayoutProps) {
  const { t } = useI18n()
  const v = t.settings.vault
  const pages = useSettingsPagesCopy()
  const showCredentials = subpage === undefined || subpage === 'credentials'
  const showSources = (subpage === undefined || subpage === 'sources') && sourceCount > 0

  return (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      <SettingsOverview>
        {showCredentials && (
          <SettingsGroup
            description={pages.vault.credentialsDesc}
            meta={itemCount ? v.count(itemCount) : undefined}
            title={pages.vault.credentials}
          >
            <SettingsCard>
              {itemCount === 0 && <p className="global-settings-empty-row">{v.emptyDesc}</p>}
              {credentialRows}
              <SettingsAddRow onClick={onAdd}>{v.add}</SettingsAddRow>
            </SettingsCard>
          </SettingsGroup>
        )}
        {showSources && (
          <SettingsGroup description={pages.vault.managersDesc} title={pages.vault.managers}>
            <SettingsCard>{sourceRows}</SettingsCard>
          </SettingsGroup>
        )}
      </SettingsOverview>
      {dialogs}
    </SettingsContent>
  )
}
