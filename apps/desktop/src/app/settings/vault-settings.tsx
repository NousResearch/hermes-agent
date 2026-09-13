import { useStore } from '@nanostores/react'
import { useCallback, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { EmptyState } from '@/components/ui/empty-state'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Plus, ShieldLock, Trash2 } from '@/lib/icons'
import { $activeConnectionId } from '@/store/connections'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { ListRow, Pill, SectionHeading, SettingsContent } from './primitives'
import { useVaultAdd } from './vault-settings-add'
import { VaultAddDialog } from './vault-settings-add-dialog'
import { useVaultData, type VaultItem, type VaultSourceName } from './vault-settings-data'
import type { VaultKind } from './vault-settings-form'
import { VaultSources } from './vault-settings-sources'

// Vault data is private to one (connection, profile); the cache key carries that owner so a
// late response from profile A can never paint under profile B.
export const vaultOwnerKey = (connectionId: null | string, profile: string) => `${connectionId ?? ''}::${profile}`

export function VaultSettings() {
  const { t } = useI18n()
  const v = t.settings.vault
  // The owner this panel edits: every RPC below goes through the owner's socket with an explicit
  // profile — never the ambient foreground gateway. The mount site keys the panel by this same
  // owner, so a profile switch / connection swap remounts it: dialogs close and drafts (including a
  // typed master password) are gone by construction rather than by cleanup code.
  const scopeProfile = useStore($settingsScopeProfile)
  const connectionId = useStore($activeConnectionId)
  const owner = vaultOwnerKey(connectionId, scopeProfile)

  const vault = useVaultData(owner, scopeProfile)
  const { items, isPending, externalSources, requestGateway, invalidate } = vault
  const add = useVaultAdd(vault)
  const { openAdd } = add
  const [pendingDelete, setPendingDelete] = useState<null | VaultItem>(null)

  const deleteItem = useCallback(
    async (item: VaultItem) => {
      await requestGateway<{ removed: boolean }>('vault.remove', { id: item.id })
      triggerHaptic('success')
      void invalidate()
    },
    [invalidate, requestGateway]
  )

  const kindLabel = useCallback((kind: string) => v.kinds[kind as VaultKind] ?? kind, [v.kinds])

  const sourceLabel = useCallback(
    (name: VaultSourceName) => externalSources.find(s => s.name === name)?.display_name ?? name,
    [externalSources]
  )

  const formatCreated = useCallback((iso: string) => {
    const parsed = new Date(iso)

    return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleDateString()
  }, [])

  return (
    <SettingsContent>
      <SectionHeading
        aside={
          <Button className="gap-1.5" onClick={() => openAdd()} size="sm" type="button" variant="outline">
            <Plus className="size-3.5" />
            {v.add}
          </Button>
        }
        icon={ShieldLock}
        meta={items.length > 0 ? v.count(items.length) : undefined}
        title={v.title}
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {v.blurb}
      </p>

      {!isPending && items.length === 0 && <EmptyState description={v.emptyDesc} title={v.empty} />}

      {items.map(item => (
        <ListRow
          action={
            item.backend && item.backend !== 'local' ? (
              <Pill tone="muted">{sourceLabel(item.backend)}</Pill>
            ) : (
              <Button
                aria-label={v.deleteAction}
                className="text-(--ui-text-tertiary) hover:text-destructive"
                onClick={() => setPendingDelete(item)}
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
              <span>{v.createdOn(formatCreated(item.created_at))}</span>
            </span>
          }
          key={item.id}
          title={
            <span className="flex items-center gap-2">
              <span className="truncate">{item.label}</span>
              <Pill tone={item.kind === 'login' ? 'primary' : 'muted'}>{kindLabel(item.kind)}</Pill>
              {item.has_otp && <Pill tone="muted">{v.twoFactorBadge}</Pill>}
            </span>
          }
        />
      ))}

      <VaultSources vault={vault} />
      <VaultAddDialog add={add} />

      {/* Delete confirmation */}
      <ConfirmDialog
        confirmLabel={v.deleteConfirm}
        description={pendingDelete ? v.deleteDescription(pendingDelete.label) : undefined}
        destructive
        onClose={() => setPendingDelete(null)}
        onConfirm={async () => {
          if (pendingDelete) {
            await deleteItem(pendingDelete)
          }
        }}
        open={pendingDelete !== null}
        title={v.deleteTitle}
      />
    </SettingsContent>
  )
}
