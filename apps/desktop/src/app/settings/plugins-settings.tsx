import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { $pluginRecords, type PluginRecord, setPluginEnabled } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { getStatus } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Package } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

import { EmptyState, ListRow, Pill, SectionHeading, SettingsContent } from './primitives'

const KIND_ORDER: Record<PluginRecord['kind'], number> = { disk: 0, runtime: 1, bundled: 2 }

function reveal(file: string) {
  void window.hermesDesktop?.revealPath?.(file)?.catch(() => undefined)
}

async function revealPluginsDir() {
  try {
    const { hermes_home } = await getStatus()
    // openDir (not reveal): the door often doesn't exist on first use, and
    // showItemInFolder on a missing path silently no-ops (esp. Windows).
    const result = await window.hermesDesktop?.openDir?.(`${hermes_home}/desktop-plugins`)

    if (result && !result.ok) {
      notifyError(result.error ?? 'unknown error', 'Could not open the plugins folder')
    }
  } catch (err) {
    notifyError(err, 'Could not resolve the plugins folder')
  }
}

function PluginRow({ record }: { record: PluginRecord }) {
  const { t } = useI18n()
  const p = t.settings.plugins

  return (
    <ListRow
      action={
        <div className="flex items-center justify-end gap-2">
          {record.file && (
            <Tip label={p.reveal}>
              <Button onClick={() => reveal(record.file!)} size="icon" variant="ghost">
                <Codicon name="folder-opened" size="0.85rem" />
              </Button>
            </Tip>
          )}
          <Switch
            aria-label={`${record.status === 'disabled' ? p.enable : p.disable} ${record.name}`}
            checked={record.status !== 'disabled'}
            onCheckedChange={on => {
              triggerHaptic('selection')
              void setPluginEnabled(record.id, on)
            }}
          />
        </div>
      }
      description={
        record.status === 'error' ? (
          <span className="text-(--ui-danger,#f87171)">{record.error}</span>
        ) : (
          record.file ?? record.id
        )
      }
      title={
        <span className="flex items-center gap-2">
          {record.name}
          <Pill>{p.kinds[record.kind]}</Pill>
          {record.status === 'error' && <Pill tone="primary">{p.failed}</Pill>}
        </span>
      }
    />
  )
}

export function PluginsSettings() {
  const { t } = useI18n()
  const p = t.settings.plugins
  const records = useStore($pluginRecords)

  const rows = Object.values(records).sort(
    (a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || a.name.localeCompare(b.name)
  )

  return (
    <SettingsContent>
      <SectionHeading icon={Package} meta={p.count(rows.length)} title={p.title} />
      <p className="mb-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{p.blurb}</p>

      <div className="mb-4 flex items-center gap-2">
        <Button onClick={() => void revealPluginsDir()} size="sm" variant="outline">
          <Codicon name="folder-opened" size="0.8rem" />
          {p.openFolder}
        </Button>
        <Button
          onClick={() => {
            triggerHaptic('selection')
            void discoverRuntimePlugins()
          }}
          size="sm"
          variant="outline"
        >
          <Codicon name="refresh" size="0.8rem" />
          {p.rescan}
        </Button>
      </div>

      {rows.length === 0 ? (
        <EmptyState title={p.empty} />
      ) : (
        <div className="divide-y divide-(--ui-stroke-tertiary)">
          {rows.map(record => (
            <PluginRow key={record.id} record={record} />
          ))}
        </div>
      )}
    </SettingsContent>
  )
}
