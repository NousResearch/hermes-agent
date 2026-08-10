import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { $pluginRecords, type PluginRecord, setPluginEnabled } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { FolderOpen, Monitor, Package, RefreshCw } from '@/lib/icons'
import { normalize } from '@/lib/text'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow,
  type GatewayRequest,
  loadAgentPlugins,
  toggleAgentPlugin
} from '@/store/agent-plugins'
import { notifyError } from '@/store/notifications'
import { $connection, $gatewayState } from '@/store/session'

import { EmptyState, ListRowSkeleton, Pill, SettingsContent, SettingsSection } from './primitives'

const KIND_ORDER: Record<PluginRecord['kind'], number> = { disk: 0, runtime: 1, bundled: 2 }

// User-installed plugins first, bundled last — mirrors `hermes plugins list`.
const SOURCE_ORDER: Record<string, number> = { user: 0, git: 0, project: 1, entrypoint: 2, bundled: 3 }

// Plugin categories (by registry key prefix) that other surfaces own — same
// curation stance as desktop-slash-commands.ts. dashboard_auth/* only matters
// to `hermes dashboard`; model-providers/* are configured in Settings →
// Models; platforms/* are managed from Messaging. The plugin switch is not
// the user-facing control for any of them, so listing them here is noise.
const HIDDEN_KEY_PREFIXES = ['dashboard_auth/', 'model-providers/', 'platforms/']

const isDesktopRelevant = (row: AgentPluginRow) => {
  const key = row.key

  return !key || !HIDDEN_KEY_PREFIXES.some(prefix => key.startsWith(prefix))
}

const agentPluginRowKey = (row: AgentPluginRow) =>
  row.key ?? [row.name, row.source, row.version, row.description].join('\0')

function reveal(file: string) {
  void window.hermesDesktop?.revealPath?.(file)?.catch(() => undefined)
}

async function revealPluginsDir() {
  try {
    // Electron owns the local plugin root — deriving it from the backend's
    // hermes_home breaks against a remote backend (#66899).
    const dir = await window.hermesDesktop?.desktopPluginsRoot?.()

    if (!dir) {
      notifyError('Desktop plugins are unavailable', 'Could not resolve the plugins folder')

      return
    }

    // openDir (not reveal): the door often doesn't exist on first use, and
    // showItemInFolder on a missing path silently no-ops (esp. Windows).
    const result = await window.hermesDesktop?.openDir?.(dir)

    if (result && !result.ok) {
      notifyError(result.error ?? 'unknown error', 'Could not open the plugins folder')
    }
  } catch (err) {
    notifyError(err, 'Could not resolve the plugins folder')
  }
}

// Agent plugins live under the BACKEND's hermes home (profile-aware), so the
// path comes from the gateway — not from the renderer's local HERMES_HOME.
// Callers gate on a local connection: openDir mkdir-creates the path, which
// must never happen for a directory that belongs to a remote box.
async function revealAgentPluginsDir(request: GatewayRequest) {
  try {
    const result = await request<{ home?: string }>('config.get', { key: 'profile' })
    const home = (result?.home ?? '').trim()

    if (!home) {
      notifyError('The backend did not report its home directory', 'Could not open the plugins folder')

      return
    }

    const opened = await window.hermesDesktop?.openDir?.(`${home}/plugins`)

    if (opened && !opened.ok) {
      notifyError(opened.error ?? 'unknown error', 'Could not open the plugins folder')
    }
  } catch (err) {
    notifyError(err, 'Could not open the plugins folder')
  }
}

// Compact row: name + pills and a wrapping description on the left, controls
// pinned top-right. Same type scale as ListRow, without its wide control grid.
function PluginLine({
  title,
  description,
  controls
}: {
  title: ReactNode
  description?: ReactNode
  controls: ReactNode
}) {
  return (
    <div className="flex items-start gap-3 py-2">
      <div className="min-w-0 flex-1 pr-4">
        <div className="flex flex-wrap items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
          {title}
        </div>
        {description && (
          <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) break-words text-(--ui-text-tertiary)">
            {description}
          </div>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2">{controls}</div>
    </div>
  )
}

function AgentPluginRowView({ row }: { row: AgentPluginRow }) {
  const { t } = useI18n()
  const p = t.settings.plugins
  const { requestGateway } = useGatewayRequest()
  const busy = useStore($agentPluginBusy)
  const key = row.key

  // Pre-contract-v6 backends return rows without a canonical key. Name-addressed
  // toggles silently flip every same-named plugin across category dirs
  // (image_gen/fal vs video_gen/fal), so keyless rows are read-only — the
  // backend-contract skew toast tells the user to update.
  const toggle = (
    <Switch
      aria-label={`${row.status === 'enabled' ? p.disable : p.enable} ${row.name}`}
      checked={row.status === 'enabled'}
      disabled={!key || busy === key}
      onCheckedChange={on => {
        if (!key) {
          return
        }

        triggerHaptic('selection')
        void toggleAgentPlugin(requestGateway, key, on, p.agent.toggleFailed(row.name))
      }}
    />
  )

  return (
    <PluginLine
      controls={key ? toggle : <Tip label={p.agent.updateBackendToManage}>{toggle}</Tip>}
      description={row.description || (row.version ? `v${row.version}` : undefined)}
      title={
        <>
          <span>{row.name}</span>
          <Pill>{p.agent.sources[row.source] ?? row.source}</Pill>
          {row.portable && <Pill tone="primary">{p.agent.portable}</Pill>}
        </>
      }
    />
  )
}

function AgentPluginsSection() {
  const { t } = useI18n()
  const p = t.settings.plugins
  const { requestGateway } = useGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const connection = useStore($connection)
  const rows = useStore($agentPlugins)
  const status = useStore($agentPluginsStatus)
  const error = useStore($agentPluginsError)
  const [query, setQuery] = useState('')

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    void loadAgentPlugins(requestGateway)
  }, [gatewayState, requestGateway])

  const needle = normalize(query)

  const sorted = rows
    .filter(isDesktopRelevant)
    .filter(
      row =>
        !needle ||
        row.name.toLowerCase().includes(needle) ||
        (row.key ?? '').toLowerCase().includes(needle) ||
        row.description.toLowerCase().includes(needle)
    )
    .sort((a, b) => (SOURCE_ORDER[a.source] ?? 9) - (SOURCE_ORDER[b.source] ?? 9) || a.name.localeCompare(b.name))

  return (
    <SettingsSection
      icon={Package}
      meta={status === 'ready' ? p.count(sorted.length) : undefined}
      title={p.agent.title}
    >
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {p.agent.blurb}
      </p>

      {connection?.mode !== 'remote' && (
        <div className="mb-2 flex items-center gap-3">
          <Button
            onClick={() => void revealAgentPluginsDir(requestGateway)}
            size="sm"
            type="button"
            variant="textStrong"
          >
            <FolderOpen className="size-3.5" />
            <span>{p.openFolder}</span>
          </Button>
        </div>
      )}

      <input
        className="mb-2 w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 text-[length:var(--conversation-caption-font-size)] outline-none placeholder:text-(--ui-text-tertiary) focus:border-(--ui-stroke-secondary)"
        onChange={event => setQuery(event.target.value)}
        placeholder={p.agent.search}
        spellCheck={false}
        value={query}
      />

      {status === 'loading' || status === 'idle' ? (
        <div>
          <ListRowSkeleton />
          <ListRowSkeleton />
          <ListRowSkeleton />
        </div>
      ) : status === 'error' ? (
        <EmptyState description={error ?? undefined} title={p.agent.loadFailed} />
      ) : sorted.length === 0 ? (
        needle ? (
          <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {p.agent.noMatches}
          </p>
        ) : (
          <EmptyState title={p.agent.empty} />
        )
      ) : (
        <div>
          {sorted.map(row => (
            <AgentPluginRowView key={agentPluginRowKey(row)} row={row} />
          ))}
        </div>
      )}
    </SettingsSection>
  )
}

function PluginRow({ record }: { record: PluginRecord }) {
  const { t } = useI18n()
  const p = t.settings.plugins

  return (
    <PluginLine
      controls={
        <>
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
        </>
      }
      description={
        record.status === 'error' ? (
          <span className="text-(--ui-danger,#f87171)">{record.error}</span>
        ) : (
          (record.description ?? record.file ?? record.id)
        )
      }
      title={
        <>
          <span>{record.name}</span>
          <Pill>{p.kinds[record.kind]}</Pill>
          {record.status === 'error' && <Pill tone="primary">{p.failed}</Pill>}
        </>
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
      <SettingsSection icon={Monitor} meta={p.count(rows.length)} title={p.title}>
        <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{p.blurb}</p>

        <div className="mb-2 flex items-center gap-3">
          <Button onClick={() => void revealPluginsDir()} size="sm" type="button" variant="textStrong">
            <FolderOpen className="size-3.5" />
            <span>{p.openFolder}</span>
          </Button>
          <Button
            onClick={() => {
              triggerHaptic('selection')
              void discoverRuntimePlugins()
            }}
            size="sm"
            type="button"
            variant="textStrong"
          >
            <RefreshCw className="size-3.5" />
            <span>{p.rescan}</span>
          </Button>
        </div>

        {rows.length === 0 ? (
          <EmptyState title={p.empty} />
        ) : (
          <div>
            {rows.map(record => (
              <PluginRow key={record.id} record={record} />
            ))}
          </div>
        )}
      </SettingsSection>

      <AgentPluginsSection />
    </SettingsContent>
  )
}
