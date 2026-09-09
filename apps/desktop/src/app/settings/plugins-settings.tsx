import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'
import { Link } from 'react-router'

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
import { $agentPlugins, $agentPluginsStatus, loadAgentPlugins } from '@/store/agent-plugins'
import { notifyError } from '@/store/notifications'
import { openPluginInstallRequest } from '@/store/plugin-install-request'
import { $gatewayState } from '@/store/session'

import { EmptyState, Pill, SettingsContent, SettingsSection } from './primitives'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

const KIND_ORDER: Record<PluginRecord['kind'], number> = { disk: 0, runtime: 1, bundled: 2 }

/** Deep-link anchor for a plugin row (`?tab=plugins&plugin=<id>`). */
export const pluginElementId = (target: string) => `plugin-${target}`

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

// Compact row: name + pills and a wrapping description on the left, controls
// pinned top-right. Same type scale as ListRow, without its wide control grid.
function PluginLine({
  title,
  description,
  controls,
  id
}: {
  title: ReactNode
  description?: ReactNode
  controls: ReactNode
  id?: string
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg py-2" id={id}>
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

/** Folder name when a desktop plugin entry lives in the UNIFIED agent-plugins
 *  root (`~/.hermes/plugins/<name>/desktop/plugin.js`) — i.e. it is the
 *  desktop half of a bundled agent+desktop package. Null for standalone
 *  desktop plugins. */
function unifiedPackageName(file?: string): null | string {
  if (!file) {
    return null
  }

  const match = /[\\/]plugins[\\/]([^\\/]+)[\\/]desktop[\\/]plugin\.js$/.exec(file)

  return match ? match[1] : null
}

/** Open the dual-target install modal pre-filled to install ONLY the agent
 *  half of a bundled package (drift repair). Provenance comes from the
 *  package's catalog sidecar when present; otherwise the git remote of the
 *  plugin folder is unknown and we fall back to asking the user via the
 *  standard flow with the folder name as identifier hint. */
async function repairAgentHalf(record: PluginRecord, packageName: string) {
  let repo = ''
  let catalogName: string | undefined
  let sha: string | undefined

  try {
    const pluginDir = record.file?.replace(/[\\/]desktop[\\/]plugin\.js$/, '')

    const raw = pluginDir ? await window.hermesDesktop?.readFileText?.(`${pluginDir}/.hermes-catalog.json`) : null

    if (raw) {
      const sidecar = JSON.parse(typeof raw === 'string' ? raw : ((raw as { content?: string }).content ?? '')) as {
        catalog_name?: string
        repo?: string
        sha?: string
      }

      repo = sidecar.repo ?? ''
      catalogName = sidecar.catalog_name
      sha = sidecar.sha
    }
  } catch {
    // No sidecar (raw-git bundled install) — fall through to the name hint.
  }

  openPluginInstallRequest({
    catalogName,
    legacyHint: 'agent',
    repo: repo || packageName,
    sha
  })
}

function PluginRow({ record, agentHalfMissing }: { record: PluginRecord; agentHalfMissing?: boolean }) {
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
      id={pluginElementId(record.id)}
      title={
        <>
          <span>{record.name}</span>
          <Pill>{p.kinds[record.kind]}</Pill>
          {record.status === 'error' && <Pill tone="primary">{p.failed}</Pill>}
          {agentHalfMissing && (
            <Tip label={p.agentHalfMissingTip}>
              <Button
                className="h-5 px-1.5 text-[0.65rem]"
                onClick={() => void repairAgentHalf(record, unifiedPackageName(record.file) ?? record.name)}
                size="xs"
                variant="outline"
              >
                {p.agentHalfMissing}
              </Button>
            </Tip>
          )}
        </>
      }
    />
  )
}

export function PluginsSettings() {
  const { t } = useI18n()
  const p = t.settings.plugins
  const records = useStore($pluginRecords)
  const { requestGateway } = useGatewayRequest()
  const gatewayState = useStore($gatewayState)
  // The agent-plugin list for the CURRENTLY connected backend's active
  // profile — used only to flag bundled packages whose desktop half is local
  // but whose agent half is not installed where the app is now pointing (one
  // desktop app, N agents: switching gateway/profile makes this drift visible
  // instead of silent). Management of agent plugins lives in Capabilities →
  // Plugins; this page keeps just the badge.
  const agentRows = useStore($agentPlugins)
  const agentStatus = useStore($agentPluginsStatus)
  const agentNames = new Set(agentRows.flatMap(row => [row.name, row.key ?? row.name]))

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    void loadAgentPlugins(requestGateway)
  }, [gatewayState, requestGateway])

  // Deep-link from settings search (?plugin=<id or key>): rows render as soon
  // as their store hydrates, so "ready" is simply target-present; the polling
  // in the hook rides out the async list loads (agent rows arrive via RPC).
  useDeepLinkHighlight({
    param: 'plugin',
    ready: () => true,
    elementId: pluginElementId
  })

  const rows = Object.values(records).sort(
    (a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || a.name.localeCompare(b.name)
  )

  return (
    <SettingsContent>
      <div className="mb-4">
        <Button onClick={() => openPluginInstallRequest({ repo: '' })} size="sm" type="button" variant="secondary">
          {p.installModal.installFromGit}
        </Button>
      </div>
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
            {rows.map(record => {
              const packageName = unifiedPackageName(record.file)

              return (
                <PluginRow
                  agentHalfMissing={packageName !== null && agentStatus === 'ready' && !agentNames.has(packageName)}
                  key={record.id}
                  record={record}
                />
              )
            })}
          </div>
        )}
      </SettingsSection>

      <SettingsSection icon={Package} title={p.agent.title}>
        <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {p.agent.movedToCapabilities}{' '}
          <Link className="text-(--ui-text-link,var(--ui-accent))" to="/skills?tab=plugins">
            {p.agent.openCapabilities}
          </Link>
        </p>
      </SettingsSection>
    </SettingsContent>
  )
}
