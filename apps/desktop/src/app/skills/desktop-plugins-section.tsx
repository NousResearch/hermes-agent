import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { $pluginRecords, type PluginRecord, setPluginEnabled } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { FolderOpen, Monitor, RefreshCw } from '@/lib/icons'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { notifyError } from '@/store/notifications'
import { openPluginInstallRequest } from '@/store/plugin-install-request'

import { Pill } from '../settings/primitives'

const KIND_ORDER: Record<PluginRecord['kind'], number> = { disk: 0, runtime: 1, bundled: 2 }

/** Deep-link anchor for a plugin row (`/skills?tab=plugins&plugin=<id>`); shared
 *  by desktop rows (record id) and agent rows (canonical key). */
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
 *  plugin folder is unknown and we fall back to the folder name as the
 *  identifier hint. */
async function repairAgentHalf(record: PluginRecord, packageName: string, profile: null | string) {
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
    profile,
    repo: repo || packageName,
    sha
  })
}

/** One list row, same type scale and rhythm as the agent-plugin rows above it. */
export function PluginListRow({
  title,
  description,
  controls,
  icon,
  id
}: {
  title: ReactNode
  description?: ReactNode
  controls: ReactNode
  icon: ReactNode
  id?: string
}) {
  return (
    <div className="flex items-start gap-3 border-b border-(--ui-stroke-tertiary) px-3 py-2 last:border-b-0" id={id}>
      {icon}
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
          {title}
        </div>
        {description && (
          <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] break-words text-(--ui-text-tertiary)">
            {description}
          </div>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2">{controls}</div>
    </div>
  )
}

function DesktopPluginRow({
  record,
  agentHalfMissing,
  profile
}: {
  record: PluginRecord
  agentHalfMissing?: boolean
  profile: null | string
}) {
  const { t } = useI18n()
  const p = t.settings.plugins

  return (
    <PluginListRow
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
      icon={<Monitor aria-hidden className="mt-0.5 size-4 shrink-0 text-(--ui-text-tertiary)" />}
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
                onClick={() => void repairAgentHalf(record, unifiedPackageName(record.file) ?? record.name, profile)}
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

/** Plugins that extend THIS app (bundled, dropped into the desktop-plugins
 *  folder, or the desktop half of a unified package). They belong to the
 *  desktop, not to a profile, so the section is the same for every scope;
 *  the drift badge compares against the SCOPED profile's agent list so a
 *  bundled package missing its agent half where the user is looking gets a
 *  one-click repair. */
export function DesktopPluginsSection({ profile }: { profile: null | string }) {
  const { t } = useI18n()
  const p = t.settings.plugins
  const records = useStore($pluginRecords)
  const agentRows = useStore($agentPlugins)
  const agentStatus = useStore($agentPluginsStatus)
  const agentNames = new Set(agentRows.flatMap(row => [row.name, row.key ?? row.name]))

  const rows = Object.values(records).sort(
    (a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || a.name.localeCompare(b.name)
  )

  return (
    <section>
      <div className="flex items-center justify-between gap-3 px-3 pt-3 pb-1">
        <div className="min-w-0">
          <div className="text-[0.62rem] font-medium tracking-wide uppercase text-(--ui-text-quaternary)">
            {p.title} · {p.count(rows.length)}
          </div>
          <p className="mt-0.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {p.blurb}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Tip label={p.openFolder}>
            <Button onClick={() => void revealPluginsDir()} size="icon" type="button" variant="ghost">
              <FolderOpen className="size-3.5" />
            </Button>
          </Tip>
          <Tip label={p.rescan}>
            <Button
              onClick={() => {
                triggerHaptic('selection')
                void discoverRuntimePlugins()
              }}
              size="icon"
              type="button"
              variant="ghost"
            >
              <RefreshCw className="size-3.5" />
            </Button>
          </Tip>
        </div>
      </div>

      {rows.length === 0 ? (
        <p className="px-3 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {p.empty}
        </p>
      ) : (
        <div className="flex flex-col">
          {rows.map(record => {
            const packageName = unifiedPackageName(record.file)

            return (
              <DesktopPluginRow
                agentHalfMissing={packageName !== null && agentStatus === 'ready' && !agentNames.has(packageName)}
                key={record.id}
                profile={profile}
                record={record}
              />
            )
          })}
        </div>
      )}
    </section>
  )
}
