import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { getToolsets, setToolsetEnabled } from '@/api/toolsets'
import { pluginServersQueryKey } from '@/app/capabilities/connectors/data/keys'
import { useConnectorSwitch } from '@/app/capabilities/connectors/data/mutations'
import { useHostedConnectors, usePluginServers } from '@/app/capabilities/connectors/data/queries'
import { useMcpServers } from '@/app/capabilities/mcp/use-mcp-servers'
import { mergePluginPackages } from '@/app/capabilities/plugins/plugin-packages'
import { skillsQueryKey, useSkillsQuery } from '@/app/capabilities/skills/skills-data'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { DropdownMenuCheckboxItem, DropdownMenuItem, DropdownMenuSearch } from '@/components/ui/dropdown-menu'
import { $pluginRecords, enablePackageDesktopHalf, setPluginEnabled } from '@/contrib/plugins-store'
import { type HermesGateway, type ProfileScope, profileScopeKey, setSkillEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { DESKTOP_PLUGIN_TOOLSETS } from '@/lib/desktop-toolsets'
import { serverEnabled } from '@/lib/mcp-servers'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { type AgentPluginRow, isDesktopRelevantPlugin } from '@/store/agent-plugins'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import type { SkillInfo } from '@/types/hermes'

export type CapabilityKind = 'skills' | 'connectors' | 'plugins'

const gatewayIds = new WeakMap<HermesGateway, number>()
let nextGatewayId = 0

function gatewayCacheKey(gateway: HermesGateway | null | undefined, connectionId: string | null): string {
  const connection = connectionId ?? 'local'

  if (!gateway) {
    return connection
  }

  let id = gatewayIds.get(gateway)

  if (id === undefined) {
    id = ++nextGatewayId
    gatewayIds.set(gateway, id)
  }

  return `${connection}:${id}`
}

interface CapabilityMenuProps {
  gateway?: HermesGateway | null
  kind: CapabilityKind
  onInsertSkill: (name: string) => void
  onClose: () => void
  profile: ProfileScope
  sessionId?: string | null
}

type CapabilityListProps = CapabilityMenuProps & { query: string }
const matches = (text: string, query: string) => text.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())

const ROW = 'flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-[length:var(--conversation-tool-font-size)]'

function CapabilityRow({
  checked,
  disabled = false,
  label,
  onToggle,
  onUse,
  subtitle
}: {
  checked: boolean
  disabled?: boolean
  label: string
  onToggle?: (enabled: boolean) => void
  onUse?: () => void
  subtitle?: string
}) {
  const { t } = useI18n()

  return (
    <div className={ROW}>
      {onUse ? (
        <DropdownMenuItem className="min-w-0 flex-1 truncate px-1" onSelect={onUse}>
          <span className="min-w-0 truncate">
            <span className="block truncate">{label}</span>
            {subtitle && <span className="block truncate text-[0.65rem] text-(--ui-text-tertiary)">{subtitle}</span>}
          </span>
        </DropdownMenuItem>
      ) : null}
      {onToggle ? (
        <DropdownMenuCheckboxItem
          aria-label={`${label}: ${checked ? t.common.on : t.common.off}`}
          checked={checked}
          className={onUse ? 'shrink-0' : 'min-w-0 flex-1'}
          disabled={disabled}
          onCheckedChange={onToggle}
          onSelect={event => event.preventDefault()}
        >
          {onUse && <span>{checked ? t.common.on : t.common.off}</span>}
          {!onUse && (
            <span className="min-w-0 truncate">
              <span className="block truncate">{label}</span>
              {subtitle && <span className="block truncate text-[0.65rem] text-(--ui-text-tertiary)">{subtitle}</span>}
            </span>
          )}
        </DropdownMenuCheckboxItem>
      ) : !onUse ? (
        <span className="min-w-0 flex-1 truncate" title={subtitle || label}>
          {label}
          {subtitle && <span className="block truncate text-[0.65rem] text-(--ui-text-tertiary)">{subtitle}</span>}
          <span className="ml-2 text-[0.65rem] text-(--ui-text-tertiary)">{checked ? t.common.on : t.common.off}</span>
        </span>
      ) : null}
    </div>
  )
}

function SkillsMenu({
  onClose,
  onInsertSkill,
  profile,
  query
}: Pick<CapabilityListProps, 'onClose' | 'onInsertSkill' | 'profile' | 'query'>) {
  const { t } = useI18n()
  const { data, error, isPending } = useSkillsQuery(profile)
  const [busy, setBusy] = useState<string | null>(null)

  const skills = (data ?? [])
    .filter(skill => matches(skill.name, query) || matches(skill.description ?? '', query))
    .sort((a, b) => Number(b.enabled) - Number(a.enabled) || a.name.localeCompare(b.name))

  const toggle = async (skill: SkillInfo, enabled: boolean) => {
    if (busy) {
      return
    }

    setBusy(skill.name)

    try {
      await setSkillEnabled(skill.name, enabled, profile)
      await queryClient.invalidateQueries({ queryKey: skillsQueryKey(profile), exact: true })
      invalidateSlashCompletions()
    } catch (cause) {
      notifyError(cause, t.skills.failedToUpdate(skill.name))
    } finally {
      setBusy(null)
    }
  }

  if (error) {
    return <p className="px-2 py-2 text-xs text-destructive">{t.skills.skillsLoadFailed}</p>
  }

  if (isPending) {
    return <p className="px-2 py-2 text-xs text-(--ui-text-tertiary)">{t.common.loading}</p>
  }

  if (skills.length === 0) {
    return <p className="px-2 py-2 text-xs text-(--ui-text-tertiary)">{t.composer.capabilityMenu.noSkills}</p>
  }

  return skills.map(skill => (
    <CapabilityRow
      checked={skill.enabled}
      disabled={busy !== null}
      key={skill.name}
      label={skill.name}
      onToggle={enabled => void toggle(skill, enabled)}
      onUse={
        skill.enabled
          ? () => {
              onInsertSkill(skill.name)
              onClose()
            }
          : undefined
      }
      subtitle={skill.description}
    />
  ))
}

function PluginsMenu({ gateway, profile, query }: Pick<CapabilityListProps, 'gateway' | 'profile' | 'query'>) {
  const { t } = useI18n()
  const { requestGateway } = useGatewayRequest()

  const owner =
    profile && typeof profile === 'object'
      ? { connectionId: (profile.connectionId ?? '').trim(), profile: profile.profile ?? '' }
      : null

  const request = <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
    owner?.connectionId && owner.profile
      ? requestGatewayForAgent<T>(owner.connectionId, owner.profile, method, params ?? {})
      : gateway
        ? (gateway.request(method, params) as Promise<T>)
        : requestGateway<T>(method, params)

  const scope = profile && typeof profile === 'object' ? profile.profile : profile
  const desktop = useStore($pluginRecords)
  const connectionId = useStore($activeConnectionId)
  const [busy, setBusy] = useState<string | null>(null)

  const agent = useQuery({
    queryKey: ['composer-agent-plugins', profileScopeKey(profile), gatewayCacheKey(gateway, connectionId)],
    queryFn: async () => {
      const result = await request<{ plugins?: AgentPluginRow[] }>(
        'plugins.manage',
        scope ? { action: 'list', profile: scope } : { action: 'list' }
      )

      return (result.plugins ?? []).filter(isDesktopRelevantPlugin)
    },
    staleTime: 0
  })

  const toggleAgent = async (row: AgentPluginRow, enabled: boolean) => {
    if (busy || !row.key) {
      return
    }

    setBusy(row.key)

    try {
      const result = await request<{ ok?: boolean }>(
        'plugins.manage',
        scope
          ? { action: 'toggle', key: row.key, enable: enabled, profile: scope }
          : { action: 'toggle', key: row.key, enable: enabled }
      )

      if (!result.ok) {
        throw new Error(t.skills.plugins.toggleFailed(row.name))
      }

      await agent.refetch()
      void queryClient.invalidateQueries({ queryKey: pluginServersQueryKey(profile) })

      if (enabled) {
        const half = mergePluginPackages(Object.values(desktop), agent.data ?? []).find(pkg => pkg.agent?.key === row.key)?.desktop?.packageName

        if (half) {
          await enablePackageDesktopHalf(half, { keepUserChoice: true })
        }
      }
    } catch (cause) {
      notifyError(cause, t.skills.plugins.toggleFailed(row.name))
    } finally {
      setBusy(null)
    }
  }

  const toggleDesktop = async (id: string, enabled: boolean) => {
    if (busy) {
      return
    }

    setBusy(id)

    try {
      const toolset = DESKTOP_PLUGIN_TOOLSETS[id]

      if (toolset) {
        try {
          await setToolsetEnabled(toolset, enabled, profile)
        } catch (error) {
          // A timeout may have committed the write. Check before touching the desktop half.
          const actual = await getToolsets(profile).then(
            rows => rows.find(row => row.name === toolset)?.enabled,
            () => undefined
          )

          if (actual !== enabled) {
            throw error
          }
        }

        void queryClient.invalidateQueries({ queryKey: ['toolsets-list'] })
      }

      await setPluginEnabled(id, enabled)
    } catch (cause) {
      notifyError(cause, t.skills.plugins.toggleFailed(desktop[id]?.name ?? id))
    } finally {
      setBusy(null)
    }
  }

  const desktopRows = Object.values(desktop)
    .filter(row => matches(row.name, query))
    .sort((a, b) => a.name.localeCompare(b.name))

  const agentRows = (agent.data ?? [])
    .filter(row => matches(row.name, query))
    .sort((a, b) => a.name.localeCompare(b.name))

  return (
    <>
      {agentRows.length > 0 && (
        <p className="px-2 pt-1 text-[0.625rem] font-semibold uppercase text-(--ui-text-tertiary)">
          {t.skills.plugins.kindAgent}
        </p>
      )}
      {agentRows.map(row => (
        <CapabilityRow
          checked={row.status === 'enabled'}
          disabled={busy !== null}
          key={row.key ?? row.name}
          label={row.name}
          onToggle={row.key ? enabled => void toggleAgent(row, enabled) : undefined}
          subtitle={row.description}
        />
      ))}
      {desktopRows.length > 0 && (
        <p className="px-2 pt-2 text-[0.625rem] font-semibold uppercase text-(--ui-text-tertiary)">
          {t.skills.plugins.kindDesktop}
        </p>
      )}
      {desktopRows.map(row => (
        <CapabilityRow
          checked={row.status !== 'disabled'}
          disabled={busy !== null}
          key={row.id}
          label={row.name}
          onToggle={enabled => void toggleDesktop(row.id, enabled)}
          subtitle={row.description}
        />
      ))}
      {agent.isPending && <p className="px-2 py-2 text-xs text-(--ui-text-tertiary)">{t.common.loading}</p>}
      {agent.isError && (
        <p className="px-2 py-2 text-xs text-destructive">{t.composer.capabilityMenu.loadPluginsFailed}</p>
      )}
      {!agent.isPending && !agent.isError && !agentRows.length && !desktopRows.length && (
        <p className="px-2 py-2 text-xs text-(--ui-text-tertiary)">{t.composer.capabilityMenu.noPlugins}</p>
      )}
    </>
  )
}

function ConnectorsMenu({ gateway, profile, query, sessionId }: Pick<CapabilityListProps, 'gateway' | 'profile' | 'query' | 'sessionId'>) {
  const { t } = useI18n()
  const { gateway: activeGateway } = useGatewayRequest()
  const hosted = useHostedConnectors(profile)
  const switcher = useConnectorSwitch(profile)
  const mcp = useMcpServers({ gateway: gateway ?? activeGateway, profile, sessionId })
  const pluginServers = usePluginServers(profile)
  const [localBusy, setLocalBusy] = useState<string | null>(null)

  const installedHosted = hosted.rows.filter(
    row => row.connected && matches(hosted.titles[row.slug] ?? row.slug, query)
  )

  const localNames = mcp.names.filter(name => matches(name, query)).sort((a, b) => a.localeCompare(b))
  const visiblePluginServers = pluginServers.filter(row => matches(row.title ?? row.name, query))

  const toggleHosted = async (slug: string, enabled: boolean) => {
    const result = await switcher.setEnabled(slug, enabled)

    if (!result.ok) {
      notifyError(result.error, t.connectorsPage.page.writeFailed)
    }
  }

  const toggleLocal = async (name: string, enabled: boolean) => {
    if (localBusy) {
      return
    }

    setLocalBusy(name)

    try {
      await mcp.setServerEnabled(name, enabled)
    } finally {
      setLocalBusy(null)
    }
  }

  return (
    <>
      {installedHosted.length > 0 && (
        <p className="px-2 pt-1 text-[0.625rem] font-semibold uppercase text-(--ui-text-tertiary)">
          {t.connectorsPage.title}
        </p>
      )}
      {installedHosted.map(row => (
        <CapabilityRow
          checked={row.enabled}
          disabled={switcher.pending !== null || row.orgLocked || hosted.rulesFailed}
          key={row.slug}
          label={hosted.titles[row.slug] ?? row.slug}
          onToggle={enabled => void toggleHosted(row.slug, enabled)}
          subtitle={row.accountLabel}
        />
      ))}
      {(localNames.length > 0 || visiblePluginServers.length > 0) && (
        <p className="px-2 pt-2 text-[0.625rem] font-semibold uppercase text-(--ui-text-tertiary)">
          {t.composer.capabilityMenu.mcpServers}
        </p>
      )}
      {localNames.map(name => (
        <CapabilityRow
          checked={serverEnabled(mcp.servers[name]!)}
          disabled={localBusy !== null || mcp.profilePending || mcp.configFailed}
          key={name}
          label={name}
          onToggle={enabled => void toggleLocal(name, enabled)}
        />
      ))}
      {visiblePluginServers.map(row => (
        <CapabilityRow
          checked={row.enabled}
          key={`plugin:${row.name}`}
          label={row.title ?? row.name}
          subtitle={row.plugin}
        />
      ))}
      {(hosted.phase === 'loading' || mcp.configLoading) && (
        <p className="px-2 py-2 text-xs text-(--ui-text-tertiary)">{t.common.loading}</p>
      )}
      {hosted.phase !== 'loading' &&
        !mcp.configLoading &&
        !installedHosted.length &&
        !localNames.length &&
        !visiblePluginServers.length && (
          <p className="px-2 py-2 text-xs text-(--ui-text-tertiary)">{t.composer.capabilityMenu.noConnectors}</p>
        )}
    </>
  )
}

/** Lazy-mounted only while its + menu category is open. No background requests per composer. */
export function CapabilityMenu({ gateway, kind, onClose, onInsertSkill, profile, sessionId }: CapabilityMenuProps) {
  const { t } = useI18n()
  const activeProfile = useStore($activeGatewayProfile)
  const scope = profile ?? activeProfile ?? null
  const [query, setQuery] = useState('')

  return (
    <div>
      <DropdownMenuSearch
        aria-label={t.composer.capabilityMenu.search(
          kind === 'skills' ? t.skills.tabSkills : kind === 'plugins' ? t.skills.tabPlugins : t.connectorsPage.title
        )}
        onValueChange={setQuery}
        placeholder={
          kind === 'skills' ? t.skills.tabSkills : kind === 'plugins' ? t.skills.tabPlugins : t.connectorsPage.title
        }
        role="searchbox"
        value={query}
      />
      <div className="max-h-72 overflow-y-auto">
        {kind === 'skills' ? (
          <SkillsMenu onClose={onClose} onInsertSkill={onInsertSkill} profile={scope} query={query} />
        ) : null}
        {kind === 'plugins' ? <PluginsMenu gateway={gateway} profile={scope} query={query} /> : null}
        {kind === 'connectors' ? (
          <ConnectorsMenu gateway={gateway} profile={scope} query={query} sessionId={sessionId} />
        ) : null}
      </div>
    </div>
  )
}
