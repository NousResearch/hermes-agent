/**
 * The advanced profile editor: skills, toolsets, MCP servers, model and SOUL,
 * backed by `profiles.describe` / `profiles.configure`.
 *
 * The optional-SDK feature detects live here because this is the surface that
 * needs them, and the create dialog — which stages the same capabilities
 * before a profile exists — reads them from here rather than duplicating the
 * detection.
 */

import * as sdk from '@hermes/plugin-sdk'
import {
  Checkbox,
  GlyphSpinner,
  host,
  Input,
  queryClient,
  surfaceModelSwitchConfirm,
  Textarea
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { $lastRoster, ROSTER_KEY } from './data'
import { labeled, ResizableFrame } from './dialog-parts'
import { useBots } from './i18n'
import { McpSetupButton } from './mcp-setup'
import { ModelPicker } from './model-picker'
import { botBackendProfileScope, requestForBot, resolveBotConnectionRoute } from './routing'
import { HubSkillsSection } from './skills-hub'
import { ensureMessagingProtocol } from './soul'
import type { RosterRow } from './types'

// Keep optional exports feature-detected; test harnesses may strip the SDK namespace.
// The Partial is the point: both are guarded at every use site because an older
// build (or a stripped harness namespace) simply doesn't export them.
const { McpTab, ToolsetConfigPanel }: Partial<Pick<typeof sdk, 'McpTab' | 'ToolsetConfigPanel'>> = sdk
export const SkillsView = typeof sdk === 'undefined' ? undefined : sdk.SkillsView
// TRUE only on builds whose SkillsView routes `fixedConnection` to the pinned
// registry connection's backend. Older builds export SkillsView WITHOUT the
// prop — rendering it for a remote-target draft there would read/write the
// ACTIVE gateway's skills under the remote bot's name (the wrong machine),
// so those builds keep the staged checklists for remote targets.
export const skillsViewRoutesConnections = Boolean(SkillsView && SkillsView.supportsFixedConnection)

// ── advanced profile config (skills / toolsets / model / SOUL) ──────────────
//
// Shared by Edit Profile and New Bot (edit mode only for skills/toolsets —
// a not-yet-created profile has nothing installed to toggle). Backed by
// profiles.describe / profiles.configure; feature-detects older gateways.

/** One toggleable capability row. Skills, toolsets, and MCP servers all
 *  arrive from `profiles.describe` / `mcp.catalog` in this shape and run
 *  through the same toggle handlers, so they share one entry type. */
export interface CapabilityEntry {
  auth?: string
  description?: string
  enabled?: boolean
  fromCatalog?: boolean
  installed?: boolean
  name: string
  requires?: string[]
  tool_count?: number
}
interface CheckListProps {
  columns?: number
  items: CapabilityEntry[]
  onToggle: (name: string, enabled: boolean) => void
}

export function CheckList({ items, onToggle, columns = 2 }: CheckListProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
        gap: '2px 12px'
      }}
    >
      {items.map(item => (
        <label
          className="flex min-w-0 cursor-pointer items-center gap-1.5 py-0.5 text-xs text-(--ui-text-secondary)"
          key={item.name}
          title={item.description || item.name}
        >
          <Checkbox checked={item.enabled} onCheckedChange={value => onToggle(item.name, Boolean(value))} />
          <span className="truncate">{item.name}</span>
          {item.tool_count ? (
            <span className="shrink-0 text-[0.6rem] text-(--ui-text-quaternary)">{`${item.tool_count}`}</span>
          ) : null}
        </label>
      ))}
    </div>
  )
}

/** `profiles.describe` as the advanced editor reads it. */
export interface ProfileDescribeResponse {
  mcp_servers?: CapabilityEntry[]
  model?: { default?: string; provider?: string }
  skills?: CapabilityEntry[]
  soul?: string
  toolsets?: CapabilityEntry[]
}
/** `mcp.catalog` — the bundled, installable MCP menu. */
export interface McpCatalogResponse {
  servers?: CapabilityEntry[]
}
/** Staged advanced-config edits. Each `dirty*` flag gates one section of the
 *  `profiles.configure` payload, so an untouched section is never written. */
interface AdvancedConfigState {
  dirtyMcp: boolean
  dirtyModel: boolean
  dirtySkills: boolean
  dirtySoul: boolean
  dirtyToolsets: boolean
  loaded: boolean
  mcp: CapabilityEntry[]
  model: string
  provider: string
  skills: CapabilityEntry[]
  soul: string
  toolsets: CapabilityEntry[]
}
interface AdvancedProfileConfigProps {
  bot: RosterRow
  setState: (update: (prev: AdvancedConfigState) => AdvancedConfigState) => void
  state: AdvancedConfigState
}

export function AdvancedProfileConfig({ bot, state, setState }: AdvancedProfileConfigProps) {
  const b = useBots()
  const [loaded, setLoaded] = useState(false)
  const [unsupported, setUnsupported] = useState(false)
  const [skillFilter, setSkillFilter] = useState('')
  // Component body = render path: degrade an orphaned row to the bot's own
  // name scope instead of throwing into the dialog's error boundary.
  const botRoute = resolveBotConnectionRoute(bot).route
  const backendProfile = botRoute?.targetProfile || botRoute?.profile || bot.name
  const backendScope = botBackendProfileScope(botRoute, bot.name)

  if (!loaded) {
    setLoaded(true)
    Promise.all([
      requestForBot(bot, 'profiles.describe', {
        name: bot.name
      }) as Promise<ProfileDescribeResponse>,
      (
        requestForBot(bot, 'mcp.catalog', {
          profile: bot.name
        }) as Promise<McpCatalogResponse>
      ).catch(() => null)
    ])
      .then(([res, cat]) => {
        const configured = res.mcp_servers || []
        const have = new Set(configured.map(m => m.name))
        const catalog = ((cat && cat.servers) || []).filter(s => !have.has(s.name))
        setState(prev => ({
          ...prev,
          provider: res.model?.provider || '',
          model: res.model?.default || '',
          soul: res.soul || '',
          skills: res.skills || [],
          toolsets: res.toolsets || [],
          mcp: [
            ...configured.map(m => ({
              ...m,
              enabled: m.enabled !== false
            })),
            ...catalog.map(s => ({
              name: s.name,
              enabled: false,
              fromCatalog: true,
              installed: s.installed,
              auth: s.auth,
              requires: s.requires || [],
              description: s.description || ''
            }))
          ],
          loaded: true
        }))
      })
      .catch(() => setUnsupported(true))
  }

  if (unsupported) {
    return (
      <div className="px-2 py-3 text-center text-xs text-(--ui-text-tertiary)">
        Full configuration needs a newer gateway (restart it after updating Hermes).
      </div>
    )
  }

  if (!state.loaded) {
    return (
      <div className="flex justify-center py-4">
        <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
      </div>
    )
  }

  const visibleSkills = skillFilter.trim()
    ? state.skills.filter(s => s.name.toLowerCase().includes(skillFilter.trim().toLowerCase()))
    : state.skills

  const toggleSkill = (name: string, enabled: boolean) =>
    setState(prev => ({
      ...prev,
      dirtySkills: true,
      skills: prev.skills.map(s =>
        s.name === name
          ? {
              ...s,
              enabled
            }
          : s
      )
    }))

  const toggleToolset = (name: string, enabled: boolean) =>
    setState(prev => ({
      ...prev,
      dirtyToolsets: true,
      toolsets: prev.toolsets.map(t =>
        t.name === name
          ? {
              ...t,
              enabled
            }
          : t
      )
    }))

  const toggleMcp = (name: string, enabled: boolean) =>
    setState(prev => ({
      ...prev,
      dirtyMcp: true,
      mcp: (prev.mcp || []).map(m =>
        m.name === name
          ? {
              ...m,
              enabled
            }
          : m
      )
    }))

  const enabledSkills = state.skills.filter(s => s.enabled).length
  const enabledToolsets = state.toolsets.filter(t => t.enabled).length
  const mcpList = state.mcp || []

  // Newer desktop builds export the WHOLE core Capabilities surface
  // (hermes-agent#87317): Skills (installed list + one-click hub installs +
  // full-skill detail), Tools (per-toolset config), and MCP — pinned to this
  // bot via fixedProfile, tab state kept out of the page router via embedded.
  // Render THAT instead of the checkbox stand-ins; writes go straight to the
  // bot's backend, so the dirty-section staging below only carries
  // model + SOUL on these builds. Older builds keep the full checklist UI.
  if (SkillsView && (!botRoute || skillsViewRoutesConnections)) {
    return (
      <div className="grid gap-4">
        <ModelPicker
          bot={bot}
          onChange={patch =>
            setState(prev => ({
              ...prev,
              dirtyModel: true,
              ...patch
            }))
          }
          value={{
            provider: state.provider,
            model: state.model
          }}
        />
        {labeled(
          'Capabilities (applies immediately — skills, tools, MCP)',
          <ResizableFrame height={460} minHeight={300}>
            <SkillsView
              embedded
              fixedProfile={backendProfile}
              {...(botRoute
                ? {
                    fixedConnection: botRoute.connectionId
                  }
                : {})}
            />
          </ResizableFrame>
        )}
        {labeled(
          'SOUL.md (persona + agent-messaging protocol)',
          <Textarea
            className="min-h-28 font-mono text-xs leading-5"
            onChange={event =>
              setState(prev => ({
                ...prev,
                dirtySoul: true,
                soul: event.target.value
              }))
            }
            value={state.soul}
          />
        )}
      </div>
    )
  }

  if (bot?.sourceScoped && botRoute?.mode === 'remote' && !skillsViewRoutesConnections) {
    return (
      <div className="grid gap-4">
        <ModelPicker
          bot={bot}
          onChange={patch =>
            setState(prev => ({
              ...prev,
              dirtyModel: true,
              ...patch
            }))
          }
          value={{
            provider: state.provider,
            model: state.model
          }}
        />
        <div className="rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs text-(--ui-text-tertiary)">
          Remote capabilities require a newer desktop. Model and SOUL changes remain staged until you save.
        </div>
        {labeled(
          'SOUL.md (persona + agent-messaging protocol)',
          <Textarea
            className="min-h-28 font-mono text-xs leading-5"
            onChange={event =>
              setState(prev => ({
                ...prev,
                dirtySoul: true,
                soul: event.target.value
              }))
            }
            value={state.soul}
          />
        )}
      </div>
    )
  }

  return (
    <div className="grid gap-4">
      <ModelPicker
        bot={bot}
        onChange={patch =>
          setState(prev => ({
            ...prev,
            dirtyModel: true,
            ...patch
          }))
        }
        value={{
          provider: state.provider,
          model: state.model
        }}
      />
      {labeled(
        `Skills (${enabledSkills}/${state.skills.length} enabled)`,
        <div className="grid gap-1.5 rounded-md border border-(--ui-stroke-secondary) p-2">
          <Input
            className="h-7 text-xs"
            onChange={event => setSkillFilter(event.target.value)}
            placeholder={b.tools.filterSkills}
            value={skillFilter}
          />
          <div
            className="overflow-y-auto overscroll-contain"
            style={{
              maxHeight: 180
            }}
          >
            <CheckList columns={2} items={visibleSkills} onToggle={toggleSkill} />
          </div>
          <HubSkillsSection
            forProfile={backendScope}
            onInstalled={name =>
              setState(prev =>
                prev.skills.some(s => s.name === name)
                  ? prev
                  : {
                      ...prev,
                      skills: [
                        ...prev.skills,
                        {
                          name,
                          enabled: true
                        }
                      ]
                    }
              )
            }
          />
        </div>
      )}
      {labeled(
        `Toolsets (${enabledToolsets}/${state.toolsets.length} enabled — unchecking all restores the default)`,
        <div className="rounded-md border border-(--ui-stroke-secondary) p-2">
          <div
            className="overflow-y-auto overscroll-contain"
            style={{
              maxHeight: 320
            }}
          >
            <div className="grid gap-1.5">
              {state.toolsets.map(tset => (
                <div className="rounded-md border border-(--ui-stroke-secondary) p-2" key={tset.name}>
                  <label className="flex items-center gap-2 text-xs font-medium text-(--ui-text-secondary)">
                    <Checkbox
                      checked={!!tset.enabled}
                      onCheckedChange={value => toggleToolset(tset.name, Boolean(value))}
                    />
                    <span>{tset.name}</span>
                  </label>
                  {/* The REAL per-toolset config (env vars / API keys / model */
                  /* picker / post-setup), scoped to THIS bot's profile, when */
                  /* the desktop build exposes it. Older builds: just the toggle. */}
                  {ToolsetConfigPanel ? (
                    <div className="mt-1.5 border-t border-(--ui-stroke-secondary) pt-1.5">
                      <ToolsetConfigPanel profile={backendScope} toolset={tset.name} />
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      {labeled(
        'MCP servers',
        <div className="overflow-hidden rounded-md border border-(--ui-stroke-secondary)">
          {McpTab && typeof host.getGateway === 'function' ? (
            <div
              style={{
                minHeight: 220,
                maxHeight: 360
              }}
            >
              <McpTab gateway={host.getGateway()} profile={backendScope} />
            </div>
          ) : mcpList.length === 0 ? (
            <div className="px-1 py-2 text-center text-xs text-(--ui-text-tertiary)">{b.tools.noMcpServers}</div>
          ) : (
            <div
              className="overflow-y-auto overscroll-contain"
              style={{
                maxHeight: 180
              }}
            >
              <div className="grid gap-1 p-2">
                {mcpList.map(m => {
                  const needsSetup =
                    m.fromCatalog &&
                    !m.installed &&
                    ((m.requires || []).length > 0 || (m.auth || '').toLowerCase() === 'oauth')

                  return (
                    <label className="flex items-start gap-2 text-xs text-(--ui-text-secondary)" key={m.name}>
                      <Checkbox
                        checked={!!m.enabled}
                        disabled={needsSetup}
                        onCheckedChange={value => toggleMcp(m.name, Boolean(value))}
                      />
                      <span className="min-w-0">
                        <span>{m.name}</span>
                        {m.fromCatalog && !needsSetup ? (
                          <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">
                            {m.installed ? 'catalog · installed' : 'catalog'}
                          </span>
                        ) : null}
                        {needsSetup ? (
                          <McpSetupButton entry={m} onDone={() => toggleMcp(m.name, true)} profile={backendScope} />
                        ) : null}
                        {m.description ? (
                          <div className="truncate text-[0.65rem] leading-4 text-(--ui-text-quaternary)">
                            {m.description}
                          </div>
                        ) : null}
                      </span>
                    </label>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
      {labeled(
        'SOUL.md (persona + agent-messaging protocol)',
        <Textarea
          className="min-h-28 font-mono text-xs leading-5"
          onChange={event =>
            setState(prev => ({
              ...prev,
              dirtySoul: true,
              soul: event.target.value
            }))
          }
          value={state.soul}
        />
      )}
    </div>
  )
}

export function emptyAdvancedState(): AdvancedConfigState {
  return {
    loaded: false,
    provider: '',
    model: '',
    soul: '',
    skills: [],
    toolsets: [],
    mcp: [],
    dirtyModel: false,
    dirtySoul: false,
    dirtySkills: false,
    dirtyToolsets: false,
    dirtyMcp: false
  }
}

/** The `profiles.configure` body this editor writes — one key per dirty
 *  section. A type alias, not an interface: it is handed straight to
 *  `requestForBot`, whose params are a `Record<string, unknown>`, and only
 *  aliases pick up the implicit index signature that requires. */
export type ProfileConfigurePayload = {
  disabled_skills?: string[]
  enabled_mcp_servers?: string[]
  enabled_toolsets?: string[]
  model?: string
  name: string
  provider?: string
  soul?: string
}
/** What `profiles.configure` answers: per-section success, plus the #95293
 *  expensive-model / data-policy confirmation gate. */
interface ProfileConfigureResult {
  applied?: Record<string, boolean>
  confirm_message?: string
  confirm_required?: boolean
}

/** Persist only the dirty sections of the advanced editor. */
export async function applyAdvancedConfig(bot: RosterRow, state: AdvancedConfigState) {
  const payload: ProfileConfigurePayload = {
    name: bot.name
  }

  const applied: Record<string, boolean> = {}

  if (state.dirtySoul) {
    payload.soul = ensureMessagingProtocol(state.soul, bot.name, $lastRoster.get())
  }

  if (state.dirtyModel) {
    const model = state.model.trim()
    const provider = state.provider.trim()

    if (model && provider) {
      payload.model = model
      payload.provider = provider
    } else if (!model && !provider) {
      try {
        const result = (await requestForBot(bot, 'cli.exec', {
          argv: ['--profile', bot.name, 'config', 'unset', 'model']
        })) as { blocked?: boolean; code?: number }

        applied.model = result?.blocked !== true && result?.code === 0
      } catch {
        applied.model = false
      }
    } else {
      applied.model = false
    }
  }

  if (state.dirtySkills) {
    payload.disabled_skills = state.skills.filter(s => !s.enabled).map(s => s.name)
  }

  if (state.dirtyToolsets) {
    const all = state.toolsets.length
    const enabled = state.toolsets.filter(t => t.enabled)
    // All enabled (or none) = clear the pin; otherwise pin the checked set.
    payload.enabled_toolsets = enabled.length === all || enabled.length === 0 ? [] : enabled.map(t => t.name)
  }

  if (state.dirtyMcp) {
    payload.enabled_mcp_servers = (state.mcp || []).filter(m => m.enabled).map(m => m.name)
  }

  if (Object.keys(payload).length === 1) {
    return {
      ok: Object.values(applied).every(Boolean),
      applied
    }
  }

  const result = (await requestForBot(bot, 'profiles.configure', payload)) as ProfileConfigureResult

  const merged = {
    ...applied,
    ...(result?.applied || {})
  }

  // #95293 remainder: the gateway now guards data-policy / expensive models
  // on THIS surface too — `confirm_required` means the model section is
  // PENDING the user's confirmation, not failed. Route it through the SAME
  // shared confirm handler the core picker uses (one applier, no forked
  // confirm logic per surface): the Confirm action resends ONLY the model
  // section with `confirm_expensive_model: true`.
  if (result?.confirm_required && payload.model && payload.provider) {
    delete merged.model
    surfaceModelSwitchConfirm({
      confirmLabel: 'Confirm',
      confirmMessage: result.confirm_message,
      failureMessage: 'Model switch failed',
      finish: () =>
        queryClient.invalidateQueries({
          queryKey: ROSTER_KEY
        }),
      requestConfirmed: () =>
        requestForBot(bot, 'profiles.configure', {
          name: bot.name,
          model: payload.model,
          provider: payload.provider,
          confirm_expensive_model: true
        }) as Promise<ProfileConfigureResult>
    })
  }

  return {
    ...result,
    ok: Object.values(merged).every(Boolean),
    applied: merged
  }
}
