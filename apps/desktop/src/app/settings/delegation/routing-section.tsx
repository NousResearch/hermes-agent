import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import type { DelegationRouting, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import type { DelegationRoutingTranslations } from '@/i18n/types_delegation_routing'
import { AlertTriangle, Check, GitFork, Plug, SlidersHorizontal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $settingsRequestProfile } from '@/store/settings-scope'

import { useOwnerCatalog } from '../delegation-model-provider-field'
import { Pill, SectionHeading, SettingsSection } from '../primitives'

import { ModelRoutesEditor } from './model-routes'
import { enableAdvancedDelegation, installAdvancedDelegation, useAdvancedDelegationPlugin } from './plugin-actions'
import { ProfileRoutesEditor } from './profile-routes'
import {
  chooseDelegationMode,
  isPluginMode,
  modeAvailable,
  modeChoices,
  MODES,
  PLUGIN_MODES,
  refreshDelegationRouting,
  useDelegationRouting
} from './routing-data'

interface DelegationRoutingSlotProps {
  sectionId: string
  /** Undefined = the whole section is shown (embedded consumers). */
  subpage?: string
  /** The config page's owner: the read origin once the config loaded, else the Applies-to profile. */
  writeScope?: ProfileScope
  readScope?: string
}

/** The Routing section on the page that owns `delegation.*` (Model › Subagents); nothing elsewhere. */
export function DelegationRoutingSlot({ sectionId, subpage, writeScope, readScope }: DelegationRoutingSlotProps) {
  if (sectionId !== 'model' || (subpage !== undefined && subpage !== 'delegation')) {
    return null
  }

  return <DelegationRoutingSection scope={writeScope ?? readScope} />
}

/** Settings › Model › Subagents: how each subagent gets its model, and the routes of the chosen mode. */
export function DelegationRoutingSection({ scope }: { scope: ProfileScope }) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const query = useDelegationRouting(scope)
  const routing = query.data

  if (routing === null) {
    return null
  }

  return (
    <>
      <SettingsSection icon={GitFork} title={r.title}>
        <p className="mb-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{r.intro}</p>
        {routing ? (
          <RoutingBody routing={routing} scope={scope} />
        ) : query.isError ? (
          <div className="flex items-center gap-2" role="status">
            <p className="text-xs text-muted-foreground">{r.loadFailed}</p>
            <Button disabled={query.isFetching} onClick={() => void query.refetch()} size="xs" variant="secondary">
              {t.common.retry}
            </Button>
          </div>
        ) : (
          <div className="grid gap-1.5">
            {MODES.map(mode => (
              <Skeleton className="h-14 rounded-lg" key={mode} />
            ))}
          </div>
        )}
      </SettingsSection>
      <SectionHeading icon={SlidersHorizontal} title={r.settingsTitle} />
    </>
  )
}

function RoutingBody({ routing, scope }: { routing: DelegationRouting; scope: ProfileScope }) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const mode = routing.configured
  const blocked = Boolean(routing.blocked_reason)
  const { providers } = useOwnerCatalog(scope)
  const providerNames = new Map(providers.map(p => [p.slug, p.name]))
  const pluginNeeded = (isPluginMode(mode) && blocked) || PLUGIN_MODES.some(option => !modeAvailable(routing, option))

  return (
    <>
      <div aria-label={r.title} className="grid gap-1.5" role="radiogroup">
        {modeChoices(routing).map(option => (
          <ModeOption
            available={modeAvailable(routing, option)}
            blocked={blocked && option === mode}
            key={option}
            mode={option}
            onChoose={() => void chooseDelegationMode(scope, option, r.modeFailed)}
            routing={routing}
            selected={option === mode}
          />
        ))}
      </div>
      {pluginNeeded ? <AdvancedDelegationStatus blocked={blocked && isPluginMode(mode)} scope={scope} /> : null}
      {blocked && !isPluginMode(mode) && <BlockedBanner message={thirdPartyBlockedMessage(routing, r)} scope={scope} />}
      {mode === 'model_map' && (
        <div className="mt-6">
          <ModelRoutesEditor providerNames={providerNames} routes={routing.model_map} scope={scope} />
        </div>
      )}
      {mode === 'profile_routes' && (
        <div className="mt-6">
          <ProfileRoutesEditor profiles={routing.profiles} routes={routing.profile_routes} scope={scope} />
        </div>
      )}
    </>
  )
}

/** A saved third-party backend that cannot run: registered but unavailable, or no plugin provides it at all. */
function thirdPartyBlockedMessage(routing: DelegationRouting, r: DelegationRoutingTranslations): string {
  const backend = routing.backends.find(b => b.name === routing.configured)

  return backend ? r.blockedUnavailable(backend.label || backend.name) : r.blockedUnregistered(routing.configured)
}

/** Label + description of a picker row: the copy of a mode the page edits, else the plugin backend's own. */
function modeCopy(routing: DelegationRouting, mode: string, r: DelegationRoutingTranslations) {
  if (mode === 'default' || isPluginMode(mode)) {
    return r.modes[mode]
  }

  const backend = routing.backends.find(b => b.name === mode)

  return { label: backend?.label || mode, description: backend?.description || r.thirdPartyDescription }
}

interface ModeOptionProps {
  routing: DelegationRouting
  mode: string
  selected: boolean
  blocked: boolean
  available: boolean
  onChoose: () => void
}

function ModeOption({ routing, mode, selected, blocked, available, onChoose }: ModeOptionProps) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const copy = modeCopy(routing, mode, r)

  return (
    <div
      className={cn(
        'rounded-lg border transition-colors hover:transition-none',
        selected
          ? blocked
            ? 'border-amber-500/50 bg-amber-500/5'
            : 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
          : 'border-(--ui-stroke-tertiary)',
        !selected && available && 'hover:bg-(--chrome-action-hover)'
      )}
      data-blocked={blocked || undefined}
      data-mode={mode}
    >
      <button
        aria-checked={selected}
        className="flex w-full items-start gap-3 px-3 py-2.5 text-left disabled:cursor-default"
        disabled={!available || selected}
        onClick={onChoose}
        role="radio"
        type="button"
      >
        <span
          aria-hidden
          className={cn(
            'mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border',
            selected
              ? 'border-primary bg-primary text-primary-foreground'
              : 'border-(--ui-stroke-primary) bg-background',
            !available && !selected && 'opacity-60'
          )}
        >
          {selected && <Check className="size-2.5" />}
        </span>
        <span className="grid min-w-0 gap-0.5">
          <span className="flex flex-wrap items-center gap-2">
            <span
              className={cn(
                'text-[length:var(--conversation-text-font-size)] font-medium',
                available || selected ? 'text-foreground' : 'text-(--ui-text-secondary)'
              )}
            >
              {copy.label}
            </span>
            {selected && (blocked ? <Pill tone="warn">{r.blockedPill}</Pill> : <Pill tone="primary">{r.inUse}</Pill>)}
            {mode !== 'default' && (!available || !isPluginMode(mode)) && !blocked && (
              <Pill tone="muted">
                <Plug className="size-3" />
                {r.plugin}
              </Pill>
            )}
          </span>
          <span className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {copy.description}
          </span>
        </span>
      </button>
    </div>
  )
}

/** The saved mode cannot run, so core refuses every subagent: say so, the fix, and the way back to Default. */
function BlockedBanner({ message, action, scope }: { message: string; action?: ReactNode; scope: ProfileScope }) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing

  return (
    <div
      className="mt-2 flex flex-wrap items-start gap-x-3 gap-y-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2.5"
      data-slot="delegation-blocked"
      role="alert"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-300" />
      <div className="grid min-w-0 flex-1 basis-64 gap-0.5">
        <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">{message}</span>
        <span className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-secondary)">
          {r.blockedDetail}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {action}
        <Button onClick={() => void chooseDelegationMode(scope, 'default', r.modeFailed)} size="sm" variant="secondary">
          {r.useDefault}
        </Button>
      </div>
    </div>
  )
}

/** Where the Advanced Delegation plugin stands for the owner: a hint under the picker, or, when the saved mode
 *  is one of its own, the blocked banner with the fix (Install / Turn on / restart). */
function AdvancedDelegationStatus({ blocked, scope }: { blocked: boolean; scope: ProfileScope }) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const { requestGateway } = useGatewayRequest()
  const owner = useStore($settingsRequestProfile) ?? null
  const { row, state } = useAdvancedDelegationPlugin(requestGateway, owner)

  if (state === 'loading') {
    return blocked ? <BlockedBanner message={r.blockedMissing} scope={scope} /> : null
  }

  const action =
    state === 'missing' ? (
      <Button onClick={() => void installAdvancedDelegation(owner, r.installUnavailable)} size="sm" variant="outline">
        {r.installPlugin}
      </Button>
    ) : state === 'disabled' && row?.key ? (
      <Button
        onClick={() =>
          void enableAdvancedDelegation(requestGateway, row.key!, owner, r.enableFailed).then(ok => {
            if (ok) {
              void refreshDelegationRouting(scope)
            }
          })
        }
        size="sm"
        variant="outline"
      >
        {r.enablePlugin}
      </Button>
    ) : null

  if (blocked) {
    const message = { disabled: r.blockedDisabled, missing: r.blockedMissing, not_loaded: r.blockedNotLoaded }[state]

    return <BlockedBanner action={action} message={message} scope={scope} />
  }

  const message = { disabled: r.pluginDisabled, missing: r.pluginMissing, not_loaded: r.pluginNotLoaded }[state]

  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 px-1" data-slot="delegation-plugin-hint">
      <Plug className="size-3.5 shrink-0 text-(--ui-text-tertiary)" />
      <span className="text-xs text-(--ui-text-secondary)">{message}</span>
      {action}
    </div>
  )
}
