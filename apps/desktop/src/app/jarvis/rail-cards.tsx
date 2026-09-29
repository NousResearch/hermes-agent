import { useStore } from '@nanostores/react'
/**
 * The dashboard's right-rail cards: the model and work mode, live AI news and
 * the agents (profiles) on this machine.
 *
 * Each card paints real state only — the served model catalog, the backend's
 * aggregated feeds, the profile list — and each owns its empty and failure
 * states, so one broken source never blanks the rail.
 */
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { type ReactNode, useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { getAiNews } from '@/hermes'
import { useI18n } from '@/i18n'
import { ArrowUpRight, Brain, Check, Cpu, Loader2, RefreshCw, Users, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles, profileLabel } from '@/store/profile'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $defaultReasoningEffort,
  markComposerSelectionManual,
  setCurrentReasoningEffort
} from '@/store/session'
import type { ModelOptionProvider } from '@/types/hermes'

import { PROFILES_ROUTE, SETTINGS_ROUTE } from '../routes'

import { LiveModelPicker } from './live-model-picker'
import {
  JARVIS_WORK_MODES,
  type JarvisWorkModeId,
  OPENROUTER_PROVIDER_SLUG,
  type OpenRouterPresetId,
  resolveOpenRouterPresets,
  workModeForEffort
} from './openrouter-presets'
import { OpenRouterQuickConnect } from './openrouter-quick-connect'

type GatewayRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

export function RailCard({
  action,
  children,
  icon: Icon,
  title,
  testId
}: {
  action?: ReactNode
  children: ReactNode
  icon: React.ComponentType<{ className?: string }>
  title: string
  testId: string
}) {
  const headingId = `jarvis-rail-${testId}`

  return (
    <section aria-labelledby={headingId} className="jarvis-panel p-4" data-testid={`jarvis-rail-${testId}`}>
      <div className="mb-3 flex min-h-9 items-center gap-2">
        <Icon className="size-4 shrink-0 text-(--ui-accent)" />
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-(--ui-text-primary)" id={headingId}>
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  )
}

function LinkAction({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      className="inline-flex min-h-8 items-center gap-1 rounded-md px-1 text-xs font-medium text-(--ui-accent) outline-none hover:underline focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
      onClick={onClick}
      type="button"
    >
      {label}
      <ArrowUpRight className="size-3.5" />
    </button>
  )
}

// ── Model i tryb ──────────────────────────────────────────────────────────

export interface JarvisModelCardProps {
  connected: boolean
  onSelectModel?: (selection: { model: string; provider: string; sessionId?: null | string }) => Promise<boolean> | void
  providers?: readonly ModelOptionProvider[]
  requestGateway?: GatewayRequest
}

function shortModel(model: string): string {
  return model.includes('/') ? model.slice(model.indexOf('/') + 1) : model
}

export function JarvisModelCard({ connected, onSelectModel, providers, requestGateway }: JarvisModelCardProps) {
  const { t, locale } = useI18n()
  const copy = t.jarvisShell.home.model
  const navigate = useNavigate()
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const currentEffort = useStore($currentReasoningEffort)
  const defaultEffort = useStore($defaultReasoningEffort)
  const activeSessionId = useStore($activeSessionId)
  const [pending, setPending] = useState<null | string>(null)
  const queryClient = useQueryClient()
  const openRouter = useMemo(() => resolveOpenRouterPresets(providers), [providers])
  const mode = workModeForEffort(currentEffort || defaultEffort || 'medium')

  const setMode = async (next: JarvisWorkModeId) => {
    const effort = JARVIS_WORK_MODES.find(item => item.id === next)?.effort

    if (!effort || next === mode) {
      return
    }

    const previous = currentEffort
    markComposerSelectionManual()
    setCurrentReasoningEffort(effort)

    // No live session yet: the composer carries the level into the first turn.
    if (!activeSessionId || !requestGateway) {
      return
    }

    try {
      await requestGateway('config.set', { key: 'reasoning', session_id: activeSessionId, value: effort })
    } catch (error) {
      setCurrentReasoningEffort(previous)
      notifyError(error, t.shell.modelOptions.updateFailed)
    }
  }

  const pick = async (model: string) => {
    if (!onSelectModel || pending) {
      return
    }

    setPending(model)

    try {
      await onSelectModel({ model, provider: OPENROUTER_PROVIDER_SLUG, sessionId: activeSessionId })
    } finally {
      setPending(null)
    }
  }

  const presetLabel = (id: OpenRouterPresetId) => copy.presets[id]

  return (
    <RailCard
      action={<LinkAction label={copy.manage} onClick={() => navigate(`${SETTINGS_ROUTE}?tab=providers`)} />}
      icon={Cpu}
      testId="model"
      title={copy.title}
    >
      <LiveModelPicker connected={connected} />
      <p className="mb-2 text-xs font-medium text-(--ui-text-secondary)">{t.jarvisShell.home.model.modeLabel} · Hermes</p>
      <div className="jarvis-well mb-3 flex min-w-0 items-center gap-3 px-3 py-2.5">
        <Brain className="size-4 shrink-0 text-(--ui-accent)" />
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-(--ui-text-primary)" title={currentModel}>
            {currentModel ? shortModel(currentModel) : copy.noModel}
          </p>
          <p className="truncate text-xs text-(--ui-text-secondary)">{currentProvider || copy.noProvider}</p>
        </div>
      </div>

      <div
        aria-label={copy.modeLabel}
        className="jarvis-well mb-3 grid grid-cols-3 gap-1 p-1"
        role="radiogroup"
      >
        {JARVIS_WORK_MODES.map(item => (
          <button
            aria-checked={mode === item.id}
            className={cn(
              'min-h-11 min-w-0 rounded-xl px-1 text-[0.7rem] leading-tight font-medium outline-none transition-colors focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)',
              mode === item.id ? 'jarvis-segment-on' : 'text-(--ui-text-secondary) hover:text-(--ui-text-primary)'
            )}
            disabled={!connected}
            key={item.id}
            onClick={() => void setMode(item.id)}
            role="radio"
            title={copy.modeHints[item.id]}
            type="button"
          >
            {copy.modes[item.id]}
          </button>
        ))}
      </div>

      <p className="mb-2 text-xs font-medium uppercase tracking-[0.16em] text-(--ui-text-tertiary)">
        {copy.openRouter}
      </p>
      {openRouter.connected ? (
        openRouter.presets.length > 0 ? (
          <ul className="grid min-w-0 gap-2">
            {openRouter.presets.map(preset => {
              const active = currentProvider === OPENROUTER_PROVIDER_SLUG && currentModel === preset.model

              return (
                <li className="min-w-0" key={preset.id}>
                  <button
                    aria-pressed={active}
                    className={cn(
                      'flex min-h-11 w-full min-w-0 items-center gap-2 rounded-xl px-3 py-2 text-left text-xs outline-none transition-colors focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)',
                      active ? 'bg-(--ui-accent)/14 text-(--ui-text-primary)' : 'jarvis-well hover:bg-(--chrome-action-hover)'
                    )}
                    disabled={!connected || !onSelectModel || pending !== null}
                    onClick={() => void pick(preset.model)}
                    title={preset.model}
                    type="button"
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block whitespace-normal leading-snug font-medium">{presetLabel(preset.id)}</span>
                      <span className="block truncate text-xs text-(--ui-text-secondary)">
                        {shortModel(preset.model)}
                      </span>
                    </span>
                    {pending === preset.model ? (
                      <Loader2 className="size-4 animate-spin text-(--ui-accent)" />
                    ) : active ? (
                      <Check className="size-4 text-(--ui-accent)" />
                    ) : null}
                  </button>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="text-xs text-(--ui-text-secondary)">{copy.noPresets}</p>
        )
      ) : (
        <OpenRouterQuickConnect
          onConnected={result => {
            void queryClient.invalidateQueries({ queryKey: ['model-options'] })
            notify({
              kind: 'success',
              message: result.model
                ? t.jarvisShell.openRouterConnect.connected(shortModel(result.model))
                : t.jarvisShell.openRouterConnect.connectedNoModel
            })

            if (result.model) {
              void pick(result.model)
            }
          }}
        />
      )}
    </RailCard>
  )
}

// ── AI News Live ──────────────────────────────────────────────────────────

const NEWS_REFRESH_MS = 15 * 60_000
const NEWS_LIMIT = 5

function relativeAge(publishedSeconds: null | number, locale: string, nowMs = Date.now()): string {
  if (!publishedSeconds) {
    return ''
  }

  const minutes = Math.round((publishedSeconds * 1000 - nowMs) / 60_000)
  const format = new Intl.RelativeTimeFormat(locale, { numeric: 'auto', style: 'short' })

  if (Math.abs(minutes) < 60) {
    return format.format(minutes, 'minute')
  }

  if (Math.abs(minutes) < 60 * 24) {
    return format.format(Math.round(minutes / 60), 'hour')
  }

  return format.format(Math.round(minutes / (60 * 24)), 'day')
}

export function JarvisNewsLiveCard({ connected }: { connected: boolean }) {
  const { locale, t } = useI18n()
  const copy = t.jarvisShell.home.news

  const news = useQuery({
    enabled: connected,
    queryFn: () => getAiNews(NEWS_LIMIT),
    queryKey: ['jarvis-ai-news', NEWS_LIMIT],
    refetchInterval: NEWS_REFRESH_MS,
    staleTime: NEWS_REFRESH_MS
  })

  const items = news.data?.items ?? []

  const open = (url: string) => {
    void window.hermesDesktop?.openExternal?.(url)
  }

  return (
    <RailCard
      action={
        <Button
          aria-label={copy.refresh}
          className="min-h-8 min-w-8"
          disabled={!connected || news.isFetching}
          onClick={() => void news.refetch()}
          size="icon"
          type="button"
          variant="ghost"
        >
          <RefreshCw className={cn('size-3.5', news.isFetching && 'animate-spin')} />
        </Button>
      }
      icon={Zap}
      testId="news"
      title={copy.title}
    >
      {!connected ? (
        <p className="text-xs text-(--ui-text-secondary)">{copy.offline}</p>
      ) : news.isPending ? (
        <p className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
          <Loader2 className="size-3.5 animate-spin" />
          {copy.loading}
        </p>
      ) : news.isError ? (
        <p className="text-xs text-(--ui-text-secondary)" role="status">
          {copy.error}
        </p>
      ) : items.length === 0 ? (
        <p className="text-xs text-(--ui-text-secondary)">{copy.empty}</p>
      ) : (
        <ul className="grid gap-1">
          {items.map((item, index) => (
            <li key={item.link}>
              <button
                className="group flex min-h-11 w-full items-start gap-3 rounded-md px-1 py-2 text-left outline-none hover:bg-(--chrome-action-hover) focus-visible:outline focus-visible:outline-solid focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
                onClick={() => open(item.link)}
                title={item.summary || item.title}
                type="button"
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    'mt-1.5 size-2 shrink-0 rounded-full',
                    index === 0 ? 'bg-emerald-400 shadow-[0_0_8px_rgb(52_211_153)]' : 'bg-(--ui-accent)/70'
                  )}
                />
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-2 text-sm leading-5 text-(--ui-text-primary) group-hover:text-(--ui-accent)">
                    {item.title}
                  </span>
                  <span className="mt-0.5 flex gap-2 text-xs text-(--ui-text-tertiary)">
                    <span className="truncate">{item.source}</span>
                    {item.published ? <span className="shrink-0">{relativeAge(item.published, locale)}</span> : null}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </RailCard>
  )
}

// ── Agenci ────────────────────────────────────────────────────────────────

const AGENTS_SHOWN = 5

export function JarvisAgentsCard() {
  const { t } = useI18n()
  const copy = t.jarvisShell.home.agents
  const navigate = useNavigate()
  const profiles = useStore($profiles)
  const activeProfile = useStore($activeGatewayProfile)
  const shown = profiles.slice(0, AGENTS_SHOWN)

  return (
    <RailCard
      action={<LinkAction label={copy.manage} onClick={() => navigate(PROFILES_ROUTE)} />}
      icon={Users}
      testId="agents"
      title={copy.title}
    >
      {shown.length === 0 ? (
        <p className="text-xs text-(--ui-text-secondary)">{copy.empty}</p>
      ) : (
        <ul className="grid gap-1">
          {shown.map(profile => {
            const active = profile.name === activeProfile

            return (
              <li className="flex min-h-10 items-center gap-3 rounded-md px-1" key={profile.name}>
                <span
                  aria-hidden="true"
                  className={cn('size-2 shrink-0 rounded-full', active ? 'bg-emerald-400' : 'bg-(--ui-text-tertiary)')}
                />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-(--ui-text-primary)">{profileLabel(profile)}</span>
                  <span className="block truncate text-xs text-(--ui-text-tertiary)">
                    {profile.model ? shortModel(profile.model) : copy.defaultModel}
                  </span>
                </span>
                <span className={cn('shrink-0 text-xs', active ? 'text-emerald-400' : 'text-(--ui-text-tertiary)')}>
                  {active ? copy.active : copy.ready}
                </span>
              </li>
            )
          })}
        </ul>
      )}
      {profiles.length > AGENTS_SHOWN ? (
        <p className="mt-2 text-xs text-(--ui-text-tertiary)">{copy.more(profiles.length - AGENTS_SHOWN)}</p>
      ) : null}
    </RailCard>
  )
}
