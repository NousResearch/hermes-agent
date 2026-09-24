import type {
  BotCatalogEntryResult,
  BotRequirementReadiness,
  BotRoutineListItem,
  BotsCatalogResult,
  BotsInstalledResult,
  BotsRoutinesListResult,
  BotsStatusResult,
  InstalledBotResult
} from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $botMarketplaceRequest,
  closeBotMarketplaceRequest,
  openBotMarketplaceRequest
} from '@/store/bot-marketplace'

import type { BotInstallReceipt, InstalledBot, MarketplaceNotify, MarketplaceRequest } from './bot-types'
import { scopeProfile, targetBotScope } from './bot-types'
import { $foregroundBotProfile } from './marketplace-actions'
import { ReadinessPanel } from './readiness-panel'
import { RoutinesPanel } from './routines-panel'

export const BOT_PICKER_URL = 'https://hermes-agent.nousresearch.com/docs/bots?embed=picker'
export const BOT_PICKER_ORIGIN = 'https://hermes-agent.nousresearch.com'

export interface BotMarketplaceTabProps {
  scope: ProfileScope
  request: MarketplaceRequest
  onKickoff: (name: string, scope: ProfileScope, kickoffPrompt?: string) => Promise<unknown>
  onOpen: (name: string, scope: ProfileScope) => Promise<unknown>
  onSetupAction: (action: string, id: string, bot: InstalledBot) => void
  invalidateRoster: () => Promise<unknown> | unknown
  notify: MarketplaceNotify
}

interface PickerMessageLike {
  origin: string
  source: MessageEventSource | null
  data: unknown
}

const emptyInstalledStatus = (bot: InstalledBotResult): BotsStatusResult => ({
  profile: bot.profile,
  catalog_name: bot.catalog_name,
  setup_state: bot.setup_state,
  requirements: [],
  runtime: { ok: false, error: '' },
  first_task: { status: 'pending' },
  can_start_first_task: false,
  can_activate_routines: false
})

function installedBotFromMetadata(
  bot: InstalledBotResult,
  liveEntry: BotCatalogEntryResult | undefined,
  scope: ProfileScope
): InstalledBot {
  const entry = liveEntry ?? {
    name: bot.catalog_name,
    version: '',
    maintainer: '',
    tier: 'community',
    category: '',
    tags: [],
    title: bot.title,
    summary: bot.summary,
    profile: {
      suggested_name: bot.profile,
      description: bot.summary,
      soul: '',
      starter_prompt: ''
    },
    capabilities: { skills: [], toolsets: [] },
    setup: { requirements: [] },
    routines: [],
    presentation: bot.presentation
  } satisfies BotCatalogEntryResult

  return {
    entry,
    profile: { name: bot.profile, path: '' },
    scope: targetBotScope(scope, bot.profile),
    status: emptyInstalledStatus(bot),
    routines: []
  }
}

/** Only a catalog key crosses the public iframe boundary. */
export function isTrustedBotPickerMessage(event: PickerMessageLike, pickerWindow: Window | null): string | null {
  if (event.origin !== BOT_PICKER_ORIGIN || !pickerWindow || event.source !== pickerWindow) {
    return null
  }

  const data = event.data

  if (!data || typeof data !== 'object') {
    return null
  }

  const record = data as Record<string, unknown>

  if (record.type !== 'hermes-bot-pick' || typeof record.catalog !== 'string') {
    return null
  }

  return record.catalog.trim() || null
}

const wait = (ms: number, signal?: AbortSignal) => new Promise<void>(resolve => {
  const timer = window.setTimeout(resolve, ms)
  signal?.addEventListener('abort', () => {
    window.clearTimeout(timer)
    resolve()
  }, { once: true })
})

export function BotMarketplaceTab({
  scope,
  request,
  onKickoff,
  onOpen,
  onSetupAction,
  invalidateRoster,
  notify
}: BotMarketplaceTabProps) {
  const { t } = useI18n()
  const m = t.skills.marketplace
  const pendingRequest = useStore($botMarketplaceRequest)
  const foregroundBotProfile = useStore($foregroundBotProfile)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const nativeCatalogRef = useRef<HTMLElement>(null)
  const mounted = useRef(true)
  const submitFlight = useRef<Promise<void> | null>(null)
  const pollController = useRef<AbortController | null>(null)
  const refreshGeneration = useRef(0)
  const [catalog, setCatalog] = useState<BotCatalogEntryResult[]>([])
  const [installed, setInstalled] = useState<InstalledBot[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null)
  const [reconciledForegroundProfile, setReconciledForegroundProfile] = useState('')
  const [entry, setEntry] = useState<BotCatalogEntryResult | null>(null)
  const [name, setName] = useState('')
  const [credentials, setCredentials] = useState<'copy_api_keys' | 'none'>('none')
  const [query, setQuery] = useState('')
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [installing, setInstalling] = useState(false)
  const [refreshing, setRefreshing] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const updateInstalled = useCallback((next: InstalledBot) => {
    setInstalled(current => {
      const existing = current.findIndex(bot => bot.profile.name === next.profile.name)

      if (existing < 0) {
        return [...current, next]
      }

      const copy = [...current]
      copy[existing] = next

      return copy
    })
  }, [])

  const loadMarketplace = useCallback(async () => {
    setCatalogLoading(true)
    setError(null)

    try {
      const profile = scopeProfile(scope)

      const [catalogResult, installedResult] = await Promise.all([
        request<BotsCatalogResult>('bots.catalog', { profile }, scope),
        request<BotsInstalledResult>('bots.installed', { profile }, scope)
      ])

      const entries = catalogResult.entries ?? []

      if (!mounted.current) {
        return
      }

      setCatalog(entries)
      setInstalled((installedResult.bots ?? []).map(bot =>
        installedBotFromMetadata(bot, entries.find(candidate => candidate.name === bot.catalog_name), scope)
      ))
    } catch (cause) {
      if (mounted.current) {
        setError(cause instanceof Error ? cause.message : m.catalogFailed)
      }
    } finally {
      if (mounted.current) {
        setCatalogLoading(false)
      }
    }
  }, [m.catalogFailed, request, scope])

  // These refs are lifecycle tokens, not mirrors of rendered values.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    mounted.current = true
    pollController.current?.abort()
    void loadMarketplace()

    return () => {
      mounted.current = false
      pollController.current?.abort()
    }
  }, [loadMarketplace])

  const loadReview = useCallback((catalogName: string) => {
    const match = catalog.find(candidate => candidate.name === catalogName)

    if (!match) {
      setError(m.notFound(catalogName))
      setEntry(null)

      return
    }

    setError(null)
    setEntry(match)
    setName(match.profile.suggested_name)
    setCredentials('none')
  }, [catalog, m])

  useEffect(() => {
    if (pendingRequest?.catalog && catalog.length) {
      loadReview(pendingRequest.catalog)
    }
  }, [catalog, loadReview, pendingRequest])

  useEffect(() => {
    const receive = (event: MessageEvent) => {
      const picked = isTrustedBotPickerMessage(event, iframeRef.current?.contentWindow ?? null)

      if (picked) {
        openBotMarketplaceRequest(picked)
      }
    }

    window.addEventListener('message', receive)

    return () => window.removeEventListener('message', receive)
  }, [])

  const closeReview = () => {
    if (installing) {
      return
    }

    closeBotMarketplaceRequest()
    setEntry(null)
    setError(null)
  }

  const validName = /^[a-z0-9][a-z0-9_-]{0,63}$/.test(name.trim())

  const refreshBot = useCallback(async (bot: InstalledBot, setup = true, quiet = false) => {
    const generation = ++refreshGeneration.current

    if (!quiet) {
      setRefreshing(bot.profile.name)
    }

    try {
      const status = await request<BotsStatusResult>(setup ? 'bots.setup' : 'bots.status', { profile: bot.profile.name }, bot.scope)
      let routines = bot.routines

      if (status.first_task.status === 'complete' || status.can_activate_routines) {
        const routineResult = await request<BotsRoutinesListResult>('bots.routines.list', { profile: bot.profile.name }, bot.scope)
        routines = routineResult.routines ?? []
      }

      const next = { ...bot, status, routines }

      if (mounted.current && generation === refreshGeneration.current) {
        updateInstalled(next)
      }

      return next
    } catch (cause) {
      if (!quiet) {
        notify({ kind: 'error', title: m.setupFailed, message: cause instanceof Error ? cause.message : String(cause) })
      }

      return bot
    } finally {
      if (!quiet && generation === refreshGeneration.current) {
        setRefreshing(null)
      }
    }
  }, [m.setupFailed, notify, request, updateInstalled])

  const selectInstalledBot = useCallback(async (bot: InstalledBot) => {
    pollController.current?.abort()
    setSelectedProfile(bot.profile.name)
    await refreshBot(bot, false, true)
  }, [refreshBot])

  // bots.installed is intentionally metadata-only. When a foreground Bot Chat
  // returns to Marketplace, reconcile just that profile so a starter completion
  // is reflected without spawning every installed bot profile.
  useEffect(() => {
    if (catalogLoading || !foregroundBotProfile || reconciledForegroundProfile === foregroundBotProfile) {
      return
    }

    const bot = installed.find(candidate => candidate.profile.name === foregroundBotProfile)

    if (!bot) {
      return
    }

    setReconciledForegroundProfile(foregroundBotProfile)
    setSelectedProfile(foregroundBotProfile)
    void refreshBot(bot, false, true)
  }, [catalogLoading, foregroundBotProfile, installed, reconciledForegroundProfile, refreshBot])

  const install = useCallback(() => {
    if (!entry || !validName || submitFlight.current) {
      return
    }

    const run = (async () => {
      setInstalling(true)
      setError(null)
      const sourceProfile = scopeProfile(scope)

      try {
        const result = await request<BotInstallReceipt>(
          'bots.install',
          {
            profile: sourceProfile,
            catalog_name: entry.name,
            name: name.trim(),
            source_profile: sourceProfile,
            credentials
          },
          scope
        )

        if (!result.committed) {
          setError(m.notCommitted)

          return
        }

        await invalidateRoster()
        const botScope = targetBotScope(scope, result.name)
        const status = await request<BotsStatusResult>('bots.status', { profile: result.name }, botScope)
        const routineResult = await request<BotsRoutinesListResult>('bots.routines.list', { profile: result.name }, botScope)

        const bot: InstalledBot = {
          entry,
          profile: { name: result.name, path: result.path },
          scope: botScope,
          status,
          routines: routineResult.routines ?? []
        }

        updateInstalled(bot)
        setSelectedProfile(result.name)
        closeBotMarketplaceRequest()
        setEntry(null)
        notify({ kind: 'info', title: m.added(result.name), message: m.setupRequired })
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : m.installFailed)
      } finally {
        setInstalling(false)
        submitFlight.current = null
      }
    })()

    submitFlight.current = run
  }, [credentials, entry, invalidateRoster, m, name, notify, request, scope, updateInstalled, validName])

  const startFirstTask = useCallback(async (bot: InstalledBot) => {
    if (pollController.current) {
      return
    }

    const controller = new AbortController()
    pollController.current = controller

    try {
      await onKickoff(bot.profile.name, bot.scope, bot.status.starter_prompt ?? bot.entry.profile.starter_prompt)
      let current = bot

      for (let attempt = 0; attempt < 30 && mounted.current && !controller.signal.aborted; attempt += 1) {
        current = await refreshBot(current, false, true)

        if (current.status.first_task.status === 'complete') {
          notify({ kind: 'success', title: m.added(bot.profile.name), message: m.chatStarted })

          return
        }

        await wait(2_000, controller.signal)
      }

      if (!controller.signal.aborted && mounted.current) {
        notify({ kind: 'warning', title: m.setupFailed, message: m.pollTimedOut })
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        notify({ kind: 'warning', title: m.added(bot.profile.name), message: m.chatWarning(cause instanceof Error ? cause.message : String(cause)) })
      }
    } finally {
      if (pollController.current === controller) {
        pollController.current = null
      }
    }
  }, [m, notify, onKickoff, refreshBot])

  const mutateRoutine = useCallback(async (
    bot: InstalledBot,
    routine: BotRoutineListItem,
    method: 'bots.routines.activate' | 'bots.routines.pause',
    values?: { schedule: string; timezone: string; destination: string }
  ) => {
    try {
      await request(method, {
        profile: bot.profile.name,
        routine_id: routine.id,
        ...(values ?? {})
      }, bot.scope)
      await refreshBot(bot, false)
    } catch (cause) {
      notify({ kind: 'error', title: m.routineFailed, message: cause instanceof Error ? cause.message : String(cause) })
    }
  }, [m.routineFailed, notify, refreshBot, request])

  const filteredCatalog = useMemo(() => {
    const needle = query.trim().toLowerCase()

    if (!needle) {
      return catalog
    }

    return catalog.filter(candidate =>
      [candidate.title, candidate.summary, candidate.category, candidate.maintainer, ...(candidate.tags ?? [])]
        .join(' ')
        .toLowerCase()
        .includes(needle)
    )
  }, [catalog, query])

  const selectedBot = installed.find(bot => bot.profile.name === selectedProfile) ?? null

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <section aria-label={m.nativeCatalog} className="space-y-3 px-4 py-4" ref={nativeCatalogRef}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-(--ui-text-primary)">{m.title}</div>
            <div className="mt-1 text-xs text-(--ui-text-secondary)">{m.description}</div>
          </div>
          {catalog.length > 0 && <SearchField aria-label={m.search} onChange={setQuery} placeholder={m.search} value={query} />}
        </div>

        {catalogLoading && <div className="text-xs text-(--ui-text-secondary)">{m.loading}</div>}
        {error && !entry && <div className="text-xs text-(--ui-destructive)">{error}</div>}
        <div className="space-y-1">
          {filteredCatalog.map(candidate => (
            <div className="flex items-center gap-3 py-2" key={candidate.name}>
              <span aria-hidden className="text-lg">{candidate.presentation.emoji}</span>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-(--ui-text-primary)">{candidate.title}</div>
                <div className="truncate text-xs text-(--ui-text-secondary)">{candidate.summary}</div>
                <div className="text-[0.7rem] text-(--ui-text-tertiary)">{m.creator(candidate.maintainer)}</div>
              </div>
              <Button aria-label={`${m.add} ${candidate.title}`} onClick={() => openBotMarketplaceRequest(candidate.name)} size="xs" variant="secondary">
                {m.add}
              </Button>
            </div>
          ))}
        </div>
      </section>

      <section aria-label={m.installed} className="border-t border-(--ui-stroke-tertiary) px-4 py-4">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs font-semibold text-(--ui-text-primary)">{m.installed}</div>
          <Button onClick={() => nativeCatalogRef.current?.scrollIntoView({ behavior: 'smooth' })} size="xs" variant="text">
            {m.addAnother}
          </Button>
        </div>
        {installed.length === 0 ? (
          <div className="mt-2 text-xs text-(--ui-text-tertiary)">{m.noInstalled}</div>
        ) : (
          <div className="mt-2 space-y-1">
            {installed.map(bot => (
              <div className="flex items-center gap-3 py-2" key={bot.profile.name}>
                <span aria-hidden>{bot.entry.presentation.emoji}</span>
                <button className="min-w-0 flex-1 text-left" onClick={() => void selectInstalledBot(bot)} type="button">
                  <span className="block text-xs font-medium text-(--ui-text-primary)">{bot.profile.name}</span>
                  <span className="block text-xs text-(--ui-text-secondary)">{bot.entry.title}</span>
                  {!catalog.some(candidate => candidate.name === bot.status.catalog_name) && (
                    <span className="block text-[0.7rem] text-(--ui-warning)">{m.catalogUnavailable}</span>
                  )}
                </button>
                {bot.status.setup_state === 'ready' ? (
                  <Button aria-label={`${m.open} ${bot.profile.name}`} onClick={() => void onOpen(bot.profile.name, bot.scope)} size="xs" variant="secondary">{m.open}</Button>
                ) : (
                  <Button aria-label={`${m.finishSetup} ${bot.profile.name}`} onClick={() => void selectInstalledBot(bot)} size="xs" variant="secondary">{m.finishSetup}</Button>
                )}
              </div>
            ))}
          </div>
        )}

        {selectedBot && (
          <div className="mt-4 space-y-4 border-t border-(--ui-stroke-tertiary) pt-4">
            <ReadinessPanel
              bot={selectedBot}
              busy={refreshing === selectedBot.profile.name}
              copy={m.readiness}
              onOpen={() => void onOpen(selectedBot.profile.name, selectedBot.scope)}
              onRefresh={() => {
                pollController.current?.abort()
                void refreshBot(selectedBot)
              }}
              onSetupAction={(requirement: BotRequirementReadiness | { action: string; id: string }) =>
                onSetupAction(requirement.action ?? ('kind' in requirement ? requirement.kind : 'model'), requirement.id, selectedBot)
              }
              onStartFirstTask={() => void startFirstTask(selectedBot)}
            />
            <RoutinesPanel
              canActivate={selectedBot.status.can_activate_routines}
              copy={m.routines}
              onActivate={(routine, values) => mutateRoutine(selectedBot, routine, 'bots.routines.activate', values)}
              onPause={routine => mutateRoutine(selectedBot, routine, 'bots.routines.pause')}
              routines={selectedBot.routines}
            />
          </div>
        )}
      </section>

      <section aria-label={m.siteCatalog} className="min-h-72 border-t border-(--ui-stroke-tertiary)">
        <iframe
          className="h-96 w-full border-0 bg-transparent"
          ref={iframeRef}
          sandbox="allow-scripts allow-same-origin"
          src={BOT_PICKER_URL}
          title={m.iframeTitle}
        />
      </section>

      <Dialog onOpenChange={next => !next && closeReview()} open={Boolean(pendingRequest)}>
        <DialogContent className="max-h-[min(52rem,92vh)] max-w-2xl" onOpenAutoFocus={preventCloseButtonAutoFocus}>
          <DialogHeader>
            <DialogTitle>{entry ? entry.title : m.reviewTitle}</DialogTitle>
            <DialogDescription>{entry?.summary ?? m.reviewDescription}</DialogDescription>
          </DialogHeader>

          {entry && (
            <div className="min-h-0 space-y-5 overflow-y-auto pr-1">
              <div className="space-y-1 text-xs text-(--ui-text-secondary)">
                <div>{m.creator(entry.maintainer)}</div>
                <div>{entry.profile.description}</div>
                <div>{m.destination(scopeProfile(scope))}</div>
                <div>{m.freshCopy}</div>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="space-y-1.5 text-xs font-medium text-(--ui-text-secondary)">
                  <span>{m.nameLabel}</span>
                  <Input aria-label={m.nameLabel} onChange={event => setName(event.target.value)} value={name} />
                  {!validName && <span className="text-(--ui-destructive)">{m.nameInvalid}</span>}
                </label>
                <label className="space-y-1.5 text-xs font-medium text-(--ui-text-secondary)">
                  <span>{m.credentialsLabel}</span>
                  <Select onValueChange={value => setCredentials(value as typeof credentials)} value={credentials}>
                    <SelectTrigger aria-label={m.credentialsLabel}><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">{m.credentialsNone}</SelectItem>
                      <SelectItem value="copy_api_keys">{m.credentialsCopy}</SelectItem>
                    </SelectContent>
                  </Select>
                </label>
              </div>
              <div className="space-y-2 text-xs">
                <div className="font-medium text-(--ui-text-secondary)">{m.capabilities}</div>
                <div>{[...(entry.capabilities.skills ?? []), ...(entry.capabilities.toolsets ?? [])].join(' · ') || m.none}</div>
              </div>
              <div className="space-y-2 text-xs">
                <div className="font-medium text-(--ui-text-secondary)">{m.starterPrompt}</div>
                <div>{entry.profile.starter_prompt}</div>
              </div>
              <div className="space-y-2 text-xs">
                <div className="font-medium text-(--ui-text-secondary)">{m.contract}</div>
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap border-l border-(--ui-stroke-tertiary) pl-3 leading-5 text-(--ui-text-secondary)">{entry.profile.soul}</pre>
              </div>
            </div>
          )}

          {error && <div className={cn('text-sm text-(--ui-destructive)')} role="alert">{error}</div>}
          <DialogFooter>
            <Button disabled={installing} onClick={closeReview} variant="text">{error && !entry ? t.common.close : t.common.cancel}</Button>
            {error && !entry ? (
              <Button onClick={() => void loadMarketplace()}>{m.retry}</Button>
            ) : (
              <Button disabled={!entry || !validName || installing} onClick={install}>{installing ? m.adding : m.add}</Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
