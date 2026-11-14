/**
 * Screen portal — the compact "this bot's computer" box in the classic
 * Sessions sidebar. The redesigned sidebar uses the Screen palette action.
 * One click opens the live Screen pane; the subtitle says whether the screen
 * is live and who holds it, so the user knows before opening whether they
 * are about to watch, take over, install or start.
 *
 * Reads the same per-bot cache the pane paints (`$screenState`) and refreshes
 * it once on mount so a bot the user never opened still shows real state.
 */

import { Codicon, host, useValue } from '@hermes/plugin-sdk'
import type { RpcEvent } from '@hermes/plugin-sdk'
import type { PaletteContribution, ProfileGroupRoute } from '@hermes/plugin-sdk'
import { useEffect, useMemo } from 'react'

import { $lastRoster } from './data'
import { botsText, useBots } from './i18n'
import { resolveBotConnectionRoute } from './routing'
import {
  type DisplayLease,
  displayRequest,
  type DisplayStatus,
  isDisplayUnavailable,
  isEventForBotScreen,
  isManagedBackend,
  leaseHeldBy,
  type ScreenViewer
} from './screen-connection'
import { openBotScreen } from './screen-open'
import {
  $screenState,
  beginScreenStatusRequest,
  screenStateFor,
  setScreenLease,
  setScreenStatus,
  setScreenUnavailable
} from './screen-state'
import type { RosterRow } from './types'

export type PortalTone = 'live' | 'human' | 'other' | 'off' | 'missing' | 'unsupported' | 'unavailable' | 'unknown'

/** Pure: map cached status + lease (+ this window's minted viewer, if attached) to what the portal says. */
export function portalTone(
  status: DisplayStatus | null,
  lease: DisplayLease | null,
  viewer: ScreenViewer | null = null,
  unavailable = false
): PortalTone {
  if (unavailable) {
    return 'unavailable'
  }

  if (!status) {
    return 'unknown'
  }

  if (!status.supported) {
    return 'unsupported'
  }

  if (!status.installed) {
    return 'missing'
  }

  if (!status.running) {
    return 'off'
  }

  if (lease?.holder === 'human') {
    return leaseHeldBy(lease, viewer) ? 'human' : 'other'
  }

  return 'live'
}

const TONE_ICON: Record<PortalTone, string> = {
  live: 'device-desktop',
  human: 'record-keys',
  other: 'eye',
  off: 'debug-stop',
  missing: 'cloud-download',
  unsupported: 'circle-slash',
  unavailable: 'circle-slash',
  unknown: 'device-desktop'
}

const TONE_DOT: Record<PortalTone, string> = {
  live: 'bg-emerald-500',
  human: 'bg-red-500',
  other: 'bg-amber-500',
  off: 'bg-(--ui-text-quaternary)',
  missing: 'bg-(--ui-text-quaternary)',
  unsupported: 'bg-(--ui-text-quaternary)',
  unavailable: 'bg-(--ui-text-quaternary)',
  unknown: 'bg-(--ui-text-quaternary)'
}

export function useScreenPortalState(bot: RosterRow) {
  const all = useValue($screenState)
  const state = screenStateFor(all, bot)
  const status = state?.status ?? null
  const profileKey = status?.profile_key

  useEffect(() => {
    if (status || state?.unavailable) {
      return
    }

    let cancelled = false
    // A network reply, not an event: hand the token back so a snapshot that a newer request or a
    // pushed status superseded while in flight is dropped instead of landing as authoritative.
    const request = beginScreenStatusRequest(bot)

    void displayRequest<DisplayStatus>(bot, 'display.status')
      .then(next => {
        if (!cancelled) {
          setScreenStatus(bot, next, request)
        }
      })
      .catch((error: unknown) => {
        // An older Hermes without display.* is a settled answer (hide the surface);
        // an offline bot is transient and stays in its unknown state.
        if (!cancelled && isDisplayUnavailable(error)) {
          setScreenUnavailable(bot)
        }
      })

    return () => {
      cancelled = true
    }
  }, [bot, state?.unavailable, status])

  useEffect(
    () =>
      host.onEvent('display.lease', (event: RpcEvent) => {
        const payload = event.payload as { profile_key?: string; lease?: DisplayLease } | undefined

        if (payload?.lease && isEventForBotScreen(bot, event, profileKey)) {
          setScreenLease(bot, payload.lease)
        }
      }),
    [bot, profileKey]
  )

  // A start/stop/crash made outside this window (CLI, gateway auto-start, another Desktop) is
  // pushed by the serve-side runtime watcher; without it the portal's status was one-shot.
  useEffect(
    () =>
      host.onEvent('display.status', (event: RpcEvent) => {
        const payload = event.payload as DisplayStatus | undefined

        if (payload?.profile_key && isEventForBotScreen(bot, event, profileKey)) {
          setScreenStatus(bot, payload)
        }
      }),
    [bot, profileKey]
  )

  return {
    status,
    lease: state?.lease ?? null,
    tone: portalTone(status, state?.lease ?? null, state?.viewer ?? null, state?.unavailable)
  }
}

export function ScreenPortal({ bot }: { bot: RosterRow }) {
  const t = useBots()
  const { status, tone } = useScreenPortalState(bot)

  const subtitle = {
    live: t.screen.portalWatching,
    human: t.screen.portalYouControl,
    other: t.screen.portalOtherControls,
    off: t.screen.portalStopped,
    missing: t.screen.portalNotInstalled,
    unsupported: t.screen.portalUnsupported,
    unavailable: isManagedBackend(bot) ? t.screen.portalUnavailableManaged : t.screen.portalUnavailable,
    unknown: status?.display ?? ''
  }[tone]

  // Nothing to offer for a screen that cannot exist: the sidebar row simply disappears.
  if (tone === 'unsupported' || tone === 'unavailable') {
    return null
  }

  return (
    <button
      aria-label={`${t.screen.portalTitle}: ${subtitle}`}
      className="group flex w-full items-center gap-2 rounded-md border border-(--ui-stroke-secondary) bg-(--chrome-action-hover)/40 px-2 py-1.5 text-left transition-colors hover:bg-(--chrome-action-hover)"
      onClick={() => openBotScreen(bot, null)}
      type="button"
    >
      <span className="relative grid size-7 shrink-0 place-items-center rounded bg-black/70 text-white/90">
        <Codicon name={TONE_ICON[tone]} />
        <span
          className={`absolute -right-0.5 -top-0.5 size-2 rounded-full ring-2 ring-(--ui-bg-primary) ${TONE_DOT[tone]}`}
        />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs font-medium">{t.screen.portalTitle}</span>
        {subtitle ? <span className="block truncate text-[0.65rem] text-(--ui-text-tertiary)">{subtitle}</span> : null}
      </span>
      <span className="flex shrink-0 items-center gap-1 text-[0.65rem] text-(--ui-text-tertiary) opacity-0 transition-opacity group-hover:opacity-100">
        {t.screen.portalOpen} <Codicon name="arrow-right" />
      </span>
    </button>
  )
}

/** Resolve the same owner for classic sidebar cards and the Screen tool action. */
function screenBotForRoute({ connectionId, profile }: ProfileGroupRoute, roster: RosterRow[]): RosterRow {
  return (
    roster.find(row => {
      const resolved = resolveBotConnectionRoute(row)

      return resolved.route
        ? (resolved.route.profile === profile || resolved.route.targetProfile === profile) &&
          resolved.route.connectionId === (connectionId ?? 'local')
        : row.name === profile && connectionId === null
    }) ??
    (connectionId
      ? ({
          name: profile,
          sourceScoped: true,
          connectionId,
          connectionKind: connectionId === 'local' ? 'local' : 'remote'
        } as RosterRow)
      : ({ name: profile } as RosterRow))
  )
}

/** Explicit tool access follows the focused chat's complete owner, never the live socket alone. */
export function profileScreenCommand(route: ProfileGroupRoute): Omit<PaletteContribution, 'id'> {
  return {
    label: botsText().screen.menu,
    keywords: ['screen', 'computer', 'display', 'desktop'],
    detail: () => `${route.profile} · ${route.connectionId ?? 'local'}`,
    run: async () => {
      const bot = screenBotForRoute(route, $lastRoster.get())

      const target = (await host.profileRoutes()).find(candidate =>
        candidate.connectionId === (route.connectionId ?? 'local') && candidate.profile === route.profile
      )

      openBotScreen(target ? { ...bot, targetProfile: target.targetProfile || target.profile } : bot, null)
    }
  }
}

/** A filtered-out profile still gets a scoped row: Screen only needs a routable owner. */
export function ProfileGroupScreenPortal({ route }: { route: ProfileGroupRoute }) {
  const roster = useValue($lastRoster)
  const { connectionId, profile } = route
  // Keep effects stable when a parent re-renders a synthesized, non-roster owner.
  const bot = useMemo(() => screenBotForRoute({ connectionId, profile }, roster), [connectionId, profile, roster])

  return <ScreenPortal bot={bot} />
}
