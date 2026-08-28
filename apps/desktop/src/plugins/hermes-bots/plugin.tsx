/**
 * Hermes Bot Mode — a "one chat per agent" roster for the Hermes desktop.
 *
 * Left pane "Bots": one row per Hermes profile (a bot = an agent profile) with
 * a customizable avatar (shape + color + eyes, image, or pet). Click opens that
 * bot's chat; right-click → Edit Profile (avatar, title, description).
 * "New Bot" creates a profile — Name / Title / Description with an
 * "Advanced" disclosure for full profile config.
 *
 * Right tile "Routines": scheduled tasks (Hermes cron jobs) scoped to the
 * bot you're currently chatting with — follows the live gateway profile.
 *
 * Bots message each other straight into each bot's ONE canonical "Bot
 * Chat" — @-mentions deliver over gateway RPCs (no CLI relay), and
 * bot-initiated sends use `hermes -p <bot> chat --in ~ -c "Bot Chat"`.
 */

import { CHAT_EMPTY_AREA, COMPOSER_AREAS, EmptyState, host, PALETTE_AREA, translateNow } from '@hermes/plugin-sdk'
import type { ChatEmptyProps, PluginContext } from '@hermes/plugin-sdk'
import type { ChangeEvent, KeyboardEvent } from 'react'

import { startFaceClock, stopFaceClock } from './avatar'
import { BotChatEmpty } from './chat-empty'
import {
  $botChatFocused,
  $botsPaneVisible,
  $focusedBotOwner,
  $openBotChat,
  $selectedBot,
  $selectedRosterHydrated,
  $selectedRosterKey,
  focusedMentionProfile
} from './bot-state'
import { openBotCanonicalChat } from './canonical-chat'
import { bindProfileSync, RoutinesPane } from './cron'
import {
  $botMeta,
  $lastRoster,
  botHandle,
  botMentionTag,
  cachedUnionRoster,
  isActiveRosterBot,
  migrateBotMeta,
  resolveRosterMentions
} from './data'
import {
  $groupChats,
  $groupChatWorkspace,
  assignLegacyThreads,
  handleSessionsGatewayTransition,
  pullGroupChatServerState,
  scheduleGroupChatServerSync,
  setGroupChatSyncDisposed,
  stopGroupChatServerSync,
  sweepGroupChatMembersForRemovedConnection,
  updateGroupChat
} from './group-chat'
import { groupWorkspaceOwnerKey } from './group-membership'
import { annotateOrphanedGroupChatMembers } from './hygiene'
import { BOTS_LOCALES } from './i18n'
import { displayName } from './labels'
import { startBotRelay, stopBotRelay } from './relay'
import { $activityToasts } from './roster-actions'
import {
  botChatOwnsWorkspace,
  BotsPane,
  releaseStaleOpenBotChat,
  selectedRosterBot,
  sessionOwnsWorkspace
} from './roster-pane'
import { botRosterMeta, botWorkspaceOwnerKey, setBotsWorkspaceOwner } from './routing'
import { startHideSweepScheduler } from './session-sweep'
import { bumpBotOpenGeneration, getBotOpenGeneration, ID, setPluginCtx } from './shared'
import type { GroupChat, GroupPromptKind, RosterRow } from './types'

// ── plugin ───────────────────────────────────────────────────────────────────

/** One row the composer's `@` popover renders from the roster. */
interface MentionCompletionItem {
  display: string
  insert: string
  meta: string
}

/** The draft a `composer.middleware` handler rewrites, passes through, or
 *  cancels (this plugin never cancels). */
interface ComposerDraftPayload {
  attachments?: unknown[]
  text: string
}

export default {
  id: ID,
  name: 'Bots',
  description:
    'Bot Mode — a one-chat-per-agent roster with avatars, routines, group chats, and bot-to-bot messaging. Ships with the app; disable here if unwanted.',
  register(ctx: PluginContext) {
    setPluginCtx(ctx)
    const disposeLocales = ctx.i18n.register(BOTS_LOCALES)
    setGroupChatSyncDisposed(false)
    startFaceClock()
    // The cross-connection relay rides every gateway socket this Desktop
    // holds: roster sync + envelope drain/deliver/reply loops.
    startBotRelay()
    // Disabling the plugin (or a hot reload) must actually stop the clock —
    // before this, the rAF loop + 1Hz document scan ran until app restart.
    if (typeof ctx.onDispose === 'function') {
      ctx.onDispose(disposeLocales)
      ctx.onDispose(stopFaceClock)
      ctx.onDispose(stopBotRelay)
    }

    // @-mention autocomplete: typing "@rese…" in ANY composer offers the
    // roster's handles (issue #88060). Reads the roster straight from the
    // query cache — useRoster keeps it ≤5s stale and the popover must answer
    // synchronously per keystroke. Multi-source rosters contribute their
    // precomputed @name-device handles via botHandle. The active profile is
    // excluded (a bot doesn't @ itself); 'default' surfaces as @hermes.
    ctx.register({
      id: 'mention-completions',
      area: COMPOSER_AREAS.atCompletions,
      data: {
        provide: (query: string): MentionCompletionItem[] => {
          const roster = cachedUnionRoster()
          const profiles = Array.isArray(roster?.profiles) ? roster.profiles : []
          if (!profiles.length) {
            return []
          }
          const active = focusedMentionProfile()
          const q = (query || '').toLowerCase()
          const items: MentionCompletionItem[] = []
          const live = {
            name: active,
            connectionId: String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
          }
          for (const profile of profiles) {
            if (!profile?.name || isActiveRosterBot(profile, live)) {
              continue
            }
            const handle = botHandle(profile.name, profile)
            const display = displayName(profile, $botMeta.get()[profile.name])
            // Renamed bots complete on their friendly name — the tag is the
            // renamed slug when one exists, the profile handle otherwise.
            const tag = botMentionTag(profile)
            if (
              q &&
              !tag.toLowerCase().startsWith(q) &&
              !handle.toLowerCase().startsWith(q) &&
              !display.toLowerCase().startsWith(q)
            ) {
              continue
            }
            const source = profile.connectionLabel ? ` · ${profile.connectionLabel}` : ''
            items.push({
              insert: `@${tag}`,
              display: `@${tag}`,
              meta: `Bot · ${display}${source}`
            })
          }
          return items.slice(0, 8)
        }
      }
    })

    // Hydrate persisted avatars/titles. Migration writes v2 only after a
    // provable sole-local topology and deliberately leaves v1 untouched for
    // one-version rollback.
    void migrateBotMeta(ctx.storage).catch(() => undefined)

    // The last selected bot, source-qualified. Restoring it is PRESENTATION
    // ONLY: it paints the roster highlight, and never activates a gateway,
    // opens a chat, or creates a session. The hydrated flag must flip on
    // every settle path — the roster holds a loading state until it does,
    // and a storage quirk must not strand it there.
    try {
      // TODO(bot-mode-types): PluginStorage.get(key, fallback) requires the fallback; every
      // Bot Mode read omits it (works at runtime, undefined fallback) — same at the two reads below.
      // @ts-expect-error typed as written rather than changing the call.
      Promise.resolve(ctx.storage?.get?.('selected-roster-bot-v1'))
        .then(value => {
          if (typeof value === 'string' && value.trim()) {
            $selectedRosterKey.set(value.trim())
          }
        })
        .catch(() => undefined)
        .finally(() => $selectedRosterHydrated.set(true))
    } catch {
      /* no storage — this window starts with no restored selection */
      $selectedRosterHydrated.set(true)
    }

    // Bot Mode sessions are always hidden now — the old "hide Bot Chats"
    // pref is gone (its stored key is simply ignored). The reconciliation
    // sweep below hides any rows born visible under the old pref.

    // Hydrate the activity-toast pref (default OFF).
    try {
      // @ts-expect-error TODO(bot-mode-types): PluginStorage.get requires a fallback argument.
      Promise.resolve(ctx.storage?.get?.('activity-toasts'))
        .then(value => {
          if (typeof value === 'boolean') {
            $activityToasts.set(value)
          }
        })
        .catch(() => undefined)
    } catch {
      /* no storage — default (silent) stays */
    }

    // Hydrate persisted group-chat room logs (epoch/running are runtime-only
    // and always reset — a loop can't survive a window reload anyway).
    try {
      // @ts-expect-error TODO(bot-mode-types): PluginStorage.get requires a fallback argument.
      Promise.resolve(ctx.storage?.get?.('group-chats'))
        .then(async value => {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            const rooms: Record<string, GroupChat> = {}
            for (const [name, room] of Object.entries(value)) {
              if (room && Array.isArray(room.log)) {
                rooms[name] = {
                  // Pre-thread entries get synthetic thread ids on hydrate so
                  // every UI/engine path can assume entry.thread exists.
                  log: assignLegacyThreads(room.log),
                  watermarks: room.watermarks && typeof room.watermarks === 'object' ? room.watermarks : {},
                  sessions: room.sessions && typeof room.sessions === 'object' ? room.sessions : {},
                  sessionOwners: room.sessionOwners && typeof room.sessionOwners === 'object' ? room.sessionOwners : {},
                  stranded: room.stranded && typeof room.stranded === 'object' ? room.stranded : {},
                  // #93129: rehydrate sticky stop holds with the same shape
                  // guard as the other maps — a held bot stays held across
                  // window restarts until explicitly released.
                  holds: room.holds && typeof room.holds === 'object' ? room.holds : {},
                  members: Array.isArray(room.members) ? room.members : [],
                  roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
                  image: typeof room.image === 'string' && room.image ? room.image : null,
                  syncRevision: Math.max(0, Number(room.syncRevision || 0)),
                  epoch: 0,
                  running: false
                }
              }
            }
            $groupChats.set({
              ...rooms,
              ...$groupChats.get()
            })

            // #93492: annotate rows orphaned before this build (their
            // connection was deleted while an older Desktop ran, so no
            // 'removed' push ever swept them). Registry read is
            // feature-detected; when unavailable only the unresolvable-route
            // shape (lost connectionId) is annotated.
            try {
              const registry =
                typeof window !== 'undefined'
                  ? await Promise.resolve(window.hermesDesktop?.connections?.list?.()).catch(() => null)
                  : null
              const liveIds = Array.isArray(registry?.connections)
                ? new Set(registry.connections.map(connection => String(connection?.id || '').trim()).filter(Boolean))
                : null
              const annotated = annotateOrphanedGroupChatMembers($groupChats.get(), liveIds)
              if (annotated.changed) {
                // Per-room updateGroupChat keeps the durable record's full
                // shape (sessionOwners, holds) in storage; sync:false —
                // the scheduleGroupChatServerSync below publishes once.
                for (const [roomName, room] of Object.entries(annotated.rooms)) {
                  if (room !== $groupChats.get()[roomName]) {
                    updateGroupChat(roomName, () => room, {
                      sync: false
                    })
                  }
                }
              }
            } catch {
              /* registry unavailable — the lost-connectionId shape is still safe to render */
            }
          }

          // Receive before publish. A fresh Desktop with no local room cache
          // must hydrate the gateway projection instead of merely avoiding an
          // empty overwrite and then rendering an empty conversation.
          await pullGroupChatServerState().catch(() => false)
          scheduleGroupChatServerSync($groupChats.get())
        })
        .catch(() => undefined)
    } catch {
      /* no storage — rooms start empty */
    }

    // Routines follow the chat you're in: track the focused chat's owner
    // profile (falls back to the live gateway profile on older desktops —
    // see $focusedBotProfile). Keying this off the socket's home alone left
    // the unread-suppression and Routines scope on the wrong bot whenever a
    // focused tab showed another profile's chat.
    // Capture the unbinds: without them a disable → re-enable cycle stacks a
    // duplicate listener per cycle (same survives-disable class as the face
    // clock before its onDispose hook — these kept firing until app restart).
    const unbindProfileListener = bindProfileSync($focusedBotOwner)
    const unbindGatewayListener = host.state.gateway.listen(handleSessionsGatewayTransition)

    // #93492 root fix: the registry pushes a lifecycle event when a
    // connection is removed. The gateway store already disposes the dead
    // sockets; the persisted group-chat rosters referencing that connection
    // were never touched, which is what left panes throwing "Bot X has no
    // connection owner" forever. Annotate (never silently delete) those
    // member rows the moment the connection goes away. Feature-detected:
    // older Electron mains don't emit it, and bare vm test harnesses have
    // no window global.
    let unbindConnectionsChanged: null | (() => void) = null
    try {
      if (typeof window !== 'undefined') {
        unbindConnectionsChanged =
          window.hermesDesktop?.connections?.onChanged?.(payload => {
            if (payload?.reason === 'removed') {
              sweepGroupChatMembersForRemovedConnection(payload.connectionId)
            }
          }) || null
      }
    } catch {
      /* registry lifecycle push unavailable — hydrate-time annotate still covers it */
    }
    if (typeof ctx.onDispose === 'function') {
      ctx.onDispose(() => {
        stopGroupChatServerSync()
        if (typeof unbindProfileListener === 'function') {
          unbindProfileListener()
        }
        if (typeof unbindGatewayListener === 'function') {
          unbindGatewayListener()
        }
        if (typeof unbindConnectionsChanged === 'function') {
          unbindConnectionsChanged()
        }
      })
    }

    // Reconciliation sweep: hide every Bot Mode session we know about, on
    // load and again on each reconnect (a swap can land on a gateway whose
    // rows were created before the always-hidden policy). Deferred a tick so
    // the meta/room storage hydrates above have landed; idempotent after that.
    // (Feature-guarded: bare vm test harnesses have no setTimeout global.)
    startHideSweepScheduler(ctx)
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'Bots',
      // dock: explicit adoption gesture — CENTER-STACK into the sessions zone
      // so the sidebar grows a SESSIONS | BOTS tab strip instead of splitting
      // two cramped panes down the column. Center is safe now: insertAtGroup
      // pins the zone's header explicitly shown on a center gain (and it
      // stays shown once the zone has stacked), so the sessions pane can
      // never vanish behind a stripless Bots tab — the lone-pane auto-hide
      // trap this dock used to work around with a 'bottom' split.
      // enforce: standing invariant, not a one-shot migration — the pane
      // re-homes into the sessions strip at EVERY boot it isn't already
      // there, whatever tokens or user placement an older install persisted.
      // The one-time heal ('sessions-tab-v1') burned its token even when its
      // guards skipped the move, so exactly the users who had fought the old
      // stacked layout (dragged panes → $userPlacedPanes) stayed stacked
      // forever. Owner's order: SESSIONS | BOTS is always a tab strip.
      // An intra-session drag still sticks until the next launch (the
      // invariant runs at adoption time only — see enforceDockedPanes in the
      // tree store).
      // collapsible: the pane lives in the sessions zone, so it must LEAVE
      // the grid with that zone below the sidebar-collapse breakpoint. The
      // sessions pane collapses alone without this flag. The zone then keeps
      // a stranded BOTS tab on screen. The narrow edge overlay mirrors the
      // zone's tab strip, so the pane stays reachable while collapsed.
      data: {
        placement: 'left',
        width: '260px',
        collapsible: true,
        hideOnly: true,
        dock: {
          pane: 'sessions',
          pos: 'center',
          enforce: true
        }
      },
      render: () => <BotsPane />
    })

    // Routines — its OWN tiling pane splitting the workspace's right edge
    // (NOT the collapsible right sidebar; placement 'right' is that sidebar's
    // role and hides the pane until "Show Right Sidebar").
    //
    // Registered ONLY while Bot Mode is on screen: the pane exists while the
    // Bots pane is visible (its zone's active tab, or a lone pane in a
    // stacked pre-heal layout) and unregisters when the user tabs back to
    // Sessions — no Cronjobs tile squatting beside the chat outside Bot Mode.
    // `ctx.register` returns the disposer that makes this cheap; the tree
    // keeps the pane's spot, so re-registering re-adopts it where it was.
    // host.paneVisibility is feature-detected: older desktops without the SDK
    // export keep the always-registered behavior.
    const registerRoutinesPane = () =>
      ctx.register({
        id: 'routines',
        area: 'panes',
        // The app's noun for these, so the tab agrees with the pane header and
        // with the core Scheduled jobs surface. `translateNow`, not `useI18n`:
        // a pane title is read at registration, outside React.
        title: translateNow('cron.title'),
        data: {
          placement: 'main',
          // Repair persisted layouts that stranded Cronjobs in the Bots tab strip.
          dock: {
            pane: 'workspace',
            pos: 'right',
            enforce: true
          },
          // A bot's schedule is glanceable, not something you sit in — it
          // arrives as the right edge's vertical tab and takes no width off the
          // chat until the user opens it.
          defaultCollapsed: true,
          width: '250px'
        },
        render: () => <RoutinesPane />
      })
    if (typeof host.paneVisibility === 'function') {
      // The contribution-scoped pane id (`register` prefixes `${ID}:`).
      const $sidebarVisible = host.paneVisibility(`${ID}:pane`)
      let unregisterRoutines: null | (() => void) = null
      const syncRoutinesPane = () => {
        if (botChatOwnsWorkspace()) {
          unregisterRoutines ??= registerRoutinesPane()
        } else if (unregisterRoutines) {
          // Clicking the Cronjobs tile moves focus onto the tile itself, which
          // drops bot-chat workspace ownership for a beat. While Bot Mode is
          // still on screen and the tile is the one holding focus, keep it —
          // a pane must never unregister itself out from under its own click.
          // Leaving Bot Mode ($botsPaneVisible false) still unregisters.
          const $self = host.paneVisibility(`${ID}:routines`)
          if ($botsPaneVisible.get() && $self && typeof $self.get === 'function' && $self.get()) return
          unregisterRoutines()
          unregisterRoutines = null
        }
      }
      const stopSidebarSync = $sidebarVisible.listen(visible => {
        $botsPaneVisible.set(Boolean(visible))
        if (visible) {
          const group = $groupChatWorkspace.get()
          const selected = selectedRosterBot($lastRoster.get(), $selectedRosterKey.get())
          // Owner routing only. With neither a group nor a selected bot there
          // is no bot-owned surface to scope, so leave the workspace on
          // whatever the user was already looking at — the roster-hydration
          // effect in BotsPane scopes it the moment a real owner exists.
          if (group) {
            setBotsWorkspaceOwner(
              groupWorkspaceOwnerKey(group),
              null,
              'New group conversations start in the group composer.'
            )
          } else if (selected) {
            setBotsWorkspaceOwner(botWorkspaceOwnerKey(selected), selected)
          }
        } else {
          // Strand any owner wake still dialing. Its SDK open will fail the
          // workspace token too; this plugin generation prevents that expected
          // cancellation from showing an error after the user deliberately
          // returned to Sessions.
          bumpBotOpenGeneration()
          host.setWorkspaceScope?.('sessions')
        }
        syncRoutinesPane()
      })
      const stopGroupSync = $groupChatWorkspace.listen(syncRoutinesPane)
      // React on the NEXT tick — a layout notification arrives mid-mutation,
      // and registering/unregistering panes from inside it would re-enter the
      // tree store.
      const scheduleRoutinesSync = () => {
        try {
          setTimeout(syncRoutinesPane, 0)
        } catch {
          syncRoutinesPane()
        }
      }
      // Tab focus moves without swapping the gateway socket, so the focused
      // STORED session is the truth about session focus; older shells fall
      // back to the active session id.
      const focusStore = host.state.focusedStoredSessionId || host.state.activeSessionId
      const stopFocusSync =
        typeof focusStore?.listen === 'function'
          ? focusStore.listen(id => {
              $botChatFocused.set(Boolean(id))
              releaseStaleOpenBotChat(id)
              syncRoutinesPane()
            })
          : null

      // Proactive reclaim refresh: when the gateway reaps the runtime behind
      // the OPEN bot chat (idle TTL, LRU cap, WS-orphan reap — the mass-reap
      // shape hits every background bot at once), re-resume the canonical
      // chat immediately instead of letting the user's next send eat the
      // stale-id error + recovery retry. Matched on the STORED id (the
      // claim's ids are stored ids; the payload carries both). Best-effort:
      // a failed re-resume (backend still down) leaves the lazy recovery on
      // next send as the backstop. Feature-detected — older shells have no
      // host.onEvent.
      const stopReclaimSync =
        typeof host.onEvent === 'function'
          ? host.onEvent('session.reclaimed', event => {
              const payload = (event?.payload || {}) as { stored_session_id?: string }
              const stored = String(payload.stored_session_id || '')
              const claim = $openBotChat.get()
              if (!stored || !claim) {
                return
              }
              const owned = [claim.openedSessionId, claim.openedRegistryId].filter(Boolean)
              if (!owned.includes(stored)) {
                return
              }
              const bot = selectedRosterBot($lastRoster.get(), $selectedRosterKey.get())
              if (!bot) {
                return
              }
              const generation = getBotOpenGeneration()
              void openBotCanonicalChat(bot)
                .then(opened => {
                  // A user action while the re-resume ran owns the center now.
                  if (!opened || generation !== getBotOpenGeneration()) {
                    return
                  }
                  $openBotChat.set({
                    key: claim.key,
                    openedRegistryId: opened.registryId,
                    openedSessionId: opened.openedId
                  })
                })
                .catch(() => {
                  /* backend still down — next send recovers via the ladder */
                })
            })
          : null
      $botsPaneVisible.set(Boolean($sidebarVisible.get()))
      $botChatFocused.set(sessionOwnsWorkspace())
      // A persisted layout can boot directly into Bot Mode. Reconcile now,
      // then once more after the layout mutation finishes.
      syncRoutinesPane()
      scheduleRoutinesSync()
      if (typeof ctx.onDispose === 'function') {
        // The registration disposer is already tracked by ctx.register; only
        // the listeners need explicit teardown or they survive plugin disable.
        ctx.onDispose(() => {
          stopSidebarSync()
          stopGroupSync()
          stopFocusSync?.()
          stopReclaimSync?.()
        })
      }
    } else {
      registerRoutinesPane()
    }
    // A bot's chat before it has spoken: core's splash is Hermes' wordmark and
    // stands down for any session that exists, so the bot titles its own.
    ctx.register({
      id: 'chat-empty',
      area: CHAT_EMPTY_AREA,
      data: {
        render: ({ sessionId }: ChatEmptyProps) => <BotChatEmpty sessionId={sessionId} />
      }
    })

    ctx.register({
      id: 'new-agent',
      area: PALETTE_AREA,
      data: {
        id: `${ID}.new-agent`,
        label: 'New Bot…',
        keywords: ['bot', 'agent', 'profile', 'teammate', 'create'],
        run: () => {
          host.notify({
            kind: 'info',
            message: ctx.i18n.t('bot.createFirstHint')
          })
        }
      }
    })

    // @-mention middleware: "@<bot> do the thing" in any chat gets an
    // IDENTIFICATION note — who the user is referring to, resolved against
    // the LIVE roster ("user@example.com" or an unknown @ passes through
    // untouched). The middleware never delivers anything itself: the agent
    // owns messaging via its message_agent tool (Bot Chats), so there is
    // exactly one send path and user text is never forwarded verbatim by
    // the renderer. The composer's @-autocomplete remains the picking aid.
    ctx.register({
      id: 'mention-middleware',
      area: COMPOSER_AREAS.middleware,
      data: {
        handler: async (draft: ComposerDraftPayload): Promise<ComposerDraftPayload> => {
          const text = draft.text || ''

          // /new inside a bot's canonical forever-chat would fork the
          // relationship into a scratch session — the one thing Bots mode
          // promises never happens. Reroute to /compact (same felt effect:
          // fresh working context, SAME conversation) and say so. Only
          // guards the canonical chat: Sessions-mode scratchpads on the
          // same profile keep full /new freedom.
          const slashNew = /^\/(new|reset)\s*$/.exec(text.trim())
          if (slashNew) {
            const activeBot = $selectedBot.get()
            // Canonical identity is the profile's "Bot Chat" registry row —
            // read it from the roster cache (canonical_session, resolved
            // server-side by name), matching either the durable row id or
            // the compression-lineage tip currently on screen.
            const roster = $lastRoster.get()
            const row = Array.isArray(roster) ? roster.find(bot => bot?.name === activeBot) : null
            const canonical = row?.canonical_session || null
            // TODO(bot-mode-types): host.activeSessionId does not exist — see host.state.activeSessionId
            // @ts-expect-error typed as written so the live bug stays visible instead of being fixed here.
            const currentId = host.activeSessionId?.get?.() ?? null
            const canonicalIds = [canonical?.id, canonical?.resolved_id].filter(Boolean).map(String)
            if (activeBot && currentId && canonicalIds.includes(String(currentId))) {
              host.notify({
                kind: 'info',
                title: 'This chat never resets',
                message:
                  'Bot chats are one continuous conversation — compacting instead. ' +
                  'For a throwaway session with this bot, use Sessions mode.'
              })
              return {
                ...draft,
                text: '/compact'
              }
            }
          }
          if (!/(^|\s)@[a-z0-9][a-z0-9_-]*/i.test(text)) {
            return draft
          }
          const live = {
            name: focusedMentionProfile(),
            connectionId: String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
          }
          const cached = cachedUnionRoster()
          const roster = Array.isArray(cached?.profiles) ? cached.profiles : null
          let mentionedBots = roster ? resolveRosterMentions(text, roster, live) : []
          if (!roster) {
            try {
              const res = await host.request<{ profiles?: RosterRow[] }>('profiles.list', {
                include_sessions: false
              })
              // Same resolver as the cached path — renamed bots (display_name
              // / ui_meta title) stay taggable when the roster cache is cold.
              mentionedBots = resolveRosterMentions(text, res?.profiles ?? [], live).map(bot => ({
                ...bot,
                remoteSource: false
              }))
            } catch {
              return draft
            }
          }
          if (!mentionedBots.length) {
            return draft
          }

          // Identification only. Each line names the agent the user's tag
          // resolves to (friendly title + device for cross-connection rows),
          // so the agent knows exactly who "@research-buddy" is. Cross-
          // connection targets carry the '@connection' suffix message_agent
          // resolves against the Desktop-synced relay roster.
          const lines = mentionedBots.map(bot => {
            const handle = botHandle(bot.name, bot)
            const title = String(
              botRosterMeta(bot, $botMeta.get())?.title || bot.ui_meta?.['hermes-bots']?.title || bot.title || ''
            ).trim()
            const target = bot.remoteSource && bot.connectionId ? `${handle}@${bot.connectionId}` : handle
            const where = bot.remoteSource
              ? ` — on ${bot.connectionLabel || bot.connectionId} (message_agent target: "${target}")`
              : ''
            return `@${handle} = agent profile "${bot.name}"${title ? ` ("${title}")` : ''}${where}`
          })
          const note =
            '\n\n[@mentions resolved from the Bot Mode roster — the user is referring to: ' +
            lines.join('; ') +
            '. If they want one of these agents contacted, compose your own message and send it with your message_agent tool (agents on other connected machines are reachable too — the Desktop relays it); never forward the user\u2019s text verbatim. If this session has no message_agent tool, agent messaging is unavailable here — say so.]'
          return {
            ...draft,
            text: text + note
          }
        }
      }
    })
  }
}
