import type { ProfilesCreateResult } from '@hermes/plugin-sdk'
import {
  Badge,
  Button,
  Checkbox,
  cn,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DisclosureCaret,
  GlyphSpinner,
  host,
  Input,
  queryClient,
  RowButton,
  SearchField,
  SegmentedControl,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Textarea,
  Tip,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
/**
 * The two creation dialogs: New Bot, and the group-chat creation flow the
 * roster's group button opens (plus the per-bot group membership dialog it
 * sits beside).
 *
 * New Bot creates its profile LAZILY, so it owns the single-flight guard and
 * the name pattern that gates the draft.
 */
import { useEffect, useRef, useState } from 'react'

import { avatarColor, blobatarSvg, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import { AvatarPicker } from './avatar-picker'
import { $selectedBot } from './bot-state'
import { createCanonicalChat } from './canonical-chat'
import { canonicalMemberName } from './canonical-group-activity'
import {
  botOnConnection,
  connectionName,
  groupCreationEngine,
  groupSeatsPeers,
  revalidateGroupCreation,
  seatableGroupBots,
  useGroupCapability
} from './canonical-group-capabilities'
import type { GroupCreationPlan } from './canonical-group-capabilities'
import { createGroupAcrossConnections, peerSetupText, usePeerTargets } from './canonical-group-peers'
import { refreshGatewayGroupRooms, registerCanonicalGroup } from './canonical-group-registry'
import type { CanonicalGroupOrigin } from './canonical-group-registry'
import { canonicalGroupCreateError, canonicalGroupCreateNotice, createCanonicalGroup } from './canonical-groups'
import type { GroupCreationIntent } from './canonical-groups'
import { $botMeta, $lastRoster, botHandle, botRosterKey, filterBots, ROSTER_KEY, saveBotMeta } from './data'
import { labeled, ResizableFrame } from './dialog-parts'
import { GROUP_CHAT_MAX_MEMBERS, uniqueGroupChatName } from './group-chat'
import { memberMachineOffline, setGroupMembership } from './group-chat-view-members'
import {
  botGroups,
  durableGroupChatMembers,
  knownGroups,
  liveGroupChatNames
} from './group-membership'
import { useBots } from './i18n'
import { botProfileIdentity, displayName } from './labels'
import { McpSetupButton } from './mcp-setup'
import { ModelPicker } from './model-picker'
import type {
  CapabilityEntry,
  McpCatalogResponse,
  ProfileConfigurePayload,
  ProfileDescribeResponse
} from './profile-config'
import { CapabilitiesView, capabilitiesViewRoutesConnections, CheckList } from './profile-config'
import { deleteBot } from './profile-ops'
import { botRosterMeta } from './routing'
import { HubSkillsSection } from './skills-hub'
import { composeSoul } from './soul'
import type { BotMeta, ConnectionRow, RosterRow } from './types'

const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

/** Share one in-flight async operation across concurrent callers. Failures
 * clear the slot so a later attempt can retry. */
export function singleFlight<T>(ref: { current: null | Promise<T> }, start: () => Promise<T> | T): Promise<T> {
  if (ref.current) {
    return ref.current
  }

  let flight: Promise<T>

  try {
    flight = Promise.resolve(start())
  } catch (err) {
    flight = Promise.reject(err)
  }

  ref.current = flight
  flight.catch(() => {
    if (ref.current === flight) {
      ref.current = null
    }
  })

  return flight
}

// ── create dialog ────────────────────────────────────────────────────────────

/** The clone source's capability catalog, staged before the profile exists. */
interface CapabilityCatalog {
  mcp: (CapabilityEntry & { installedInDraft?: boolean })[]
  skills: CapabilityEntry[]
  source: string
  toolsets: CapabilityEntry[]
}
interface CreateAgentDialogProps {
  onClose: () => void
  /** Opens the editor for a just-created local bot whose model is not ready. */
  onConfigureModel?: (bot: RosterRow) => void
  open: boolean
  roster: RosterRow[]
}

type CapabilityPicks = Pick<ProfileConfigurePayload, 'disabled_skills' | 'enabled_mcp_servers' | 'enabled_toolsets'>

/** The `profiles.configure` fields for the Advanced-tab picks the user changed. */
function capabilityPayload(
  caps: CapabilityCatalog | null,
  dirty: { mcp: boolean; skills: boolean; toolsets: boolean }
): CapabilityPicks {
  const payload: CapabilityPicks = {}

  if (!caps) {
    return payload
  }

  if (dirty.skills) {
    payload.disabled_skills = caps.skills.filter(s => !s.enabled).map(s => s.name)
  }

  if (dirty.toolsets) {
    const en = caps.toolsets.filter(t => t.enabled)
    payload.enabled_toolsets = en.length === caps.toolsets.length || en.length === 0 ? [] : en.map(t => t.name)
  }

  if (dirty.mcp) {
    payload.enabled_mcp_servers = caps.mcp.filter(m => m.enabled).map(m => m.name)
  }

  return payload
}

/** Gateway RPC on a create target: `target` is a registry connection id (its
 *  default backend, no gateway switch) or '' for the active gateway. */
function requestOnTarget<T>(target: string, method: string, params: Record<string, unknown> = {}): Promise<T> {
  return target
    ? host.requestProfile({ connectionId: target, mode: 'remote', profile: 'default', targetProfile: 'default' }, method, params)
    : host.request(method, params)
}

/** Pins the Configure model action to the connection the bot was created on.
 *  An unknown owner (legacy remote gateway) or a host without per-profile
 *  routing keeps the unscoped row the active gateway resolves. */
function creationOwnerFields(owner: string, remote: boolean): Partial<RosterRow> {
  const scoped = owner && typeof host.requestProfile === 'function' ? { connectionId: owner, sourceScoped: true } : {}

  return remote ? { ...scoped, remoteSource: true } : scoped
}

/** Delete a draft profile the New Agent dialog minted, on the `target` it was
 *  created on: a remote connection (`cli.exec`) or '' for local. Fire-and-forget
 *  with a toast. A cancel while profiles.create is still in flight is caught by
 *  that flight itself (the dialog's draftGenRef), since the draft is only
 *  recorded once the RPC returns (#141623). */
function discardDraftProfile(draft: string, target: string) {
  const discard = target
    ? requestOnTarget(target, 'cli.exec', { argv: ['profile', 'delete', draft, '--yes'] })
    : deleteBot({ name: draft })

  void Promise.resolve(discard)
    .then(() => host.notify({ kind: 'success', message: `Draft agent "${draft}" discarded` }))
    .catch(err => host.notifyError(err, `Could not clean up draft profile "${draft}"`))
}

/** The live Capabilities surface needs a real profile. Materialize it whenever
 *  the tab is showing a valid, untaken name that has no draft yet, not only on
 *  the tab click: a user who opens the tab first and names the bot after (or
 *  renames it) otherwise sat on a spinner forever. Debounced so typing a name
 *  does not mint a draft per keystroke. */
function useCapabilitiesDraft(ready: boolean, slug: string, draft: null | string, materialize: () => void) {
  // The timer is keyed on the name, but must run the LATEST callback: a title
  // or description typed during the wait belongs in the created profile.
  const latest = useRef(materialize)
  latest.current = materialize

  useEffect(() => {
    if (!ready || draft === slug) {
      return
    }

    const timer = window.setTimeout(() => latest.current(), 800)

    return () => window.clearTimeout(timer)
  }, [ready, slug, draft])
}

/** New Bot's live Capabilities tab: a hint until the name is usable, a spinner
 *  until its draft profile exists, then the real surface pinned to that draft. */
function CapabilitiesTab({ connection, draft, hint }: { connection?: string; draft: null | string; hint: null | string }) {
  const b = useBots()
  const note = 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)'

  if (hint) {
    return <div className={note}>{hint}</div>
  }

  if (!draft) {
    return (
      <div className="flex justify-center py-4">
        <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
      </div>
    )
  }

  // The tab list is gated on the same export, so this is only reachable via
  // persisted tab state on a build that lacks it — a message rather than
  // rendering `undefined` as a component, which throws.
  if (!CapabilitiesView) {
    return <div className={note}>{b.editor.newerDesktop}</div>
  }

  return (
    <ResizableFrame>
      <CapabilitiesView embedded fixedProfile={draft} {...(connection ? { fixedConnection: connection } : {})} />
    </ResizableFrame>
  )
}

/** Is `slug` already a bot on the create target (`target`: a remote connection
 *  id, or '' for local)? The dialog's own materialized draft never counts. */
function slugTaken(roster: RosterRow[], slug: string, target: string, draft: null | string): boolean {
  return roster.some(
    b =>
      b.name === slug &&
      b.name !== draft &&
      (target ? Boolean(b.remoteSource) && b.connectionId === target : !b.remoteSource)
  )
}

/** Draft-profile ledger for the New Agent dialog. Opening Capabilities (or MCP
 *  setup) materializes a DRAFT profile so the live config surfaces have a
 *  backend; discarding deletes it on the machine it was CREATED on
 *  (`draftTargetRef`, recorded with it): the "Create on" picker may point
 *  elsewhere by then, and a same-named bot there is not the draft. Each discard,
 *  and the dialog unmounting, bumps `draftGenRef` so an in-flight create
 *  deletes its own draft and configures nothing. */
function useDraftProfile() {
  const [caps, setCaps] = useState<CapabilityCatalog | null>(null)
  const createdRef = useRef<null | string>(null)
  const flightRef = useRef<Promise<null | string> | null>(null)
  // The name the in-flight create is minting; a different name replaces it.
  const flightSlugRef = useRef('')
  const [createdForCaps, setCreatedForCaps] = useState<null | string>(null)
  const draftGenRef = useRef(0)
  const draftTargetRef = useRef('')
  const draftOwnerRef = useRef('')
  const draftWarningRef = useRef<{ generation: number; message: string; title: string } | null>(null)
  // Keys the MCP setup buttons: bumping it drops their forms and pending flows.
  const [setupGen, setSetupGen] = useState(0)

  // Pending and completed MCP setup belongs to the old name's draft, never
  // its replacement: drop the buttons' flows and un-install draft-only rows.
  const retireSetup = () => {
    setSetupGen(gen => gen + 1)
    setCaps(prev => prev?.mcp.some(m => m.installedInDraft) ? {
      ...prev,
      mcp: prev.mcp.map(m => m.installedInDraft
        ? { ...m, installed: false, enabled: false, installedInDraft: false }
        : m)
    } : prev)
  }

  // Drops the draft itself. Its setup was already retired by the rename that
  // made it stale, and retiring again here would remount the very button
  // whose click is replacing the draft.
  const replaceDraft = () => {
    const draft = createdRef.current
    draftGenRef.current += 1
    draftWarningRef.current = null
    createdRef.current = flightRef.current = null
    setCreatedForCaps(null)

    if (draft) {
      discardDraftProfile(draft, draftTargetRef.current)
    }
  }

  const discardDraft = () => {
    replaceDraft()
    retireSetup()
  }

  useEffect(
    () => () => {
      draftGenRef.current += 1
      draftWarningRef.current = null
    },
    []
  )

  // A rename retires the draft's setup but not the draft: it is deleted only
  // once replaced, so renaming back reuses it instead of racing its delete.
  const renameDraft = (next: string, slug: string, setName: (name: string) => void) => {
    if ((createdRef.current || flightRef.current) && botProfileIdentity(next, '').slug !== slug) {
      retireSetup()
    }

    setName(next)
  }

  return { caps, setCaps, createdRef, flightRef, flightSlugRef, createdForCaps, setCreatedForCaps, discardDraft, replaceDraft, renameDraft, draftGenRef, draftTargetRef, draftOwnerRef, draftWarningRef, setupGen }
}

export function CreateAgentDialog({ open, onClose, onConfigureModel, roster }: CreateAgentDialogProps) {
  const { t } = useI18n()
  const b = useBots()
  const [name, setName] = useState('')

  const { caps, setCaps, createdRef, flightRef, flightSlugRef, createdForCaps, setCreatedForCaps, discardDraft, replaceDraft, renameDraft, draftGenRef, draftTargetRef, draftOwnerRef, draftWarningRef, setupGen } = useDraftProfile()

  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  // Default shapes mode: deterministic blob face drawn from the agent's name
  // (falls back to the legacy shape vocabulary on older SDKs).
  const [shape, setShape] = useState(blobatarSvg ? 'blobatar' : 'circle')
  const [color, setColor] = useState<null | string>(null)
  const [image, setImage] = useState<null | string>(null)
  const [advanced, setAdvanced] = useState(false)
  const [cloneFrom, setCloneFrom] = useState('default')
  const [model, setModel] = useState('')
  const [provider, setProvider] = useState('')
  const [soul, setSoul] = useState('')
  const [noSkills, setNoSkills] = useState(false)
  const [shareAuth, setShareAuth] = useState(true)
  const [advTab, setAdvTab] = useState('general')
  // Where the profile is created: '' = the active gateway (unchanged default),
  // else a registry connection id — the profiles.create lands on THAT
  // machine's backend via host.requestProfile, no gateway switch. Only
  // rendered when the desktop has a multi-connection registry.
  const [targetConnection, setTargetConnection] = useState('')
  const [connections, setConnections] = useState<ConnectionRow[] | null>(null)
  useEffect(() => {
    if (
      !open ||
      connections !== null ||
      typeof host.connections !== 'function' ||
      typeof host.requestProfile !== 'function'
    ) {
      return
    }

    host
      .connections()
      // host.connections() returns the registry ROWS on current SDKs, but the
      // envelope object ({version, primary, connections: [...]}) on desktops
      // that predate the SDK-side unwrap — accept both shapes.
      .then((value: ConnectionRow[] | { connections?: ConnectionRow[] }) =>
        setConnections(Array.isArray(value) ? value : Array.isArray(value?.connections) ? value.connections : [])
      )
      .catch(() => setConnections([]))
  }, [open, connections])
  const activeConnectionId = String(host.state?.connectionId?.get?.() || '').trim()
  // Remote target = an explicitly picked registry connection that is not the
  // one this window is already on.
  const remoteTarget = Boolean(targetConnection) && targetConnection !== (activeConnectionId || 'local')

  const targetLabel = remoteTarget
    ? (connections || []).find(c => c.id === targetConnection)?.label || targetConnection
    : ''

  /** Where the profile is created right now: the picked connection id for a
   *  remote target, '' for the active gateway. */
  const createTarget = remoteTarget ? targetConnection : ''

  /** Gateway RPC on the create target: the picked connection's default
   *  backend for remote targets, the active gateway otherwise. */
  const requestForTarget = <T,>(method: string, params: Record<string, unknown> = {}): Promise<T> =>
    requestOnTarget<T>(createTarget, method, params)

  const [capsFailed, setCapsFailed] = useState(false)

  const [dirtyCaps, setDirtyCaps] = useState({
    skills: false,
    toolsets: false,
    mcp: false
  })

  const [capFilter, setCapFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const { slug, title: botTitle } = botProfileIdentity(name, title)
  const descriptionText = [botTitle, description].filter(Boolean).join(' — ')
  const valid = slug.length > 0 && NAME_RE.test(slug)

  // Once the draft profile is materialized (Capabilities tab / MCP setup) it
  // shows up in the roster — its OWN slug must not read as "taken".
  // A remote-target create is gated by the TARGET machine's roster: a local
  // name clash is fine there, and the remote's own duplicate check rejects
  // real collisions at profiles.create time.
  const taken = slugTaken(roster, slug, createTarget, createdRef.current)

  const reset = () => {
    setName('')
    setTitle('')
    setDescription('')
    setShape(blobatarSvg ? 'blobatar' : 'circle')
    setColor(null)
    setImage(null)
    setAdvanced(false)
    // Same default as the initial useState — resetting to '__none__' made
    // the second agent you create silently start from a fresh profile
    // instead of cloning the main one like the first dialog open did.
    setCloneFrom('default')
    setModel('')
    setProvider('')
    setSoul('')
    setNoSkills(false)
    setShareAuth(true)
    setAdvTab('general')
    setCreatedForCaps(null)
    setCaps(null)
    setCapsFailed(false)
    setDirtyCaps({
      skills: false,
      toolsets: false,
      mcp: false
    })
    setCapFilter('')
    setTargetConnection('')
    setBusy(false)
    setError(null)
    createdRef.current = null
    flightRef.current = null
    draftWarningRef.current = null
    draftGenRef.current += 1
  }

  // Capability catalog for the tabs: the profile doesn't exist yet, so show
  // what it WILL have — the clone source's catalog, else the main profile's.
  // Same rule as the `clone_from` payload below: a remote target can only
  // clone ITS default, so a local roster name picked before the target
  // switched must not key the preview (or the describe call) either.
  const capSource = cloneFrom === '__none__' || remoteTarget ? 'default' : cloneFrom

  const materializeForCaps = () =>
    void ensureAgentCreated()
      .then(created => created && setCreatedForCaps(created))
      .catch(err => host.notifyError(err, b.bot.createFailed))

  useCapabilitiesDraft(advTab === 'capabilities' && valid && !taken, slug, createdForCaps, materializeForCaps)

  const ensureCaps = () => {
    if ((caps && caps.source === capSource) || capsFailed) {
      return
    }

    Promise.all([
      requestForTarget<ProfileDescribeResponse>('profiles.describe', {
        name: capSource
      }),
      requestForTarget<McpCatalogResponse>('mcp.catalog', {}).catch(() => null)
    ])
      .then(([res, cat]) => {
        // Full MCP menu = the profile's configured servers + the bundled
        // catalog (installable). Configured entries win on name clash.
        const configured = res.mcp_servers || []
        const have = new Set(configured.map(m => m.name))
        const catalog = ((cat && cat.servers) || []).filter(s => !have.has(s.name))
        setCaps({
          source: capSource,
          skills: res.skills || [],
          toolsets: res.toolsets || [],
          mcp: [
            ...configured,
            ...catalog.map(s => ({
              name: s.name,
              enabled: false,
              fromCatalog: true,
              installed: s.installed,
              auth: s.auth,
              requires: s.requires || [],
              description: s.description || ''
            }))
          ]
        })
      })
      .catch(() => setCapsFailed(true))
  }

  const toggleCap = (kind: 'mcp' | 'skills' | 'toolsets', name: string, enabled: boolean) => {
    setDirtyCaps(prev => ({
      ...prev,
      [kind === 'mcp' ? 'mcp' : kind]: true
    }))
    setCaps(prev =>
      prev
        ? {
            ...prev,
            [kind]: prev[kind].map(x =>
              x.name === name
                ? {
                    ...x,
                    enabled
                  }
                : x
            )
          }
        : prev
    )
  }

  // Materialize the profile exactly once. createdRef stores the finished slug
  // (its consumers — the taken check, draft discard on cancel, the MCP setup
  // button's profile param — all read a string); flightRef shares the
  // in-flight creation promise so simultaneous MCP setup / Create clicks fire
  // ONE profiles.create. A failed flight clears its slot so a retry can run.
  const ensureAgentCreated = (): Promise<null | string> => {
    // Renamed since the draft materialized (or started)? The old draft is
    // orphaned — discard it and create fresh under the new slug.
    if ((createdRef.current || flightRef.current) && (createdRef.current || flightSlugRef.current) !== slug) {
      replaceDraft()
    }

    // A recorded draft can still be applying its initial capability picks.
    if (flightRef.current) {
      return flightRef.current
    }

    if (createdRef.current) {
      return Promise.resolve(createdRef.current)
    }

    // Never start a flight for an unusable name: singleFlight only clears its
    // slot on failure, so a null result would be returned to every later
    // caller and the draft could never be created once a name is typed.
    if (!valid || taken) {
      return Promise.resolve(null)
    }

    const generation = draftGenRef.current
    const target = createTarget
    // Unknown on a legacy remote gateway: never guess 'local' for it.
    const targetId = target || activeConnectionId
    const machine = (connections || []).find(c => c.id === targetId)?.label
    const warningTitle = targetId ? `${slug} — ${machine ? `${machine} (${targetId})` : targetId}` : slug

    const flight = singleFlight(flightRef, async () => {
      const result = await requestForTarget<ProfilesCreateResult>('profiles.create', {
        name: slug,
        description: descriptionText,
        // Clone sources are profiles of the TARGET backend. The picker's
        // roster is the local one, so a remote create always starts from the
        // remote machine's default (or fresh) — never a local profile name
        // the remote box doesn't have.
        clone_from: cloneFrom === '__none__' ? null : remoteTarget ? 'default' : cloneFrom,
        no_skills: noSkills,
        // Shared (not copied) auth keeps ONE OAuth/token pool with the default
        // profile, so refreshes can't invalidate each other. From a named launch
        // profile the gateway copies its logins instead (single-use OAuth still
        // read from the default profile). Older gateways ignore the param and copy.
        share_auth: shareAuth,
        soul: composeSoul({
          name: slug,
          title: botTitle,
          description,
          roster,
          customSoul: soul
        }),
        ...(model.trim() && provider.trim()
          ? {
              model: model.trim(),
              provider: provider.trim()
            }
          : {})
      })

      // Prefs a deleted profile left under this name must not become the new
      // bot's. Older builds without the door keep them, as before.
      if (typeof host.dropLeftoverProfilePrefs === 'function') {
        host.dropLeftoverProfilePrefs(slug, target || null)
      }

      // Discarded mid-create: delete the profile this flight minted, configure nothing.
      if (generation !== draftGenRef.current) {
        discardDraftProfile(slug, target)

        return null
      }

      createdRef.current = slug
      draftTargetRef.current = target
      draftOwnerRef.current = targetId

      // Apply capability picks from the Advanced tabs (best-effort; the
      // profile exists either way and Edit Profile can finish the job).
      try {
        const capPayload = capabilityPayload(caps, dirtyCaps)

        if (Object.keys(capPayload).length) {
          await requestForTarget('profiles.configure', {
            name: slug,
            ...capPayload
          })
        }
      } catch {
        /* capability application is best-effort */
      }

      // Cancelled while configure ran: the discard already deleted the draft,
      // so it gets no bot meta and is not adopted.
      if (generation !== draftGenRef.current) {
        return null
      }

      if (remoteTarget) {
        // The bot lives on ANOTHER machine — local bot-meta is scoped to the
        // active gateway, so write appearance/title into the remote
        // profile's ui_meta (and asset store) directly. Best-effort: the
        // profile exists either way.
        const { image: avatarImage, ...look } = {
          shape,
          color,
          image,
          imageKind: image ? 'photo' : 'shape',
          title: botTitle,
          created: Date.now()
        }

        try {
          void requestForTarget('profiles.configure', {
            name: slug,
            ui_meta: {
              'hermes-bots': look
            }
          }).catch(() => undefined)

          if (avatarImage) {
            void requestForTarget('profiles.set_asset', {
              name: slug,
              asset: 'avatar',
              data: avatarImage
            }).catch(() => undefined)
          }
        } catch {
          /* older remote gateway */
        }
      } else {
        saveBotMeta(slug, {
          shape,
          color: color ?? undefined,
          image,
          imageKind: image ? 'photo' : 'shape',
          title: botTitle,
          created: Date.now()
        })
      }

      queryClient.invalidateQueries({
        queryKey: ROSTER_KEY
      })

      // A disposable draft owns its warnings until final adoption, not a toast.
      draftWarningRef.current = result.warnings?.length
        ? { generation, title: warningTitle, message: result.warnings.join('\n') }
        : null

      return slug
    })

    flightSlugRef.current = slug

    return flight
  }

  const submit = async () => {
    if (!valid || taken || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const slugCreated = await ensureAgentCreated()

      if (!slugCreated) {
        setBusy(false)
        setError(b.bot.createFailed)

        return
      }

      // The intro turn is a real model call. Probe the new profile's route
      // first so a bot whose machine has no usable credential lands as
      // "created, needs a model" instead of a doomed "No LLM provider" turn.
      // Only an explicit `ok: false` counts; older gateways proceed as before.
      // The profile is adopted now: unmounting while readiness runs must not
      // drop its warnings, notice or chat.
      const warning = draftWarningRef.current?.generation === draftGenRef.current ? draftWarningRef.current : null

      const readiness = await requestForTarget<{ error?: string; ok?: boolean }>('setup.runtime_check', {
        profile: slugCreated
      }).catch(() => null)

      const needsModel = readiness?.ok === false
      const who = displayName({ name: slug, title: botTitle })
      const createdMessage = remoteTarget ? b.editor.createdOn(who, targetLabel) : b.editor.created(who)

      const bot: RosterRow = {
        name: slugCreated,
        description: descriptionText,
        title: botTitle,
        ...creationOwnerFields(draftOwnerRef.current, remoteTarget)
      }

      if (warning) {
        host.notify({ kind: 'warning', title: warning.title, message: warning.message, persistent: true })
        draftWarningRef.current = null
      }

      host.notify(
        needsModel
          ? {
              kind: 'warning',
              title: createdMessage,
              message: b.editor.needsModel,
              persistent: true,
              detail: readiness?.error,
              ...(!remoteTarget && onConfigureModel
                ? { action: { label: b.editor.configureModel, onClick: () => onConfigureModel(bot) } }
                : {})
            }
          : { kind: 'success', message: createdMessage }
      )
      const wasRemote = remoteTarget
      reset()
      onClose()

      if (wasRemote) {
        // The bot lives on another machine: it appears in the roster via the
        // union enumeration; chat routes through its own source. No local
        // canonical chat to birth here.
        queryClient.invalidateQueries({
          queryKey: ROSTER_KEY
        })

        return
      }

      $selectedBot.set(slug)

      // Birth the bot's forever chat right away: it introduces itself as
      // the first thing the user sees, and the pin exists from minute one.
      try {
        // Creates, pins, opens, and kicks off the intro in one flow. This is
        // the ONE caller allowed to request the intro turn — genuine New
        // Agent creation. Click-path resolution (openBotCanonicalChat) mints
        // silently so a resolution miss never burns a turn (ScottFive).
        const sid = await createCanonicalChat(slug, {
          kickoff: !needsModel
        })

        if (!sid && typeof host.newChat === 'function') {
          host.newChat(slug)
        }
      } catch {
        if (typeof host.newChat === 'function') {
          host.newChat(slug)
        }
      }
    } catch (err) {
      setBusy(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <Dialog
      onOpenChange={value => {
        if (!value && !busy) {
          // Cancel path (esc / overlay click): a materialized draft profile is
          // discarded — preconfigure-then-back-out leaves nothing behind.
          discardDraft()
          reset()
          onClose()
        }
      }}
      open={open}
    >
      <DialogContent
        className={advanced ? 'max-w-3xl' : 'max-w-md'} // Native resize handle (bottom-right corner): the dialog becomes a
        // window the user can grow/shrink. overflow:auto is required for CSS
        // resize to engage; caps keep it on screen.
        style={
          advanced
            ? {
                resize: 'both',
                overflow: 'auto',
                minWidth: 420,
                minHeight: 360,
                maxWidth: '95vw',
                maxHeight: '90vh'
              }
            : undefined
        }
      >
        <DialogHeader>
          <DialogTitle>{b.bot.newTitle}</DialogTitle>
          <DialogDescription>{b.editor.newDescription}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3.5">
          <div className="flex justify-center py-1">
            <BotFace
              color={avatarColor(color, slug || 'agent')}
              image={image}
              name={slug || 'agent'}
              shape={shape}
              size={56}
            />
          </div>
          <AvatarPicker
            color={color}
            generateSeed={{
              name: slug || 'agent',
              title,
              description
            }}
            image={image}
            onColor={setColor}
            onImage={setImage}
            onShape={setShape}
            shape={shape}
          />
          {labeled(
            b.editor.name,
            <Input
              autoFocus
              disabled={busy}
              onChange={event => renameDraft(event.target.value, slug, setName)}
              placeholder="inbox-triage"
              value={name}
            />
          )}
          {taken ? (
            <div className="text-xs text-(--ui-accent)">
              {remoteTarget ? b.editor.nameTakenOn(slug, targetLabel) : b.editor.nameTaken(slug)}
            </div>
          ) : null}
          {/* Multi-connection desktops choose WHERE the agent lives. Hidden */
          /* on single-connection setups — the active gateway is the only */
          /* possible home, exactly the old behavior. */}
          {Array.isArray(connections) && connections.length > 1
            ? labeled(
                b.editor.createOn,
                <Select
                  disabled={busy}
                  onValueChange={value => {
                    // A draft minted on the old home is not this bot anymore.
                    discardDraft()
                    setTargetConnection(value === (activeConnectionId || 'local') ? '' : value)
                    // The capability catalog and clone list belong to the
                    // target backend — refetch for the new home. The live
                    // Capabilities tab re-pins to it via fixedConnection on
                    // builds that route it (staged checklists otherwise).
                    setCaps(null)
                    setCapsFailed(false)
                    setAdvTab('general')
                  }}
                  value={targetConnection || activeConnectionId || 'local'}
                >
                  <SelectTrigger className="h-8 rounded-md">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {connections.map(connection => (
                      <SelectItem key={connection.id} value={connection.id}>
                        {connection.id === (activeConnectionId || 'local')
                          ? b.editor.currentConnection(connection.label || connection.id)
                          : connection.label || connection.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )
            : null}
          {remoteTarget ? (
            <div className="text-[0.7rem] leading-5 text-(--ui-text-tertiary)">{b.editor.remoteHint(targetLabel)}</div>
          ) : null}
          {labeled(
            b.editor.title,
            <Input onChange={event => setTitle(event.target.value)} placeholder="Inbox Triage" value={title} />
          )}
          {labeled(
            b.editor.description,
            <Textarea
              className="min-h-16"
              onChange={event => setDescription(event.target.value)}
              placeholder={b.bot.helpPromptPlaceholder}
              value={description}
            />
          )}
          <Button
            className="flex items-center gap-1 text-xs font-medium text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)"
            onClick={() => {
              setAdvanced(v => {
                if (!v) {
                  ensureCaps()
                }

                return !v
              })
            }}
            size="inline"
            variant="text"
          >
            <DisclosureCaret open={advanced} />
            {b.bot.advanced}
          </Button>
          {advanced ? (
            <div className="grid gap-3 rounded-md border border-(--ui-stroke-secondary) p-3">
              <SegmentedControl
                onChange={id => {
                  setAdvTab(id)
                  setCapFilter('')

                  if (id === 'capabilities') {
                    // The live surface needs a real profile; a name typed later
                    // is picked up by useCapabilitiesDraft.
                    materializeForCaps()
                  } else if (id !== 'general') {
                    ensureCaps()
                  }
                }}
                options={
                  CapabilitiesView && (!remoteTarget || capabilitiesViewRoutesConnections)
                    ? [
                        { id: 'general', label: b.editor.general },
                        { id: 'capabilities', label: b.editor.capabilities }
                      ]
                    : [
                        { id: 'general', label: b.editor.general },
                        { id: 'skills', label: b.editor.skills },
                        { id: 'toolsets', label: b.editor.tools },
                        { id: 'mcp', label: 'MCP' }
                      ]
                }
                value={advTab}
              />
              {advTab === 'general' ? (
                <div className="grid gap-3.5">
                  {labeled(
                    remoteTarget ? b.editor.cloneFromOn(targetLabel) : b.editor.cloneFrom,
                    <Select
                      onValueChange={value => {
                        setCloneFrom(value)
                        setCaps(null)
                        setCapsFailed(false)
                      }}
                      value={remoteTarget && cloneFrom !== '__none__' ? 'default' : cloneFrom}
                    >
                      <SelectTrigger className="h-8 rounded-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none__">{b.editor.freshProfile}</SelectItem>
                        {/* The roster lists THIS window's profiles; the only clone
                            source guaranteed to exist on another machine is its
                            own default, so a remote target offers that or fresh. */}
                        {(remoteTarget ? [{ name: 'default' }] : roster).map(b => (
                          <SelectItem key={b.name} value={b.name}>
                            {b.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                  <ModelPicker
                    onChange={patch => {
                      if ('provider' in patch) {
                        setProvider(patch.provider)
                      }

                      if ('model' in patch) {
                        setModel(patch.model)
                      }
                    }}
                    placeholderModel={b.editor.inheritedModel}
                    value={{
                      provider,
                      model
                    }}
                  />
                  {labeled(
                    b.editor.soul,
                    <Textarea
                      className="min-h-24 font-mono text-xs leading-5"
                      onChange={event => setSoul(event.target.value)}
                      placeholder={b.avatar.describeHint}
                      value={soul}
                    />
                  )}
                  <label className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
                    <Checkbox checked={shareAuth} onCheckedChange={value => setShareAuth(Boolean(value))} />
                    {remoteTarget ? b.editor.shareKeysOn(targetLabel) : b.editor.shareKeys}
                  </label>
                  <div className="pl-6 pt-0.5 text-[0.7rem] leading-5 text-(--ui-text-tertiary)">
                    {b.editor.shareKeysHint}
                  </div>
                  <label className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
                    <Checkbox checked={noSkills} onCheckedChange={value => setNoSkills(Boolean(value))} />
                    {b.editor.createEmpty}
                  </label>
                </div>
              ) : advTab === 'capabilities' ? (
                <CapabilitiesTab
                  connection={remoteTarget ? targetConnection : undefined}
                  draft={createdForCaps === slug ? createdForCaps : null}
                  hint={!valid ? b.editor.nameFirstHint : taken ? b.editor.nameTakenHint : null}
                />
              ) : capsFailed ? (
                <div className="px-2 py-3 text-center text-xs text-(--ui-text-tertiary)">{b.editor.newerGateway}</div>
              ) : !caps ? (
                <div className="flex justify-center py-4">
                  <GlyphSpinner className="text-(--ui-text-tertiary)" spinner="breathe" />
                </div>
              ) : advTab === 'skills' ? (
                noSkills ? (
                  <div className="px-2 py-3 text-center text-xs text-(--ui-text-tertiary)">
                    {b.editor.emptySkillsHint}
                  </div>
                ) : (
                  <div className="grid gap-1.5">
                    <Input
                      className="h-7 text-xs"
                      onChange={event => setCapFilter(event.target.value)}
                      placeholder={b.tools.filterSkills}
                      value={capFilter}
                    />
                    <div
                      className="overflow-y-auto overscroll-contain"
                      style={{
                        maxHeight: 200
                      }}
                    >
                      <CheckList
                        columns={2}
                        items={
                          capFilter.trim()
                            ? caps.skills.filter(s => s.name.toLowerCase().includes(capFilter.trim().toLowerCase()))
                            : caps.skills
                        }
                        onToggle={(name, enabled) => toggleCap('skills', name, enabled)}
                      />
                    </div>
                    <div className="text-[0.65rem] leading-4 text-(--ui-text-quaternary)">
                      {b.editor.catalogHint(caps.source)}
                    </div>
                    <HubSkillsSection
                      onInstalled={name =>
                        setCaps(prev =>
                          !prev || prev.skills.some(s => s.name === name)
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
                )
              ) : advTab === 'toolsets' ? (
                <div className="grid gap-1.5">
                  <div
                    className="overflow-y-auto overscroll-contain"
                    style={{
                      maxHeight: 200
                    }}
                  >
                    <CheckList
                      columns={2}
                      items={caps.toolsets}
                      onToggle={(name, enabled) => toggleCap('toolsets', name, enabled)}
                    />
                  </div>
                  <div className="text-[0.65rem] leading-4 text-(--ui-text-quaternary)">
                    {b.editor.defaultToolsHint}
                  </div>
                </div>
              ) : caps.mcp.length === 0 ? (
                <div className="px-2 py-3 text-center text-xs text-(--ui-text-tertiary)">{b.tools.noMcpServers}</div>
              ) : (
                <div className="grid gap-1.5">
                  <div
                    className="overflow-y-auto overscroll-contain"
                    style={{
                      maxHeight: 200
                    }}
                  >
                    <div className="grid gap-1">
                      {caps.mcp.map(m => {
                        const needsSetup =
                          m.fromCatalog &&
                          !m.installed &&
                          ((m.requires || []).length > 0 || (m.auth || '').toLowerCase() === 'oauth')

                        return (
                          <label className="flex items-start gap-2 text-xs text-(--ui-text-secondary)" key={m.name}>
                            <Checkbox
                              checked={!!m.enabled}
                              disabled={needsSetup}
                              onCheckedChange={value => toggleCap('mcp', m.name, Boolean(value))}
                            />
                            <span className="min-w-0">
                              <span>{m.name}</span>
                              {m.fromCatalog && !needsSetup ? (
                                <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">
                                  {m.installed ? b.editor.catalogInstalled : b.editor.catalog}
                                </span>
                              ) : null}
                              {needsSetup ? (
                                <McpSetupButton
                                  ensureProfile={ensureAgentCreated}
                                  entry={m}
                                  key={setupGen}
                                  onDone={() => {
                                    // Setup done: mark installed so the row's
                                    // checkbox un-disables, and enable it.
                                    setCaps(prev =>
                                      prev
                                        ? {
                                            ...prev,
                                            mcp: prev.mcp.map(x =>
                                              x.name === m.name
                                                ? {
                                                    ...x,
                                                    installed: true,
                                                    installedInDraft: true,
                                                    enabled: true
                                                  }
                                                : x
                                            )
                                          }
                                        : prev
                                    )
                                    setDirtyCaps(prev => ({
                                      ...prev,
                                      mcp: true
                                    }))
                                  }}
                                  // A renamed-away draft is not this name's profile.
                                  profile={createdRef.current === slug ? slug : null}
                                />
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
                  <div className="text-[0.65rem] leading-4 text-(--ui-text-quaternary)">{b.editor.mcpHint}</div>
                </div>
              )}
            </div>
          ) : null}
          {error ? (
            <div className="rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs text-(--ui-accent)">
              {error}
            </div>
          ) : null}
        </div>
        <DialogFooter>
          <Button
            disabled={busy}
            onClick={() => {
              discardDraft()
              reset()
              onClose()
            }}
            variant="ghost"
          >
            {t.common.cancel}
          </Button>
          <Button disabled={busy || !valid || taken} onClick={submit}>
            {busy ? b.editor.creating : b.editor.createBot}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface GroupDialogProps {
  bot: RosterRow
  onClose: () => void
  /** Called with the group chat New group chat created for a new name. */
  onCreated?: (group: string, origin: CanonicalGroupOrigin) => void
}

/** Assign a bot to a group-chat membership without replacing its others.
 *  Existing groups are independent toggles. A new name is a new group chat, so
 *  Create & join opens New group chat with this bot and the name filled in: a
 *  gateway runs it, and nothing here makes a Desktop-run room. */
export function GroupDialog({ bot, onClose, onCreated }: GroupDialogProps) {
  const b = useBots()
  const meta = useValue($botMeta)
  const [name, setName] = useState('')
  const connectionId = useValue(host.state.connectionId)
  const roster: RosterRow[] = useValue($lastRoster)
  const current = botGroups(botRosterMeta(bot, meta))
  const groups = knownGroups(meta)
  const botName = displayName(bot, botRosterMeta(bot, meta))
  // Its groups[] also live on its own machine: a change saved while that
  // machine is offline would land only here and be undone on reconnect.
  const row = roster.find(candidate => botRosterKey(candidate) === botRosterKey(bot)) || bot
  const locked = memberMachineOffline(row, connectionId)
  const [creating, setCreating] = useState<null | string>(null)

  if (creating !== null) {
    return <CreateGroupChatDialog onClose={onClose} onCreated={onCreated} open roster={roster} seed={{ bots: [row], name: creating }} />
  }

  const setMembership = async (group: string, enabled: boolean) => {
    try {
      await setGroupMembership(bot, group, enabled)
    } catch (error) {
      host.notify({ kind: 'error', message: error instanceof Error ? error.message : String(error) })

      return false
    }

    host.notify({ kind: 'info', message: enabled ? b.group.memberAdded(botName, group) : b.group.memberRemoved(botName, group) })

    return true
  }

  return (
    <Dialog
      onOpenChange={value => {
        if (!value) {
          onClose()
        }
      }}
      open={Boolean(bot)}
    >
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{b.group.manageTitle}</DialogTitle>
          <DialogDescription>{b.group.manageDesc}</DialogDescription>
        </DialogHeader>
        {groups.length ? (
          <div className="grid gap-1.5">
            {groups.map(group => {
              const enabled = current.includes(group)

              return (
                <label
                  className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-(--chrome-action-hover)"
                  key={group}
                >
                  <Checkbox
                    checked={enabled}
                    disabled={locked}
                    onCheckedChange={checked => void setMembership(group, checked === true)}
                  />
                  <span>{group}</span>
                </label>
              )
            })}
          </div>
        ) : null}
        {locked ? (
          <div className="text-xs text-(--ui-text-tertiary)" data-testid="group-dialog-locked">
            {b.crossGateway.groupsOffline(connectionName(bot), botName)}
          </div>
        ) : null}
        <form
          className="flex items-center gap-1.5"
          onSubmit={event => {
            event.preventDefault()
            const trimmed = name.trim()

            if (trimmed && !locked) {
              setCreating(trimmed)
            }
          }}
        >
          <Input
            autoFocus
            disabled={locked}
            onChange={event => setName(event.target.value)}
            placeholder={groups.length ? b.group.newGroup : b.group.namePlaceholder}
            value={name}
          />
          <Button disabled={locked || !name.trim()} size="sm" type="submit">
            {b.group.createAndJoin}
          </Button>
        </form>
        {current.length ? (
          <Button
            className="justify-self-start"
            disabled={locked}
            onClick={() =>
              void (async () => {
                // Sequential: each toggle patches groups[] from the current meta;
                // stop at the first one that did not land.
                for (const group of current) {
                  if (!(await setMembership(group, false))) {
                    return
                  }
                }
              })()
            }
            size="sm"
            variant="ghost"
          >
            {b.bot.removeFromAllGroups}
          </Button>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}

interface CreateGroupChatDialogProps {
  onClose: () => void
  onCreated?: (group: string, origin: CanonicalGroupOrigin) => void
  open: boolean
  roster: RosterRow[]
  /** The bots and name a draft opens with (Manage groups' Create & join). */
  seed?: { bots: RosterRow[]; name: string }
}

interface GroupPlanNoteProps {
  connectionId: null | string
  plan: GroupCreationPlan
  owner: string
  selected: RosterRow[]
}

/** Before creation: where the group chat will run, or exactly what is missing and on which
 *  computer. Every new group chat runs on a gateway; nothing falls back to this Desktop. */
function GroupPlanNote({ connectionId, plan, owner, selected }: GroupPlanNoteProps) {
  const b = useBots()
  const { engine, reason } = plan
  const where = plan.connection || owner
  const local = (plan.connectionId ?? connectionId) === 'local'

  // The dialog's status line already says the owner can't run group chats right now.
  if (engine === 'gateway' || reason === 'owner_unavailable') {return null}

  const text = engine === 'gateway-peer' ? b.crossGateway.gatewayPeer(owner, (connectionId ?? '') === 'local')
    : engine === 'checking' ? b.crossGateway.checkingPeers
      : engine === 'elsewhere' ? b.crossGateway.selectionElsewhere(connectionName(selected[0]))
        : groupPlanProblem(reason, where, local, b)

  return <p className="text-xs text-(--ui-text-secondary)" data-group-engine={engine} role="note">{text}</p>
}

function groupPlanProblem(reason: string | undefined, where: string, local: boolean, b: ReturnType<typeof useBots>): string {
  const place = local ? b.bot.thisDevice : where

  const words: Record<string, string> = {
    unreachable: b.crossGateway.peerCheckFailed(where, local),
    owner_update_required: b.crossGateway.ownerUpdateRequired(place),
    peer_owner_unavailable: b.crossGateway.ownerCannotSeatPeers(place),
    route_unavailable: b.crossGateway.peerRouteMissing,
    profile_unseatable: b.canonical.createOtherConnection,
    peer_gateway_changed: b.crossGateway.peerGatewayChanged(place)
  }

  return words[reason ?? ''] ?? peerSetupText(reason ?? '', place, b.canonical)
}

/** Discord-style group chat creation: pick 2+ routable bots via checkboxes
 *  (with search), name the group, create. Every group chat is created on and run
 *  by this connection's gateway, which keeps it running with Desktop closed; bots
 *  on other computers are seated through their own gateways. When that is not
 *  possible the dialog says what is missing on which computer and offers Retry. */
export function CreateGroupChatDialog({ open, roster, onClose, onCreated, seed }: CreateGroupChatDialogProps) {
  const { t } = useI18n()
  const b = useBots()
  const allMeta: Record<string, BotMeta> = useValue($botMeta)
  const [peerCheck, setPeerCheck] = useState(0)
  const { capability, connectionId, mode, observedRoute, seatable, source } = useGroupCapability(open, peerCheck)

  const intent = useRef<GroupCreationIntent>({})
  const [query, setQuery] = useState('')
  const [checked, setChecked] = useState<Record<string, boolean>>({})
  const [name, setName] = useState('')
  const [notice, setNotice] = useState('')
  // A notice speaks for the draft that asked; a late answer must not land on a newer one.
  const draft = useRef(0)

  const newDraft = () => {
    draft.current += 1
    setNotice('')
  }

  // Reset per open so a cancelled draft doesn't leak into the next one.
  // eslint-disable-next-line no-restricted-syntax -- Clears a dialog-owned request intent; no reactive state is copied.
  useEffect(() => {
    if (open) {
      intent.current = {}
      newDraft()
      setQuery('')
      setChecked(Object.fromEntries((seed?.bots ?? []).map(bot => [botRosterKey(bot), true])))
      setName(seed?.name ?? '')
    }
    // The seed is read once per open: a re-render must not wipe the person's picks.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // An outage placeholder or a bot the authority will not seat is never offered.
  const selectableRoster = seatableGroupBots(roster, { connectionId, mode, seatable })

  const selected = selectableRoster.filter(bot => checked[botRosterKey(bot)])
  const targets = usePeerTargets(selectableRoster, connectionId, open && groupSeatsPeers(capability?.snapshot), peerCheck)
  const plan = groupCreationEngine(mode, selected, connectionId, capability?.snapshot, targets)
  const engine = plan.engine

  const owner = selectableRoster.find(bot => botOnConnection(bot, connectionId))?.connectionLabel ||
    (connectionId === 'local' ? b.bot.thisDevice : connectionId || b.bot.thisDevice)

  const spansConnections = selectableRoster.some(bot => !botOnConnection(bot, connectionId))
  const visible: RosterRow[] = filterBots(selectableRoster, allMeta, query)
  const atCap = selected.length >= GROUP_CHAT_MAX_MEMBERS

  const placeholder = selected.length
    ? selected.map(bot => displayName(bot, botRosterMeta(bot, allMeta))).join(', ')
    : b.group.nameLabel

  const canCreate = selected.length >= 2 && Boolean(name.trim() || selected.length) &&
    ['gateway', 'gateway-peer'].includes(engine)

  const select = (bot: RosterRow, value: boolean) => {
    newDraft()
    setChecked(prev => ({ ...prev, [botRosterKey(bot)]: value }))
  }

  const creating = useRef(false)

  const create = async () => {
    if (creating.current) {return}
    creating.current = true
    const submitted = draft.current
    setNotice('')

    try {
    const entered = name.trim().slice(0, 64)

    if (selected.length < 2) {
      return
    }

    const route = observedRoute

    if (!route) {
      throw new Error(b.canonical.driverUnavailable)
    }

    const { plan: fresh, targets: freshTargets, sourceCurrent } = await revalidateGroupCreation(
      route, capability, source, selected, engine, b.canonical.driverUnavailable).catch(error => {
        setPeerCheck(value => value + 1)
        throw error
      })

    if (!sourceCurrent()) {throw new Error(b.canonical.driverUnavailable)}

    const members = durableGroupChatMembers(selected)

    // Keyed on the name as entered, blank included: the Bots' names that fill a blank one change
    // with the roster, so the sent name is chosen once, inside the frozen request.
    const fallback = placeholder.slice(0, 64)

    const created = fresh.engine === 'gateway-peer'
      ? await createGroupAcrossConnections(route, entered, members, freshTargets, intent.current, typed => typed || fallback)
      : { ...await createCanonicalGroup(route, entered, members, intent.current,
        typed => uniqueGroupChatName(typed || fallback, new Set(liveGroupChatNames()))), unjoined: [] }

    // Creation already succeeded; leave it on its owner without adopting a stale result.
    if (!sourceCurrent()) {return}
    intent.current = {}

    const key = registerCanonicalGroup(route, created.room, { created: true })
    refreshGatewayGroupRooms()
    onClose()
    onCreated?.(key, 'key')

    // The room is made either way; say who is missing so a seat that failed is never silent.
    if (created.unjoined.length) {
      host.notify({ kind: 'warning', message: b.canonical.createdNotJoined
        .replace('{names}', created.unjoined.map(({ member }) => canonicalMemberName(member)).join(', ')) })
    }
    } catch (error) {
      const next = canonicalGroupCreateNotice(error, b.canonical)

      if (!next) {
        host.notify({ kind: 'error', message: canonicalGroupCreateError(error, b.canonical) })
      } else if (draft.current === submitted) {
        setNotice(next)
      }
    } finally { creating.current = false }
  }

  return (
    <Dialog
      onOpenChange={value => {
        if (!value) {
          onClose()
        }
      }}
      open={open}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{b.group.newTitle}</DialogTitle>
          <DialogDescription>{b.crossGateway.pickBots(GROUP_CHAT_MAX_MEMBERS)} {b.crossGateway.gatewayMembership}</DialogDescription>
        </DialogHeader>
        {mode === 'unavailable' && <p role="status">{b.canonical.driverUnavailable}</p>}
        <GroupPlanNote connectionId={connectionId} owner={owner} plan={plan} selected={selected} />
        {engine === 'unavailable' &&
          <Button onClick={() => setPeerCheck(value => value + 1)} size="sm" variant="ghost">{t.common.retry}</Button>}
        {/* TODO(bot-mode-types): this search box never takes focus when the dialog
            opens — SearchField accepts no `autoFocus` prop and forwards no extra
            props, so the `autoFocus` that used to sit here was inert. */}
        <SearchField
          aria-label={b.group.searchToAdd}
          containerClassName="w-full"
          inputClassName="w-full"
          onChange={setQuery}
          placeholder={b.group.searchToAddPlaceholder}
          value={query}
        />
        {selected.length ? (
          <div className="flex flex-wrap gap-1">
            {selected.map(bot => (
              <Badge
                asChild
                className="rounded-full bg-(--chrome-action-hover) pl-2 pr-1.5 text-[0.6875rem] text-(--ui-text-secondary) transition-colors hover:text-foreground"
                key={botRosterKey(bot)}
                variant="muted"
              >
                <Tip label={b.group.removeFromSelection}>
                  <RowButton
                    onClick={() => select(bot, false)}
                  >
                    {displayName(bot, botRosterMeta(bot, allMeta))}
                    <Codicon className="text-[0.6rem]" name="close" />
                  </RowButton>
                </Tip>
              </Badge>
            ))}
          </div>
        ) : null}
        <div className="max-h-64 min-h-0 overflow-y-auto overscroll-contain">
          <div className="grid gap-0.5 pr-2">
            {visible.length ? (
              visible.map(bot => {
                const meta = botRosterMeta(bot, allMeta)
                const { shape, color, image } = botAppearance(bot.name, meta)
                const isChecked = Boolean(checked[botRosterKey(bot)])
                const disabled = !isChecked && atCap
                const currentGroups = botGroups(meta)

                return (
                  <label
                    className={cn(
                      'flex min-w-0 cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 transition-colors hover:bg-(--chrome-action-hover)',
                      disabled && 'cursor-not-allowed opacity-50'
                    )}
                    key={botRosterKey(bot)}
                  >
                    <BotFace
                      color={avatarColor(color, bot.name)}
                      image={image && !isBackfilledFacePng(image) ? image : null}
                      name={bot.name}
                      shape={shape}
                      size={24}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs text-foreground">{displayName(bot, meta)}</div>
                      <div className="truncate text-[0.625rem] text-(--ui-text-quaternary)">
                        {currentGroups.length
                          ? `@${botHandle(bot.name, bot)} · ${b.group.memberGroups(currentGroups.map(group => `“${group}”`).join(', '))}`
                          : `@${botHandle(bot.name, bot)}`}
                      </div>
                    </div>
                    {spansConnections && (
                      <Badge className="shrink-0 rounded-full px-1.5 text-[0.625rem]" data-device-badge variant="muted">
                        {bot.connectionLabel || (botOnConnection(bot, connectionId) ? b.bot.thisDevice : bot.connectionId)}
                      </Badge>
                    )}
                    <Checkbox
                      checked={isChecked}
                      disabled={disabled}
                      onCheckedChange={value => select(bot, Boolean(value))}
                    />
                  </label>
                )
              })
            ) : (
              <div className="px-1.5 py-3 text-center text-xs text-(--ui-text-tertiary)">
                {query.trim() ? b.group.noBotsMatch(query.trim()) : b.group.createBotFirst}
              </div>
            )}
          </div>
        </div>
        <div className="grid gap-2">
          <form
            onSubmit={event => {
              event.preventDefault()
              create()
            }}
          >
            <Input
              aria-label={b.group.nameLabel}
              maxLength={64}
              onChange={event => {
                setName(event.target.value)
                newDraft()
              }}
              placeholder={placeholder}
              value={name}
            />
          </form>
        </div>
        {notice && <p className="text-xs text-(--ui-text-secondary)" role="status">{notice}</p>}
        <DialogFooter>
          <Button onClick={onClose} variant="secondary">
            {t.common.cancel}
          </Button>
          <Button
            disabled={!canCreate}
            onClick={create}
          >{`${b.group.createAction}${selected.length ? ` (${selected.length})` : ''}`}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
