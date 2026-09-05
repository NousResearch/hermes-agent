/**
 * Bot Mode domain model.
 *
 * Derived from how the gateway's payloads are actually consumed, not from a
 * published schema — so almost everything is optional. A roster row can arrive
 * three ways (a rich `profiles.list` row from the active gateway, a thin
 * `host.agents()` union row from another registered connection, or an offline
 * "ghost" twin), and older gateways omit whole fields. Widening a field to
 * required is a claim that every one of those paths supplies it.
 */

import type { ClassicFileRef, ClassicTurn } from './classic-output'

/**
 * The compact age suffixes the sidebar's session rows render ("now", "m", "h",
 * "d"). Structural rather than an import of core's `Translations`, which the
 * plugin fence puts out of reach — `t.sidebar.row` satisfies it.
 */
export interface SidebarRowLabels {
  ageDay: string
  ageHour: string
  ageMin: string
  ageNow: string
}

/** Where a row came from when several connections contribute to one roster. */
export interface ProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  profile: string
  targetProfile: string
}

/**
 * A bot's one canonical forever-chat: the profile's session titled exactly
 * "Bot Chat". Resolved server-side by title and reported on every roster row —
 * there is deliberately no stored session-id pointer (see AGENTS.md).
 */
export interface CanonicalSession {
  /** Durable id, stable across reloads. */
  id?: string
  /** Compression-lineage tip — the live session a durable id currently maps to. */
  resolved_id?: string
  last_active?: number
  preview?: string
  root_title?: string
  title?: string
}

export interface SessionPreview {
  /** Unix seconds, not milliseconds. */
  last_active?: number
  preview?: string
}

/** Per-bot presentation state, persisted in the profile's `ui_meta`. */
export interface BotMeta {
  /** Which user-made section this bot is filed under (`user-sections.ts`).
   *  Membership lives on the BOT, not as a member list on the section: a bot
   *  can only be in one place, deleting a section cannot orphan anybody, and
   *  the assignment rides the same profile.yaml sync every other bot setting
   *  already uses — so sections follow the profile to another machine. */
  sectionId?: null | string
  color?: string
  /** Set when the user has customized the avatar, so defaults stop applying. */
  custom?: boolean
  description?: string
  groups?: string[]
  hidden?: boolean
  /** Data URL. Stripped before `profiles.configure`; travels via `set_asset`. */
  image?: null | string
  imageKind?: 'photo' | 'shape'
  /** Legacy single-group scalar, projected alongside `groups`. */
  group?: null | string
  pinned?: boolean
  shape?: string
  title?: string
  /** Creation timestamp in ms. Deliberately not copied when duplicating a bot. */
  created?: number
}

export interface RosterRow {
  name: string
  canonical_session?: CanonicalSession | null
  connectionId?: string
  connectionKind?: string
  connectionLabel?: string
  description?: string
  display_name?: string
  /** An offline twin of a selected bot, kept visible so the row doesn't vanish. */
  ghost?: boolean
  handle?: string
  /** Server member identity is independent of this Desktop's connection aliases. */
  hostedIdentity?: {
    installationId: string
    profile: string
    roomId: string
    memberId: string
  }
  has_avatar?: boolean
  last_session?: SessionPreview | null
  remoteSource?: boolean
  route?: ProfileRoute
  sourceError?: null | string
  sourceMissing?: boolean
  sourceReachable?: boolean | null
  sourceScoped?: boolean
  targetProfile?: string
  /** Nullable: the gateway sends `null` for a profile with no configured role,
   *  and the create form threads its own optional title through the same shape. */
  title?: null | string
  ui_meta?: Record<string, unknown> & { 'hermes-bots'?: BotMeta }
  /** Compare-and-swap revisions, per ui_meta key. */
  ui_meta_revisions?: Record<string, number>
  worker_session?: { last_active?: number } | null
}

/** A roster row reduced to what a group room needs to seat a member. */
export type GroupMember = Pick<
  RosterRow,
  | 'connectionId'
  | 'connectionKind'
  | 'connectionLabel'
  | 'display_name'
  | 'ghost'
  | 'handle'
  | 'hostedIdentity'
  | 'name'
  | 'remoteSource'
  | 'route'
  | 'sourceMissing'
  | 'sourceReachable'
  | 'sourceScoped'
  | 'targetProfile'
  | 'title'
>

export type AttachmentKind = 'file' | 'image' | 'pdf'

export interface Attachment {
  /** Non-secret, explicitly published producer reference. Never cache its bytes here. */
  classicExport?: ClassicFileRef
  /** Local composer/preview data URL. Durable replay omits raw bytes. */
  data?: string
  /** Opaque gateway identity after durable staging. */
  attachmentId?: string
  kind: AttachmentKind
  mime?: string
  name: string
  size?: number
  /** Window-local idempotency key retained when a failed upload restores the draft. */
  uploadId?: string
}

export type HostedMessageIdentity = Pick<NonNullable<GroupMember['hostedIdentity']>, 'roomId' | 'memberId'> & {
  profile?: string
}

export interface GroupMessageAuthor {
  kind: 'member' | 'user'
  name: string
  /** Connection label, present when the speaker lives on another machine. */
  source?: string
  /** Hosted event actor identity for display binding, never a dispatch route. */
  hostedIdentity?: HostedMessageIdentity
  /** Display-only evidence: weak inheritance or a durable unresolved actor conflict. */
  hostedIdentityEvidence?: 'mirror' | 'mirror-conflict' | 'replay-conflict'
}

export interface GroupMessage {
  /** Milliseconds. */
  at: number
  /** Stable gateway event identity after a hosted-room replay. */
  eventId?: string
  /** Hosted room scope, never a dispatch route. */
  roomId?: string
  /** Untrusted legacy display metadata; never proof of outgoing intent. */
  clientEventId?: string
  from: GroupMessageAuthor
  id?: string
  images?: Attachment[]
  /** Monotonic gateway order for hosted-room events. */
  seq?: number
  text: string
  /** Messages predating threading carry the sentinel thread `'legacy'`. */
  thread?: string
}

export interface GroupHold {
  at?: number
  noted?: boolean
}

export interface GroupChat {
  /** User-facing continuity choice. Missing records are classic Desktop rooms. */
  continuityMode?: 'desktop' | 'distributed' | 'gateway'
  /** Bumped to abandon in-flight member turns from a previous round. */
  epoch?: number
  holds?: Record<string, GroupHold>
  image?: null | string
  log: GroupMessage[]
  members?: GroupMember[]
  /** Immutable identity, so a rename doesn't fork the room. */
  roomId?: null | string
  running?: boolean
  /** Stable authority installation id for a gateway-hosted room. */
  hosted?: null | string
  /** The local Desktop connection that currently reaches the authority. */
  hostedConnectionId?: null | string
  /** Server-issued fencing epoch for the hosted authority. */
  hostedEpoch?: null | number
  /** Last contiguous hosted-room event sequence applied locally. */
  hostedSeq?: number
  /** Local groups.state verification; never accepted from or sent to ui_meta. */
  hostedMembersVerified?: boolean
  /** A conflicting display mirror requires a fresh authoritative membership read. */
  hostedMembersNeedRefresh?: boolean
  hostedStatus?: null | {
    canReconnect?: boolean
    canRetry?: boolean
    canStop?: boolean
    label: string
    reconnectMemberId?: string
    retryCommandId?: string
    state: string
    taskId?: string
  }
  /** Short, actionable continuity problem for the room surface. */
  continuityIssue?: null | string
  /** The immutable owner descriptor captured beside each plumbing session,
   *  keyed the same way as `sessions`. Partial: legacy records hold a bare
   *  `{ name }`, and the sweep re-validates the route before trusting one. */
  sessionOwners?: Record<string, Partial<RosterRow>>
  sessions?: Record<string, string | true>
  stranded?: Record<string, number | { before: number; thread?: string; classicTurn?: ClassicTurn }>
  syncRevision?: number
  /** Left behind when a room is disbanded, so sync can't resurrect it. */
  tombstone?: boolean
  /** Read when ordering rooms; no write site in the plugin today. */
  pinned?: boolean
  /** How far each `<thread>::<member>` has read into `log`. Required: unlike
   *  the gateway-sourced shapes above, a room record is plugin-owned — every
   *  writer (hydrate, server-sync merge, updateGroupChat, room reset) seeds
   *  the map, and the turn engine indexes it unguarded. */
  watermarks: Record<string, number>
}

export type GroupPromptKind = 'approval' | 'clarify'

/**
 * One sub-question of a batch clarify, straight off the wire. `choices` and
 * `question` stay unknown because the card re-validates them; the two id
 * spellings are the keys it maps drafts and answers by.
 */
export interface GroupPromptQuestion {
  choices?: unknown
  id?: string
  multi_select?: boolean
  multiSelect?: boolean
  qid?: string
  question?: unknown
}

export interface GroupPrompt {
  at: number
  choices: string[]
  command?: string
  group: string
  /** Exact hosted-task identity when the prompt is owned by a Group Chat
   * authority rather than a visible member session. */
  hostedApproval?: {
    executionGeneration: number
    memberId: string
    roomId: string
    taskId: string
  }
  kind: GroupPromptKind
  member: string
  memberKey: string
  multiSelect: boolean
  question: string
  questions?: GroupPromptQuestion[] | null
  requestId: string
  sessionId?: null | string
}

export type GroupActivityKind =
  | 'cancelled'
  | 'capped'
  | 'delivered'
  | 'failed'
  | 'held'
  | 'passed'
  | 'queued'
  | 'replied'
  | 'settled'
  | 'stopped'
  | 'timed-out'
  | 'working'

export interface GroupActivityEvent {
  at: number
  group: string
  kind: GroupActivityKind
  member?: string
  preview?: string
}

/**
 * A cron job as Bot Mode reads it. Deliberately NOT the core `CronJob` type:
 * the gateway's `cron.manage` payload keys the id as `job_id`, carries the
 * schedule as a plain string rather than a structured object, and splits the
 * error into three separate fields. Reusing the core interface here would
 * typecheck against fields that never arrive.
 */
export interface RoutineJob {
  deliver?: string
  enabled?: boolean
  job_id: string
  last_delivery_error?: string
  last_fire_error?: string
  last_run_at?: string
  last_status?: string
  model?: string
  /** Prefixed `[bot:<slug>]` so the job can be scoped back to its bot. */
  name?: string
  next_run_at?: string
  paused_reason?: string
  prompt?: string
  prompt_preview?: string
  repeat?: number | string
  schedule?: string
  state?: string
  workdir?: string
}

export interface ConnectionRow {
  id: string
  label?: string
  primary?: boolean
}

export interface GatewaySource {
  connectionId: string
  count?: number
  error?: null | string
  kind?: string
  label?: string
  reachable?: boolean
}

export type AvatarShape = 'circle' | 'cloud' | 'drop' | 'hexagon' | 'pill' | 'squircle' | 'triangle'

export type BlobKind =
  'boxy' | 'capsule' | 'cloud' | 'droplet' | 'hexagon' | 'nub' | 'organic' | 'round' | 'sun' | 'triangle'

export type FaceMood = 'idle' | 'work'

export interface AvatarAppearance {
  /** `null` when nothing is picked — the name's deterministic hue stands in.
   *  `profileColor` returns null for the unnamed/default profile, so this has
   *  always been nullable in practice; `avatarColor` is what resolves it. */
  color: null | string
  image: null | string
  /** Free-form: a bare shape, `sigil-<n>`, a platonic solid, or `blobatar:<seed>:<kind>`. */
  shape: string
}

export type AttentionClass = 'agent_blocked' | 'missing_config' | 'provider_auth_or_access' | 'provider_quota_limit'

export type RosterKindFilter = 'all' | 'bots' | 'groups'
export type RosterActivityFilter = 'active' | 'all' | 'older' | 'recent'
