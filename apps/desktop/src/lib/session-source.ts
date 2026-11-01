import { normalize } from '@/lib/text'

const SOURCE_LABELS: Record<string, string> = {
  acp: 'ACP',
  api_server: 'API',
  bluebubbles: 'iMessage',
  cli: 'CLI',
  codex: 'Codex',
  desktop: 'Desktop',
  discord: 'Discord',
  email: 'Email',
  gateway: 'Gateway',
  kanban: 'Kanban',
  local: 'Local',
  matrix: 'Matrix',
  mattermost: 'Mattermost',
  oneshot: 'One-shot',
  photon: 'Photon',
  qqbot: 'QQ',
  signal: 'Signal',
  slack: 'Slack',
  sms: 'SMS',
  telegram: 'Telegram',
  tui: 'TUI',
  webhook: 'Webhook',
  weixin: 'WeChat',
  whatsapp: 'WhatsApp',
  yuanbao: 'Yuanbao'
}

const SOURCE_ALIASES: Record<string, string[]> = {
  bluebubbles: ['apple messages', 'imessage'],
  photon: ['imessage', 'messages'],
  cli: ['terminal'],
  desktop: ['app', 'gui'],
  local: ['machine'],
  qqbot: ['qq'],
  telegram: ['tg'],
  tui: ['terminal'],
  weixin: ['wechat'],
  whatsapp: ['wa']
}

// Sources that run on the local machine rather than an external messaging
// platform. A handoff *from* one of these isn't a platform origin worth a badge.
// Exported so the recents fetch can keep these in the main list while the
// messaging fetch excludes them. `acp` runs as a local stdio process spawned
// by an editor, and its rows must never land in the messaging slice either.
export const LOCAL_SESSION_SOURCE_IDS = [
  'acp',
  'cli',
  'codex',
  'desktop',
  'gateway',
  'kanban',
  'local',
  'oneshot',
  'tui'
]
const LOCAL_SOURCE_IDS = new Set(LOCAL_SESSION_SOURCE_IDS)

// Sessions that never get their own sidebar section: everything that runs on
// THIS machine (CLI/TUI/desktop/editor surfaces, finite runs, automation
// bookkeeping) rather than arriving through a gateway platform adapter. This is
// the inverse side of `isMessagingSource`: a source outside the set is treated
// as an external platform, so plugin platforms and custom `--source` tags the
// label/icon catalogs have never heard of still group into their own section
// instead of burying their threads under the generic Sessions list (#67794).
// Union of the two ingest exclusions the sidebar already sends (recents +
// messaging slices): every id here was already barred from the platform path.
export const INTERNAL_SESSION_SOURCE_IDS = [...LOCAL_SESSION_SOURCE_IDS, 'cron', 'subagent', 'tool']
const INTERNAL_SOURCE_IDS = new Set(INTERNAL_SESSION_SOURCE_IDS)

// Platform ids the sidebar knows by name (recents SQL exclusion + label/icon
// catalogs). NOT the section gate: `isMessagingSource` decides platform
// membership, so a platform missing here still gets its own section — it just
// keeps arriving in the recents page until the client re-files it. Keep in sync
// with PLATFORM_ICONS in app/messaging/platform-icon.tsx where a brand mark
// exists.
export const MESSAGING_SESSION_SOURCE_IDS = [
  'telegram',
  'discord',
  'slack',
  'mattermost',
  'matrix',
  'signal',
  'whatsapp',
  'bluebubbles',
  'photon',
  'homeassistant',
  'email',
  'sms',
  'webhook',
  'api_server',
  'weixin',
  'wecom',
  'qqbot',
  'yuanbao',
  'dingtalk',
  'feishu'
]

/** True when a source id is an external platform session (gets its own sidebar
 *  section) rather than a local/CLI/desktop/automation session. Gateway
 *  adapters record `source = platform id`, so the test is exclusion-based: any
 *  source the internal set doesn't name is a platform, and an unrecognized one
 *  (a new plugin platform, a custom `--source` tag) still gets its own
 *  collapsible group instead of falling into the generic Sessions list
 *  (#67794). */
export function isMessagingSource(source: null | string | undefined): boolean {
  const id = normalizeSessionSource(source)

  return id != null && !INTERNAL_SOURCE_IDS.has(id)
}

export function normalizeSessionSource(source: null | string | undefined): string | null {
  return normalize(source) || null
}

/**
 * Resolve the origin messaging platform for a handed-off session. Returns the
 * normalized platform id (e.g. 'telegram') when the session completed a handoff
 * from a real messaging platform, otherwise null. After a handoff the live
 * source is local, so this is what drives the row's origin-platform badge.
 */
export function handoffOriginSource(
  handoffState: null | string | undefined,
  handoffPlatform: null | string | undefined
): string | null {
  if (handoffState !== 'completed') {
    return null
  }

  const id = normalizeSessionSource(handoffPlatform)

  if (!id || LOCAL_SOURCE_IDS.has(id)) {
    return null
  }

  return id
}

export function sessionSourceLabel(source: null | string | undefined): string | null {
  const id = normalizeSessionSource(source)

  if (!id) {
    return null
  }

  return SOURCE_LABELS[id] || id.replace(/[_-]+/g, ' ').replace(/\b\w/g, char => char.toUpperCase())
}

export function sessionSourceSearchTerms(source: null | string | undefined): string[] {
  const id = normalizeSessionSource(source)
  const label = sessionSourceLabel(id)

  if (!id) {
    return []
  }

  return [id, label ?? '', ...(SOURCE_ALIASES[id] ?? [])].filter(Boolean)
}
