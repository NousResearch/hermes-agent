import { normalize } from '@/lib/text'

const SOURCE_LABELS: Record<string, string> = {
  a2a: 'A2A',
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

// External messaging platforms that each get their own self-managed sidebar
// section (fetched separately from local recents). Mirrors the gateway platform
// adapters; keep in sync with PLATFORM_ICONS in app/messaging/platform-icon.tsx.
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
const MESSAGING_SOURCE_IDS = new Set(MESSAGING_SESSION_SOURCE_IDS)

/** True when a source id is an external messaging platform (gets its own
 *  sidebar section) rather than a local/CLI/desktop session. */
export function isMessagingSource(source: null | string | undefined): boolean {
  const id = normalizeSessionSource(source)

  return id != null && MESSAGING_SOURCE_IDS.has(id)
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

/**
 * Session sources the Desktop recents slice hides by default because they are
 * machinery rather than conversations. `a2a` is the one with a user switch
 * (`sessions.recents_show_a2a`); the rest never come back.
 */
export const HIDDEN_RECENTS_MACHINE_SOURCES = ['a2a']

/**
 * Whether the user opted A2A peer dispatches back into the passive recents list.
 * Session search ignores this: searching is an explicit request, so A2A sessions
 * stay findable either way.
 */
export function recentsShowA2A(record: unknown): boolean {
  const sessions = (record as { sessions?: { recents_show_a2a?: unknown } } | null | undefined)?.sessions

  return sessions?.recents_show_a2a === true
}

/**
 * Read `sessions.exclude_sources` out of the raw config record: extra sources the
 * user wants hidden from recents, additive to the built-ins. A missing key, a
 * non-array value, and non-string entries all degrade to "nothing extra excluded"
 * rather than throwing inside the sidebar render path. Entries normalize the same
 * way session sources do, so `A2A` and `a2a` both match.
 */
export function configuredExcludeSources(record: unknown): string[] {
  const sessions = (record as { sessions?: { exclude_sources?: unknown } } | null | undefined)?.sessions
  const raw = sessions?.exclude_sources

  if (!Array.isArray(raw)) {
    return []
  }

  const ids = raw.map(value => normalizeSessionSource(typeof value === 'string' ? value : null))

  return Array.from(new Set(ids.filter((id): id is string => id != null)))
}

/**
 * Full exclusion list for the recents fetch: built-in machine sources (minus any
 * the user switched back on) plus configured extras. Returns the built-in array
 * identity when nothing is added, so an unchanged list never re-fetches.
 */
export function recentsExcludedSources(builtIn: string[], record: unknown): string[] {
  const machine = recentsShowA2A(record) ? [] : HIDDEN_RECENTS_MACHINE_SOURCES
  const extra = [...machine, ...configuredExcludeSources(record)]

  if (!extra.length) {
    return builtIn
  }

  return Array.from(new Set([...builtIn, ...extra]))
}
