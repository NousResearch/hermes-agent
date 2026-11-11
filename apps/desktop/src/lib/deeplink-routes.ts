import type { PluginInstallLegacyHint } from '@/store/plugin-install-request'

export interface DeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

export type DeepLinkAction =
  | { type: 'plugin-install'; repo: string; enable: boolean; force: boolean; legacyHint: PluginInstallLegacyHint }
  /** `hermes://plugin/install?catalog=<name>` — resolved against the curated
   *  catalog by the caller; the raw name is never treated as a git identifier. */
  | { type: 'plugin-catalog-install'; name: string }
  | { type: 'skill-install'; identifier: string }
  | { type: 'composer-blueprint'; name: string; params: Record<string, string> }
  | { type: 'connection-done'; op: string; status: string }
  /** `hermes://chat/new?prompt=<text>` — a fresh chat whose composer is seeded
   *  with `prompt`, never sent. An empty/absent prompt is still a new chat. */
  | { type: 'chat-new'; prompt: string }
  | { type: 'ignore' }

/** Composer seed ceiling for a `hermes://chat/new` prompt.
 *
 *  A deep link is an OS argv string, so an unbounded value is somebody else's
 *  payload budget, not a prompt. Generous enough for a real paragraph of task
 *  text (every `switch.command`-sized ask on the website is two orders of
 *  magnitude below it) and small enough that the composer stays a composer. */
export const CHAT_NEW_PROMPT_MAX_CHARS = 4000

/**
 * Normalize an inbound prompt for the composer.
 *
 * The link crosses an OS boundary, so treat the text as hostile input that
 * happens to be readable:
 *  - CR/LF are folded to `\n` so a Windows-origin link doesn't paint `\r`.
 *  - C0/C1 control characters are dropped. They cannot be typed, and the one
 *    that matters is U+001B: the composer feeds terminal context blocks, and
 *    the bracketed-paste sanitizer downstream reads escape sequences.
 *  - U+2028/U+2029 become newlines (they are line breaks that no editor shows).
 *  - Everything is trimmed and hard-capped.
 *
 * Deliberately NOT stripped: ordinary Unicode, emoji, RTL text and combining
 * marks. The value is shown in the composer for the user to read and send, not
 * executed, so mangling legitimate prose would be the worse failure.
 */
export function normalizeChatNewPrompt(raw: string | undefined): string {
  if (!raw) {
    return ''
  }

  return (
    raw
      .replace(/\r\n?/g, '\n')
      .replace(/[\u2028\u2029]/g, '\n')
      // eslint-disable-next-line no-control-regex -- the point is to remove these
      .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, '')
      .trim()
      .slice(0, CHAT_NEW_PROMPT_MAX_CHARS)
      .trim()
  )
}

function truthyParam(value: string | undefined, defaultValue = false): boolean {
  if (value === undefined || value === '') {
    return defaultValue
  }

  const normalized = value.trim().toLowerCase()

  return normalized === '1' || normalized === 'true' || normalized === 'yes'
}

/** Links claimed by an exact `kind/name`, checked before the plugin and skill
 *  rules below. Add an exact route here rather than as another branch in
 *  `resolveDeepLinkAction`: the code-health ratchet holds that function at its
 *  complexity on main. */
const EXACT_ROUTES = new Map<string, (params: Record<string, string>) => DeepLinkAction>([
  // hermes://chat/new?prompt=… — the website's "Run in Hermes" buttons. Only
  // `new` is claimed: every other chat/<name> stays ignored so a future
  // chat/<id> resume shape is still free to be designed.
  ['chat/new', params => ({ type: 'chat-new', prompt: normalizeChatNewPrompt(params.prompt) })],
  // The browser leg of a connection came back (hermes://connections/done?op=…&status=…). The op id
  // names the operation to show; the status is carried but never moves a row, because the link is
  // whatever the user's browser was pointed at.
  [
    'connections/done',
    params => {
      const op = (params.op || '').trim()

      return op ? { type: 'connection-done', op, status: (params.status || '').trim() } : { type: 'ignore' }
    }
  ]
])

export function resolveDeepLinkAction(payload: DeepLinkPayload | null | undefined): DeepLinkAction {
  if (!payload?.kind) {
    return { type: 'ignore' }
  }

  if (payload.kind === 'blueprint' && payload.name) {
    return { type: 'composer-blueprint', name: payload.name, params: payload.params || {} }
  }

  const exact = EXACT_ROUTES.get(`${payload.kind}/${payload.name}`)

  if (exact) {
    return exact(payload.params || {})
  }

  // A `catalog` param claims the link outright: even when a `repo` rides along
  // (or the name is empty/bogus) the outcome is the catalog lookup's verdict,
  // never a git-path install of whatever else the link carried.
  if (payload.kind === 'plugin' && payload.name === 'install' && payload.params?.catalog !== undefined) {
    return { type: 'plugin-catalog-install', name: payload.params.catalog.trim() }
  }

  if (payload.kind === 'skill') {
    const identifier = payload.params?.identifier

    return payload.name === 'install' && identifier && identifier === identifier.trim()
      ? { type: 'skill-install', identifier }
      : { type: 'ignore' }
  }

  const repo = (
    payload.params?.repo ||
    payload.params?.identifier ||
    (payload.kind !== 'plugin' ? payload.name : '') ||
    ''
  ).trim()

  if (payload.kind === 'plugin' && payload.name === 'install' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: null
    }
  }

  if (payload.kind === 'plugin-agent' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: 'agent'
    }
  }

  if (payload.kind === 'plugin-desktop' && repo) {
    return {
      type: 'plugin-install',
      repo,
      enable: truthyParam(payload.params?.enable, true),
      force: truthyParam(payload.params?.force, false),
      legacyHint: 'desktop'
    }
  }

  return { type: 'ignore' }
}
