import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { getGhAuthStatus } from '@/hermes'
import { translateNow } from '@/i18n'
import { type ComposerSuggestion, registerDraftProvider } from '@/store/composer-suggestions'

/**
 * GitHub draft provider — the deliberate NON-MCP integration path.
 *
 * GitHub has no entry in the MCP catalog and never gets a connect pill: its
 * hosted MCP requires a per-host OAuth app (generic Dynamic Client
 * Registration 404s at /register), and more importantly the bundled
 * github/* skills driving the `gh` CLI are a strictly more capable
 * integration (PRs, reviews, issues, releases, workflows) than the remote
 * MCP's tool surface. So when the draft signals GitHub intent, the right
 * offer is onboarding onto the skills:
 *
 * - `gh` already authenticated → no pill. The skills just work; suggesting
 *   setup at an already-set-up user is noise.
 * - not authenticated (or gh missing) → offer the `github-auth` skill. The
 *   invoke prefixes the draft with `/github-auth` (same reversible,
 *   never-sends-on-your-behalf contract as the skill provider) and the
 *   agent walks the user through `gh auth login` / install.
 *
 * The auth probe is served by `GET /api/git/gh-auth` (cached backend-side);
 * a probe failure (older backend, transient error) suggests nothing rather
 * than nagging.
 */

const AUTH_TTL_MS = 5 * 60_000
const SKILL_NAME = 'github-auth'

let needsSetup: boolean | null = null
let checkedAt = 0

/** Drop the cached gh-auth probe (e.g. after the setup flow completes). */
export function invalidateGithubSuggestionIndex(): void {
  needsSetup = null
  checkedAt = 0
}

// Whole-word "github" mention or a pasted github.com link. Same completed-
// word discipline as the MCP provider: a keyword still under the caret is a
// word in progress, not intent — but a pasted host counts immediately.
const KEYWORD_RE = /(?<![\p{L}\p{N}])github(?![\p{L}\p{N}])(?=[\s\S])/iu
const HOST_RE = /https?:\/\/([^\s/,)\]}"'<>]*@)?([\w.-]*\.)?github\.com(?=[/\s:,)\]}"'<>]|$)/i

/** Pure matcher, exported for tests. */
export function githubHit(text: string): boolean {
  if (HOST_RE.test(text)) {
    return true
  }

  // Keyword matching runs with URLs removed: "github" inside a pasted
  // lookalike domain (notgithub.com, github.com.evil.example) is the URL's
  // business, and the host matcher above already rejected it.
  const withoutUrls = text.replace(/https?:\/\/[^\s]+/gi, ' ')
  const match = KEYWORD_RE.exec(withoutUrls.toLowerCase())

  // Completed word: at least one character follows the mention.
  return match !== null && match.index + 'github'.length < withoutUrls.length
}

async function loadNeedsSetup(): Promise<boolean> {
  if (needsSetup !== null && Date.now() - checkedAt < AUTH_TTL_MS) {
    return needsSetup
  }

  const status = await getGhAuthStatus()

  needsSetup = !status.authenticated
  checkedAt = Date.now()

  return needsSetup
}

function toSuggestion(): ComposerSuggestion {
  const copy = (key: string, ...args: unknown[]) => translateNow(`composer.githubSuggestions.${key}`, ...args)

  return {
    brand: 'github',
    doneLabel: copy('done'),
    doneTip: copy('doneTip'),
    id: SKILL_NAME,
    invoke: async () => {
      // Prefix, don't replace — and never send. The agent takes over when
      // the user sends: the github-auth skill installs gh if needed and
      // runs the device-code OAuth flow.
      requestComposerInsert(`/${SKILL_NAME} `, { mode: 'prefix' })
      requestComposerFocus()
    },
    label: copy('label'),
    provider: 'github',
    tip: copy('tip'),
    workingLabel: copy('label'),
    workingTip: copy('tip')
  }
}

registerDraftProvider('github', async ({ text }) => {
  const trimmed = text.trimStart()

  // Already a slash command (possibly ours from a previous click).
  if (trimmed.startsWith('/')) {
    return []
  }

  if (!githubHit(text)) {
    return []
  }

  if (!(await loadNeedsSetup())) {
    return []
  }

  return [toSuggestion()]
})
