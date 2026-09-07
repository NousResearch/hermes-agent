import {
  addMcpServer,
  authMcpServer,
  cancelMcpOAuthFlow,
  getActionStatus,
  getMcpCatalog,
  getMcpOAuthFlow,
  installMcpCatalogEntry,
  listMcpServers,
  type McpCatalogEntry,
  type McpRegistryEntry,
  type McpTestResult,
  removeMcpServer,
  setMcpServerEnabled,
  testMcpServer
} from '@/hermes'
import { completeMcpDesktopOAuth } from '@/lib/mcp-dashboard-oauth'
import { MCP_DIRECTORY } from '@/lib/mcp-directory'
import { classifyProbe } from '@/lib/mcp-probe-cache'
import { prettyName } from '@/lib/text'

/**
 * One connector vocabulary for the whole app.
 *
 * "Connector" is the user-facing noun for an MCP server — Linear, Notion,
 * Jira. Before this module every surface spoke a different dialect: the
 * consent card thought in catalog install actions, the composer pills in
 * directory entries, the Capabilities tab in raw `mcp_servers` rows. Adding a
 * fourth source (the public registry) to each of them separately would have
 * cemented that. Instead they all resolve a `Connector` here and act on its
 * state.
 *
 * Two ideas do most of the work:
 *
 * **A resolution ladder, precedence written down once.** Reviewed catalog →
 * curated vendor directory → public registry. Higher rungs win on name, so a
 * registry entry can never shadow a reviewed manifest, and a rung that fails
 * to answer falls through instead of failing the lookup.
 *
 * **State, not mechanism.** Callers ask "what is this connector's situation"
 * (missing / switched off / signed out / working) and `connect` does whatever
 * that situation requires. That is what lets a no-auth server be a plain
 * switch with no browser round-trip, while an OAuth one opens a tab — from
 * the same click, with no caller branching on transport or auth type.
 */

export type ConnectorTrust = 'catalog' | 'community' | 'verified'
export type ConnectorSource = 'catalog' | 'directory' | 'registry'
/** `unknown` means the source didn't say — resolved by probing, not guessing. */
export type ConnectorAuth = 'api_key' | 'none' | 'oauth' | 'unknown'
export type ConnectorState = 'connected' | 'disabled' | 'needs_auth' | 'not_configured'

export interface Connector {
  name: string
  title: string
  description: string
  url: null | string
  trust: ConnectorTrust
  source: ConnectorSource
  auth: ConnectorAuth
  /** Credentials the user must supply before install (names/prompts only). */
  requiredEnv: { name: string; prompt: string; required: boolean }[]
  /** Vendor docs or website, when the source knows one. */
  docs: string
  /** The product's own site, when the source names one explicitly. */
  homepage: string
  /** Registrable domain the registry publisher proved it owns, or "". */
  publisher: string
  /** Registry identity ("com.notion/mcp") — shown so two connectors that slug
   *  the same stay distinguishable. Empty for catalog/directory entries. */
  registryName: string
  /** Catalog entries that clone + build rather than just writing config. */
  needsInstall: boolean
  /** Ordered work only the user can do, elsewhere, before this can connect.
   *  Empty for the ordinary case where signing in is the whole setup. */
  setup: string[]
  /** What people call this connector when they aren't reading its name.
   *  Reused from the manifest's suggestion triggers. */
  keywords: string[]
}

/**
 * Match key for a connector name.
 *
 * The agent proposes connectors the way a person says them ("Unreal Engine",
 * "Hugging Face") while catalog names are slugs (`unreal-engine`,
 * `hugging_face`). Matching on lowercase alone drops those on the floor and
 * the card reports a reviewed connector as not found, so separators are
 * normalized on both sides of every comparison.
 */
export const connectorKey = (name: string): string =>
  name
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, '-')

/** The product name behind a slug: `unreal-engine` → "Unreal Engine". Every
 *  surface shows this, never the raw config key. */
export const connectorTitle = (name: string): string => prettyName(name.replace(/-/g, ' '))

/** Hosts whose favicon would name the wrong thing: a bridge published on
 *  GitHub is not GitHub, and a package page is not the product. */
const NOT_A_LOGO =
  /(^|\.)(github\.com|githubusercontent\.com|gitlab\.com|bitbucket\.org|npmjs\.com|pypi\.org|readthedocs\.io)$/

const isLoopback = (host: string) =>
  host === 'localhost' || host === '::1' || /^127\./.test(host) || /^(10|192\.168)\./.test(host)

/**
 * Where to read this connector's mark from.
 *
 * The product's own site first (the only source that is certainly the right
 * logo), then the endpoint it talks to — `mcp.linear.app` is Linear — then
 * vendor docs, which for most catalog entries is `docs.stripe.com` and friends.
 * A local endpoint and a code host answer with the wrong mark or none, so
 * they're skipped and the connector keeps its generic glyph.
 */
export function connectorLogoSource(connector: Partial<Pick<Connector, 'docs' | 'homepage' | 'url'>>): string {
  for (const candidate of [connector.homepage, connector.url, connector.docs]) {
    if (!candidate) {
      continue
    }

    try {
      const { hostname, origin } = new URL(candidate)

      if (!NOT_A_LOGO.test(hostname) && !isLoopback(hostname)) {
        // The origin, never the path: the icon lives on the site, and this
        // way a not-yet-connected connector's MCP endpoint is never fetched
        // just to draw its logo.
        return origin
      }
    } catch {
      // Not a URL (a bare repo path, a note) — nothing to read a mark from.
    }
  }

  return ''
}

const CATALOG_TTL_MS = 5 * 60_000

let catalogCache: { at: number; entries: McpCatalogEntry[] } | null = null

/** Drop memoized lookups (after an install, on profile switch). */
export function invalidateConnectorCache(): void {
  catalogCache = null
}

async function loadCatalog(): Promise<McpCatalogEntry[]> {
  if (catalogCache && Date.now() - catalogCache.at < CATALOG_TTL_MS) {
    return catalogCache.entries
  }

  const { entries } = await getMcpCatalog()

  catalogCache = { at: Date.now(), entries }

  return entries
}

function fromCatalog(entry: McpCatalogEntry): Connector {
  const auth = entry.auth_type === 'oauth' ? 'oauth' : entry.auth_type === 'api_key' ? 'api_key' : 'none'

  return {
    auth,
    description: entry.description,
    docs: entry.source,
    homepage: entry.homepage ?? '',
    keywords: entry.suggest?.keywords ?? [],
    name: entry.name,
    needsInstall: entry.needs_install,
    publisher: '',
    registryName: '',
    requiredEnv: entry.required_env,
    setup: entry.setup ?? [],
    source: 'catalog',
    title: connectorTitle(entry.name),
    trust: 'catalog',
    url: entry.url
  }
}

function fromRegistry(entry: McpRegistryEntry): Connector {
  // A documented secret header IS the credential requirement; anything else
  // is unknown until we probe, because the registry has no auth field.
  const secretHeaders = entry.headers.filter(header => header.secret)

  return {
    auth: secretHeaders.length > 0 ? 'api_key' : 'unknown',
    description: entry.description,
    docs: entry.website,
    homepage: entry.website,
    keywords: [],
    name: entry.name,
    needsInstall: false,
    publisher: entry.publisher,
    registryName: entry.registry_name,
    requiredEnv: secretHeaders.map(header => ({
      name: header.name,
      prompt: header.description || header.name,
      required: header.required
    })),
    setup: [],
    source: 'registry',
    title: entry.title || connectorTitle(entry.name),
    trust: entry.trust,
    url: entry.url
  }
}

/**
 * Resolve connector names down the ladder, in one pass.
 *
 * Unresolvable names are simply absent from the result — a connector the
 * agent invented shouldn't produce a broken row, and the card reports the
 * miss rather than offering a card that cannot work. Registry lookups run
 * only for names the two local rungs didn't answer, so the common case costs
 * one cached catalog read.
 */
/**
 * Find a reviewed connector by the name a person would say for it.
 *
 * The agent asks for "Unreal"; the manifest is `unreal-engine`. Exact match
 * first, then a segment-subset match in either direction, so "Unreal" and
 * "Hugging Face MCP" both land. An ambiguous near-match resolves to nothing
 * — connecting the wrong server is worse than reporting a miss — and this
 * only ever runs against the local reviewed rungs, never the registry.
 *
 * The last rung is the manifest's own keywords, which exist because a
 * connector's name is frequently not what anyone calls it: `google-workspace`
 * is the thing you reach for when you say "google docs", and no amount of
 * string surgery on the two names finds that. The manifest already lists
 * those phrases to trigger composer suggestions; they are just as true here.
 */
export function matchLocalConnector(wanted: string, entries: Connector[]): Connector | undefined {
  const exact = entries.find(entry => connectorKey(entry.name) === wanted)

  if (exact) {
    return exact
  }

  const parts = wanted.split('-').filter(Boolean)

  const near = entries.filter(entry => {
    const entryParts = connectorKey(entry.name).split('-').filter(Boolean)

    return entryParts.every(part => parts.includes(part)) || parts.every(part => entryParts.includes(part))
  })

  if (near.length === 1) {
    return near[0]
  }

  const byKeyword = entries.filter(entry => entry.keywords.some(keyword => connectorKey(keyword) === wanted))

  return byKeyword.length === 1 ? byKeyword[0] : undefined
}

export interface ConnectorResolution {
  /** Distinct connectors, in the order they were asked for. Two spellings of
   *  one connector ("Unreal", "Unreal Engine") collapse to a single entry. */
  connectors: Connector[]
  /** Names nothing answered to, spelled as the caller wrote them. */
  unresolved: string[]
}

export async function resolveConnectors(names: string[]): Promise<ConnectorResolution> {
  // Keyed by the normalized name but remembering how it was asked, because
  // the answer for a miss is shown to the user in their own words.
  const asked = new Map<string, string>()

  for (const name of names) {
    const key = connectorKey(name)

    if (key && !asked.has(key)) {
      asked.set(key, name)
    }
  }

  const wanted = [...asked.keys()]

  if (wanted.length === 0) {
    return { connectors: [], unresolved: [] }
  }

  const resolved = new Map<string, Connector>()
  const local = await listLocalConnectors().catch((): Connector[] => [])

  for (const name of wanted) {
    const hit = matchLocalConnector(name, local)

    if (hit) {
      resolved.set(name, hit)
    }
  }

  const missing = wanted.filter(name => !resolved.has(name))

  if (missing.length > 0) {
    const found = await Promise.all(missing.map(name => searchConnectors(name, 8).catch((): Connector[] => [])))

    missing.forEach((name, index) => {
      // Only an exact name hit counts here. The card is about to offer this
      // by name; silently substituting the registry's best fuzzy guess for
      // "notion" would connect something the agent never named.
      const match = found[index]?.find(candidate => connectorKey(candidate.name) === name)

      if (match) {
        resolved.set(name, match)
      }
    })
  }

  // Preserve the caller's order — it's the order the agent asked in — and
  // report the miss against the name that was asked, not the one we matched:
  // "Unreal" resolving to `unreal-engine` is a hit, not a miss.
  const connectors: Connector[] = []
  const seen = new Set<string>()

  for (const name of wanted) {
    const hit = resolved.get(name)

    if (hit && !seen.has(connectorKey(hit.name))) {
      seen.add(connectorKey(hit.name))
      connectors.push(hit)
    }
  }

  return {
    connectors,
    unresolved: wanted.filter(name => !resolved.has(name)).map(name => asked.get(name) ?? name)
  }
}

/** Every connector we can offer without a network round-trip: the reviewed
 *  catalog plus the curated vendor directory. This is the onboarding grid's
 *  default content — a first-run user should see recognizable apps, not an
 *  empty box waiting on a registry query they don't know to type into. */
export async function listLocalConnectors(): Promise<Connector[]> {
  const catalog = await loadCatalog().catch((): McpCatalogEntry[] => [])
  const connectors = catalog.map(fromCatalog)
  const taken = new Set(connectors.map(entry => entry.name.toLowerCase()))

  for (const entry of MCP_DIRECTORY) {
    if (!taken.has(entry.name.toLowerCase())) {
      connectors.push({
        auth: 'oauth',
        description: entry.description,
        docs: entry.docs,
        homepage: '',
        keywords: [],
        name: entry.name,
        needsInstall: false,
        publisher: '',
        registryName: '',
        requiredEnv: [],
        setup: [],
        source: 'directory',
        title: connectorTitle(entry.name),
        trust: 'verified',
        url: entry.url
      })
    }
  }

  return connectors.sort((a, b) => a.title.localeCompare(b.title))
}

/** Free-text connector search across catalog + registry, catalog first.
 *  Powers the onboarding grid's search box and off-catalog card resolution. */
export async function searchConnectors(query: string, limit = 12): Promise<Connector[]> {
  const needle = query.trim().toLowerCase()

  if (needle.length < 2) {
    return []
  }

  const { searchMcpRegistry } = await import('@/api/mcp')

  const [catalog, registry] = await Promise.all([
    loadCatalog().catch((): McpCatalogEntry[] => []),
    searchMcpRegistry(needle, limit)
      .then(response => response.entries)
      .catch((): McpRegistryEntry[] => [])
  ])

  const results: Connector[] = catalog
    .filter(entry => entry.name.toLowerCase().includes(needle) || entry.description.toLowerCase().includes(needle))
    .map(fromCatalog)

  const taken = new Set(results.map(entry => entry.name.toLowerCase()))

  for (const entry of registry) {
    if (!taken.has(entry.name.toLowerCase())) {
      taken.add(entry.name.toLowerCase())
      results.push(fromRegistry(entry))
    }
  }

  return results.slice(0, limit)
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/**
 * What this connector's situation is right now.
 *
 * Deliberately derived from config alone — cheap, synchronous, no probe.
 * "Configured and switched on" reads as `connected` even if its token expired
 * last night, because finding that out costs a real connection per row. The
 * agent's explicit `authorize` action is the override for the case where it
 * already knows the server answered 401.
 */
export function connectorState(name: string, servers: { name: string; enabled: boolean }[]): ConnectorState {
  const server = servers.find(candidate => candidate.name === name)

  if (!server) {
    return 'not_configured'
  }

  return server.enabled ? 'connected' : 'disabled'
}

export async function loadConnectorStates(names: string[]): Promise<Record<string, ConnectorState>> {
  const servers = await listMcpServers()
    .then(response => response.servers)
    .catch(() => [])

  return Object.fromEntries(names.map(name => [name, connectorState(name, servers)]))
}

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------

export type ConnectPhase = 'adding' | 'enabling' | 'installing' | 'probing' | 'signing_in'

export interface ConnectOptions {
  /** Polled at every boundary; true aborts and rolls back. */
  cancelled: () => boolean
  /** Narrates the flow so a row can say "Signing in…" rather than spinning. */
  onPhase?: (phase: ConnectPhase) => void
  /** Credential values for `requiredEnv`, collected by the caller's UI. */
  env?: Record<string, string>
}

export interface ConnectResult {
  /** Tool names the connector brought in, when the flow learned them. */
  tools: string[]
}

/** Cancel sentinel — callers swallow it rather than reporting a failure. */
export class ConnectorCancelled extends Error {
  constructor() {
    super('cancelled')
    this.name = 'ConnectorCancelled'
  }
}

/**
 * The connector is reachable and refusing us.
 *
 * Worth its own type because the answer is different in kind: a wrong URL or
 * a dead endpoint is retried, while a rejected credential needs the user to
 * grant something. A scope the OAuth grant never asked for lands here too —
 * the server is happy to talk, it just will not run the tools we want.
 */
export class ConnectorNeedsAuth extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ConnectorNeedsAuth'
  }
}

/**
 * Ask the connector for its tools, and treat silence as failure.
 *
 * A connect that only proves we wrote a config stanza is worth very little.
 * The key can be wrong, the endpoint can be down, an OAuth grant can be
 * missing the scope the tools actually need — and every one of those is
 * indistinguishable from success until the model calls a tool three messages
 * later and the user reads an error instead of an answer. So the paths that
 * used to return an empty, unexamined success end here instead, and the tool
 * names this returns are the receipt the card shows.
 */
async function verify(connector: Connector): Promise<ConnectResult> {
  const probe = await testMcpServer(connector.name).catch((error: unknown): McpTestResult => ({
    error: error instanceof Error ? error.message : String(error),
    ok: false,
    tools: []
  }))

  if (classifyProbe(probe) === 'needs-auth') {
    throw new ConnectorNeedsAuth(probe.error || `${connector.title} refused the credentials`)
  }

  if (!probe.ok) {
    throw new Error(probe.error || `${connector.title} did not answer`)
  }

  return { tools: probe.tools.map(tool => tool.name) }
}

/**
 * What a finished sign-in actually bought.
 *
 * The handshake reports the tool list when it can, and when it can't the
 * honest thing is to ask rather than to claim a connector that brought in
 * nothing. A probe that merely fails is not allowed to sink a sign-in that
 * genuinely completed — but a probe that comes back *unauthorized* is the
 * insufficient-scope case, where the grant went through and still doesn't
 * cover the tools, and that one has to reach the user.
 */
async function signedIn(connector: Connector, flow: { tools?: { name: string }[] }): Promise<ConnectResult> {
  const tools = (flow.tools ?? []).map(tool => tool.name)

  if (tools.length > 0) {
    return { tools }
  }

  return verify(connector).catch((error: unknown) => {
    if (error instanceof ConnectorNeedsAuth) {
      throw error
    }

    return { tools: [] }
  })
}

const CATALOG_INSTALL_POLL_MS = 1500

const oauth = (name: string, cancelled: () => boolean) =>
  completeMcpDesktopOAuth({
    cancel: cancelMcpOAuthFlow,
    cancelled,
    openExternal: url => window.hermesDesktop.openExternal(url),
    serverName: name,
    start: authMcpServer,
    status: getMcpOAuthFlow
  })

/**
 * Make a connector usable, whatever that takes.
 *
 * The interesting case is a connector whose auth requirement is `unknown` —
 * every registry entry without a documented secret header. Rather than
 * assuming OAuth and throwing the user at a browser tab that may 404 at
 * /register, the flow *probes*: add the server, try to list its tools, and
 * only fall through to sign-in if the endpoint actually refuses. A public
 * no-auth server therefore connects with a switch and no interruption, which
 * is the behavior that makes "connector" feel like one concept instead of
 * three.
 *
 * Failure and cancellation both roll the config write back. A declined flow
 * must leave no server behind — a half-configured entry squatting in
 * `mcp_servers` would fail every subsequent probe and look like a Hermes bug.
 */
export async function connectConnector(
  connector: Connector,
  state: ConnectorState,
  options: ConnectOptions
): Promise<ConnectResult> {
  const { cancelled, env = {}, onPhase } = options

  const abortIfCancelled = () => {
    if (cancelled()) {
      throw new ConnectorCancelled()
    }
  }

  if (state === 'disabled') {
    onPhase?.('enabling')
    await setMcpServerEnabled(connector.name, true)
    abortIfCancelled()
    onPhase?.('probing')

    // Enabling stands even if the probe then fails: the user asked for this
    // connector on, and switching it back off would hide the reason it isn't
    // working behind a control that looks untouched.
    return verify(connector)
  }

  if (state === 'needs_auth' || state === 'connected') {
    // Already in config — a retry, or an explicit re-auth. A failure here must
    // NOT remove the server the user already had.
    if (connector.auth === 'api_key' || connector.auth === 'none') {
      // A corrected credential has to be written before it can be tested, and
      // the install endpoint is what owns writing one. Skipping this is how a
      // wrong key becomes permanent: the card re-probes the same stored value
      // forever and every retry fails identically.
      if (connector.source === 'catalog' && Object.keys(env).length > 0) {
        onPhase?.('installing')
        await installMcpCatalogEntry(connector.name, env)
        abortIfCancelled()
      }

      // Nothing to sign into. The useful question is whether it answers now.
      onPhase?.('probing')

      return verify(connector)
    }

    onPhase?.('signing_in')
    const flow = await oauth(connector.name, cancelled)

    return signedIn(connector, flow)
  }

  // Not configured. Catalog entries with a git bootstrap or declared
  // credentials go through the reviewed install path; everything else is a
  // URL we write straight into config.
  if (connector.source === 'catalog' && (connector.needsInstall || connector.requiredEnv.length > 0)) {
    onPhase?.('installing')

    const response = await installMcpCatalogEntry(connector.name, env)

    if (response.background && response.action) {
      for (;;) {
        abortIfCancelled()

        const status = await getActionStatus(response.action, 1)

        if (!status.running) {
          if (status.exit_code !== 0) {
            throw new Error(`Install failed for ${connector.title}`)
          }

          break
        }

        await new Promise(resolve => setTimeout(resolve, CATALOG_INSTALL_POLL_MS))
      }
    }

    if (connector.auth === 'oauth') {
      onPhase?.('signing_in')
      const flow = await oauth(connector.name, cancelled)

      return signedIn(connector, flow)
    }

    abortIfCancelled()
    onPhase?.('probing')

    // The key the user just typed either works or it doesn't, and this is the
    // only moment they still have the context to fix it. No rollback on
    // failure: the install is reviewed and the credential may be one
    // character out, so a retry re-probes rather than starting over.
    return verify(connector)
  }

  if (!connector.url) {
    throw new Error(`${connector.title} has no endpoint to connect to`)
  }

  onPhase?.('adding')
  await addMcpServer({ name: connector.name, url: connector.url })

  try {
    if (connector.auth === 'none') {
      onPhase?.('probing')

      return verify(connector)
    }

    if (connector.auth === 'unknown') {
      // Probe before interrupting: plenty of hosted servers are open.
      onPhase?.('probing')
      const probe = await testMcpServer(connector.name)

      abortIfCancelled()

      if (probe.ok) {
        return { tools: probe.tools.map(tool => tool.name) }
      }
    }

    onPhase?.('signing_in')
    const flow = await oauth(connector.name, cancelled)

    return signedIn(connector, flow)
  } catch (error) {
    // Decline means "no connector", not an unauthorized entry left behind.
    // The rollback is best-effort and must never replace the original error:
    // a try/catch rather than `.catch()` so even a synchronous throw here
    // still surfaces the reason the connect actually failed.
    try {
      await removeMcpServer(connector.name)
    } catch {
      // Config still holds the entry; the real failure is the one below.
    }

    throw error
  }
}
