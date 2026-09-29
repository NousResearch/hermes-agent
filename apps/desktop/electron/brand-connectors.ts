import { app, session } from 'electron'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import type { BrandSession } from './brand-scope'

const PORTAL_ORIGIN = 'https://admin.intelli-verse-x.ai'
const PARTITION = 'persist:ivx-portal'

export interface BrandConnectorCatalogEntry {
  id: string
  label: string
  mcpUrl: string
}

/** Official MCP address when the desktop can connect the tool itself. */
export const BRAND_CONNECTOR_CATALOG: BrandConnectorCatalogEntry[] = [
  { id: 'firecrawl', label: 'Firecrawl', mcpUrl: 'https://mcp.firecrawl.dev/v2/mcp' },
  { id: 'postiz', label: 'Postiz', mcpUrl: '' },
  { id: 'telnyx', label: 'Telnyx', mcpUrl: '' },
  { id: 'notifuse', label: 'Mail Studio', mcpUrl: '' },
  { id: 'chatwoot', label: 'Inbox Studio', mcpUrl: '' },
  { id: 'twenty', label: 'CRM', mcpUrl: '' },
  { id: 'fonoster', label: 'Voice Studio', mcpUrl: '' },
  { id: 'stripe', label: 'Stripe', mcpUrl: 'https://mcp.stripe.com' },
  { id: 'revenuecat', label: 'RevenueCat', mcpUrl: 'https://mcp.revenuecat.ai/mcp' },
  { id: 'appsflyer', label: 'AppsFlyer', mcpUrl: 'https://mcp.appsflyer.com/auth/mcp' },
  { id: 'beehiiv', label: 'Beehiiv', mcpUrl: '' },
  { id: 'n8n', label: 'n8n', mcpUrl: '' },
  { id: 'slack', label: 'Slack', mcpUrl: '' },
  { id: 'notion', label: 'Notion', mcpUrl: 'https://mcp.notion.com/mcp' },
  { id: 'linear', label: 'Linear', mcpUrl: 'https://mcp.linear.app/sse' }
]

export interface BrandConnectorView {
  connectorId: string
  label: string
  status: string
  mcpUrl: string
  credential: string
  hermesName: string
  source: 'web' | 'desktop' | 'both'
}

export interface BrandConnectorList {
  appId: string
  connectors: BrandConnectorView[]
  catalog: BrandConnectorCatalogEntry[]
  error: string
  webSaved: boolean
}

interface WebRow {
  connectorId: string
  label: string
  status: string
  mcpUrl: string
  credential: string
}

function catalogById(id: string): BrandConnectorCatalogEntry | undefined {
  return BRAND_CONNECTOR_CATALOG.find(entry => entry.id === id)
}

export function brandMcpServerName(appId: string, connectorId: string): string {
  const app = appId.trim().toLowerCase().replace(/[^a-z0-9-]/g, '')
  const connector = connectorId.trim().toLowerCase().replace(/[^a-z0-9-]/g, '')

  if (!app || !connector) {
    return ''
  }

  return `ivx-${app}-${connector}`
}

export function mergeBrandConnectors(appId: string, web: WebRow[], local: Record<string, string>): BrandConnectorView[] {
  const byId = new Map<string, BrandConnectorView>()

  for (const row of web) {
    const known = catalogById(row.connectorId)
    const localCredential = local[row.connectorId] || ''
    const credential = row.credential || localCredential
    const mcpUrl = row.mcpUrl || known?.mcpUrl || ''
    let source: BrandConnectorView['source'] = 'web'

    if (row.credential && localCredential) {
      source = 'both'
    } else if (!row.credential && localCredential) {
      source = 'desktop'
    }

    byId.set(row.connectorId, {
      connectorId: row.connectorId,
      label: row.label || known?.label || row.connectorId,
      status: row.status || (credential ? 'saved' : 'requested'),
      mcpUrl,
      credential,
      hermesName: credential && mcpUrl ? brandMcpServerName(appId, row.connectorId) : '',
      source
    })
  }

  for (const [connectorId, credential] of Object.entries(local)) {
    if (!credential || byId.has(connectorId)) {
      continue
    }

    const known = catalogById(connectorId)
    const mcpUrl = known?.mcpUrl || ''

    byId.set(connectorId, {
      connectorId,
      label: known?.label || connectorId,
      status: 'saved',
      mcpUrl,
      credential,
      hermesName: mcpUrl ? brandMcpServerName(appId, connectorId) : '',
      source: 'desktop'
    })
  }

  return [...byId.values()].sort((a, b) => a.label.localeCompare(b.label))
}

function storeFile(): string {
  return path.join(app.getPath('userData'), 'ivx-brand-mcp.json')
}

function readLocal(): Record<string, Record<string, string>> {
  try {
    const parsed = JSON.parse(readFileSync(storeFile(), 'utf8')) as unknown

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    const out: Record<string, Record<string, string>> = {}

    for (const [appId, connectors] of Object.entries(parsed)) {
      if (!connectors || typeof connectors !== 'object' || Array.isArray(connectors)) {
        continue
      }

      out[appId] = {}

      for (const [connectorId, credential] of Object.entries(connectors)) {
        if (typeof credential === 'string' && credential) {
          out[appId][connectorId] = credential
        }
      }
    }

    return out
  } catch {
    return {}
  }
}

function writeLocal(next: Record<string, Record<string, string>>): void {
  mkdirSync(path.dirname(storeFile()), { recursive: true })
  writeFileSync(storeFile(), JSON.stringify(next), 'utf8')
}

function parseRows(body: unknown): WebRow[] {
  if (!body || typeof body !== 'object') {
    return []
  }

  const data = (body as { data?: unknown }).data

  if (!Array.isArray(data)) {
    return []
  }

  const rows: WebRow[] = []

  for (const item of data) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const connectorId = String((item as { connectorId?: unknown }).connectorId || '')
      .trim()
      .toLowerCase()

    if (!connectorId) {
      continue
    }

    const credential = (item as { credential?: unknown }).credential

    rows.push({
      connectorId,
      label: typeof (item as { label?: unknown }).label === 'string' ? (item as { label: string }).label : '',
      status: typeof (item as { status?: unknown }).status === 'string' ? (item as { status: string }).status : '',
      mcpUrl: typeof (item as { mcpUrl?: unknown }).mcpUrl === 'string' ? (item as { mcpUrl: string }).mcpUrl : '',
      credential: typeof credential === 'string' ? credential : ''
    })
  }

  return rows
}

async function portalJson(method: string, urlPath: string, body?: unknown): Promise<{ ok: boolean; status: number; body: unknown }> {
  const response = await session.fromPartition(PARTITION).fetch(`${PORTAL_ORIGIN}${urlPath}`, {
    method,
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body)
  })
  const parsed = await response.json().catch(() => null)

  return { ok: response.ok, status: response.status, body: parsed }
}

function emptyList(appId: string, error = ''): BrandConnectorList {
  return { appId, connectors: [], catalog: BRAND_CONNECTOR_CATALOG, error, webSaved: false }
}

async function webRows(appId: string): Promise<{ rows: WebRow[]; error: string }> {
  const desktop = await portalJson('GET', `/api/portal/connectors/desktop?appId=${encodeURIComponent(appId)}`)

  if (desktop.ok) {
    return { rows: parseRows(desktop.body), error: '' }
  }

  if (desktop.status !== 404) {
    return { rows: [], error: desktop.status === 401 ? 'Sign in again to load this brand’s tools.' : 'Could not load this brand’s tools from the website.' }
  }

  const listed = await portalJson('GET', `/api/portal/connectors/requests?appId=${encodeURIComponent(appId)}`)

  if (!listed.ok) {
    return {
      rows: [],
      error: listed.status === 401 ? 'Sign in again to load this brand’s tools.' : 'Could not load this brand’s tools from the website.'
    }
  }

  return { rows: parseRows(listed.body), error: '' }
}

export async function loadBrandConnectors(brand: BrandSession): Promise<BrandConnectorList> {
  if (!brand.signedIn || !brand.activeAppId) {
    return emptyList(brand.activeAppId || '', brand.signedIn ? 'Choose a brand to see its tools.' : '')
  }

  const local = readLocal()[brand.activeAppId] || {}

  try {
    const web = await webRows(brand.activeAppId)

    return {
      appId: brand.activeAppId,
      connectors: mergeBrandConnectors(brand.activeAppId, web.rows, local),
      catalog: BRAND_CONNECTOR_CATALOG,
      error: web.error,
      webSaved: false
    }
  } catch {
    return {
      appId: brand.activeAppId,
      connectors: mergeBrandConnectors(brand.activeAppId, [], local),
      catalog: BRAND_CONNECTOR_CATALOG,
      error: 'Could not load this brand’s tools from the website.',
      webSaved: false
    }
  }
}

export async function saveBrandConnector(brand: BrandSession, connectorId: unknown, credential: unknown): Promise<BrandConnectorList> {
  if (!brand.signedIn || !brand.activeAppId) {
    return emptyList('', 'Sign in as a brand before saving a tool key.')
  }

  const id = typeof connectorId === 'string' ? connectorId.trim().toLowerCase() : ''
  const secret = typeof credential === 'string' ? credential.trim() : ''
  const known = catalogById(id)
  const current = await loadBrandConnectors(brand)

  if (!known) {
    return { ...current, error: 'Choose a tool from the list.', webSaved: false }
  }

  if (secret.length < 8 || secret.length > 8000) {
    return { ...current, error: 'Paste the key for this brand.', webSaved: false }
  }

  const all = readLocal()
  const forBrand = { ...(all[brand.activeAppId] || {}), [id]: secret }

  all[brand.activeAppId] = forBrand
  writeLocal(all)

  let webSaved = false
  let error = ''

  try {
    const saved = await portalJson('POST', '/api/portal/connectors/requests', {
      appId: brand.activeAppId,
      connectorId: id,
      authKind: 'api-key',
      credential: secret
    })

    webSaved = saved.ok

    if (!saved.ok) {
      error = 'Saved on this computer for this brand. The website did not keep the key.'
    }
  } catch {
    error = 'Saved on this computer for this brand. The website did not keep the key.'
  }

  const loaded = await loadBrandConnectors(brand)

  return { ...loaded, error: loaded.error || error, webSaved }
}
