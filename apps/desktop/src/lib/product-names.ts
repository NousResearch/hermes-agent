/**
 * Product names the client sees. Registered tool ids stay on the wire.
 * Same names as the portal: CRM, Inbox Studio, Mail Studio, and the rest.
 */
import { capitalize } from '@/lib/text'

interface ProductName {
  product: string
  /** Letters and digits only, so mcp__ and hyphenated server ids match. */
  token: string
}

const PRODUCTS: ProductName[] = [
  { token: 'chatwoot', product: 'Inbox Studio' },
  { token: 'inboxstudio', product: 'Inbox Studio' },
  { token: 'notifuse', product: 'Mail Studio' },
  { token: 'mailstudio', product: 'Mail Studio' },
  { token: 'fonoster', product: 'Voice Studio' },
  { token: 'voicestudio', product: 'Voice Studio' },
  { token: 'firecrawl', product: 'Firecrawl' },
  { token: 'gojiberry', product: 'Outreach' },
  { token: 'postiz', product: 'Social media' },
  { token: 'telnyx', product: 'SMS' },
  { token: 'n8n', product: 'n8n' },
  { token: 'automationstudio', product: 'Automation Studio' },
  { token: 'twenty', product: 'CRM' }
]

const VENDOR_LEAF = /^(?:chatwoot|notifuse|fonoster|firecrawl|gojiberry|postiz|telnyx|twenty|n8n|inbox|mail|voice|automation)[_-]?/i

const squash = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, '')

function productFor(value: string): string | null {
  const key = squash(value)

  if (!key) {
    return null
  }

  return PRODUCTS.find(row => key.includes(row.token))?.product ?? null
}

const VENDOR_WORD =
  /^(?:chatwoot|notifuse|fonoster|firecrawl|gojiberry|postiz|telnyx|twenty|n8n|inboxstudio|mailstudio|voicestudio|automationstudio)$/i

function actionPhrase(name: string): string {
  const segments = name
    .split('·')
    .map(part => part.trim())
    .filter(Boolean)
  const source = segments.length > 1 ? segments[segments.length - 1] : name
  const leaf = source.split('__').filter(Boolean).pop() ?? source
  let rest = leaf

  for (let pass = 0; pass < 4; pass += 1) {
    const next = rest.replace(/^ivx[_-][a-z0-9]+[_-]/i, '').replace(VENDOR_LEAF, '')

    if (next === rest) {
      break
    }

    rest = next
  }

  const words = rest
    .replace(/[_-]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(word => word && !/^(ivx|mcp)$/i.test(word) && !VENDOR_WORD.test(word))

  return words.map(word => capitalize(word.toLowerCase())).join(' ')
}

/** Chat label for one tool. Null when the name is not one of our products. */
export function productToolLabel(name: string): string | null {
  const raw = name.trim()
  const product = productFor(raw)

  if (!product) {
    return null
  }

  const action = actionPhrase(raw)

  return action ? `${product} · ${action}` : product
}

const PROSE: ReadonlyArray<readonly [RegExp, string]> = [
  [/\bchatwoot(?:[_-][a-z0-9_-]+)?\b/gi, 'Inbox Studio'],
  [/\bnotifuse(?:[_-][a-z0-9_-]+)?\b/gi, 'Mail Studio'],
  [/\bfonoster(?:[_-][a-z0-9_-]+)?\b/gi, 'Voice Studio'],
  [/\bfirecrawl(?:[_-][a-z0-9_-]+)?\b/gi, 'Firecrawl'],
  [/\bgojiberry(?:[_-][a-z0-9_-]+)?\b/gi, 'Outreach'],
  [/\bpostiz(?:[_-][a-z0-9_-]+)?\b/gi, 'Social media'],
  [/\btelnyx(?:[_-][a-z0-9_-]+)?\b/gi, 'SMS'],
  [/\btwenty[_-][a-z0-9_-]+\b/gi, 'CRM']
]

/** Strip vendor ids from text the chat is about to show. */
export function hideVendorNames(text: string): string {
  let next = String(text || '')

  next = next.replace(/\bmcp__[a-z0-9_]+(?:__[a-z0-9_]+)?\b/gi, match => productToolLabel(match) ?? match)
  next = next.replace(
    /\bivx-[a-z0-9-]+-(chatwoot|notifuse|fonoster|firecrawl|twenty|telnyx|n8n|gojiberry|postiz)\b/gi,
    match => productFor(match) ?? match
  )

  for (const [pattern, product] of PROSE) {
    next = next.replace(pattern, product)
  }

  return next
}
