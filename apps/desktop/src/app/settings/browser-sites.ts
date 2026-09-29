/**
 * The browser's blocked-sites list (`security.website_blocklist`), edited from
 * Settings. The backend accepts bare hosts, URLs and `host/path`; the editor
 * stores what it enforces — a lowercase host without `www.` — so what the
 * person sees is what is matched.
 */

export interface WebsiteBlocklist {
  domains: string[]
  enabled: boolean
}

/** A host from whatever was typed or pasted; null when nothing usable is left. */
export function normalizeSiteRule(input: string): null | string {
  let value = input.trim().toLowerCase()

  if (!value || value.startsWith('#')) {
    return null
  }

  if (value.includes('://')) {
    try {
      value = new URL(value).hostname
    } catch {
      return null
    }
  }

  const host = value.split(/[/?#]/, 1)[0]?.replace(/^\*\./, '').replace(/^www\./, '').replace(/\.$/, '') ?? ''

  return /^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$/.test(host) ? host : null
}

export function readBlocklist(config: Record<string, unknown> | undefined): WebsiteBlocklist {
  const security = config?.security as Record<string, unknown> | undefined
  const raw = security?.website_blocklist as Record<string, unknown> | undefined
  const domains = Array.isArray(raw?.domains) ? raw.domains.filter((d): d is string => typeof d === 'string') : []

  return { domains, enabled: raw?.enabled === true }
}

/** Add a site, once; turns the list on as a side effect of adding to it. */
export function withSiteAdded(list: WebsiteBlocklist, input: string): WebsiteBlocklist {
  const host = normalizeSiteRule(input)

  if (!host) {
    return list
  }

  const known = new Set(list.domains.map(d => normalizeSiteRule(d)))

  return known.has(host) ? { ...list, enabled: true } : { domains: [...list.domains, host], enabled: true }
}

export function withSiteRemoved(list: WebsiteBlocklist, host: string): WebsiteBlocklist {
  return { ...list, domains: list.domains.filter(d => normalizeSiteRule(d) !== normalizeSiteRule(host)) }
}
