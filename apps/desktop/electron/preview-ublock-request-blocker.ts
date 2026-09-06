import fs from 'node:fs'
import path from 'node:path'

export interface PreviewUblockRequestDetails {
  id?: number
  method?: string
  referrer?: string
  resourceType?: string
  url: string
  webContentsId?: number
}

export interface PreviewUblockRequestBlockerSession {
  webRequest: {
    onBeforeRequest(listener: (details: PreviewUblockRequestDetails, callback: (response: { cancel: boolean }) => void) => void): void
    removeListener?(listener: (details: PreviewUblockRequestDetails, callback: (response: { cancel: boolean }) => void) => void): void
  }
}

interface DnrRule {
  action?: { type?: string }
  condition?: DnrCondition
  priority?: number
}

interface DnrCondition {
  domainType?: 'firstParty' | 'thirdParty'
  excludedInitiatorDomains?: string[]
  excludedRequestDomains?: string[]
  excludedResourceTypes?: string[]
  initiatorDomains?: string[]
  requestDomains?: string[]
  requestMethods?: string[]
  resourceTypes?: string[]
  urlFilter?: string
}

interface CompiledRule {
  action: 'allow' | 'block'
  condition: DnrCondition
  priority: number
  urlFilter: RegExp | null
}

interface PreviewUblockRuleIndex {
  domainRules: Map<string, CompiledRule[]>
  fallbackRules: CompiledRule[]
  trigramRules: Map<string, CompiledRule[]>
}

interface ExtensionManifest {
  declarative_net_request?: {
    rule_resources?: Array<{ enabled?: boolean; path?: string }>
  }
}

const RESOURCE_TYPE_ALIASES: Record<string, string> = {
  cspReport: 'csp_report',
  mainFrame: 'main_frame',
  subFrame: 'sub_frame',
  xmlhttprequest: 'xmlhttprequest',
  xhr: 'xmlhttprequest'
}

function normalizeHost(rawUrl: string | undefined): string | null {
  if (!rawUrl) {
    return null
  }

  try {
    return new URL(rawUrl).hostname.toLowerCase()
  } catch {
    return null
  }
}

function domainMatches(host: string | null, domains: string[] | undefined): boolean {
  if (!host || !domains?.length) {
    return false
  }

  return domains.some(domain => {
    const normalized = domain.toLowerCase()

    return host === normalized || host.endsWith(`.${normalized}`)
  })
}

function compileUrlFilter(filter: string): RegExp | null {
  if (!filter || filter.includes('\n') || filter.includes('\r')) {
    return null
  }

  let source = ''
  let offset = 0
  let startAnchored = false

  if (filter.startsWith('||')) {
    source += '^[a-z][a-z0-9+.-]*://(?:[^/]*\\.)?'
    offset = 2
    startAnchored = true
  } else if (filter.startsWith('|')) {
    source += '^'
    offset = 1
    startAnchored = true
  }

  for (; offset < filter.length; offset += 1) {
    const character = filter[offset]

    if (character === '*') {
      source += '.*'
    } else if (character === '^') {
      source += '(?:[^A-Za-z0-9_.%-]|$)'
    } else if (character === '|' && offset === filter.length - 1) {
      source += '$'
    } else {
      source += character.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&')
    }
  }

  try {
    return new RegExp(source || (startAnchored ? '^' : '.*'), 'i')
  } catch {
    return null
  }
}

function resourceType(details: PreviewUblockRequestDetails): string {
  const raw = details.resourceType ?? 'other'

  return RESOURCE_TYPE_ALIASES[raw] ?? raw
}

function ruleMatches(rule: CompiledRule, details: PreviewUblockRequestDetails): boolean {
  const condition = rule.condition
  const requestHost = normalizeHost(details.url)
  const initiatorHost = normalizeHost(details.referrer)
  const type = resourceType(details)

  if (condition.requestDomains && !domainMatches(requestHost, condition.requestDomains)) {
    return false
  }
  if (domainMatches(requestHost, condition.excludedRequestDomains)) {
    return false
  }
  if (condition.initiatorDomains && !domainMatches(initiatorHost, condition.initiatorDomains)) {
    return false
  }
  if (domainMatches(initiatorHost, condition.excludedInitiatorDomains)) {
    return false
  }
  if (condition.resourceTypes && !condition.resourceTypes.includes(type)) {
    return false
  }
  if (condition.excludedResourceTypes?.includes(type)) {
    return false
  }
  if (condition.requestMethods && !condition.requestMethods.includes((details.method ?? 'GET').toLowerCase())) {
    return false
  }
  if (condition.domainType) {
    const firstParty = Boolean(requestHost && initiatorHost && (requestHost === initiatorHost || requestHost.endsWith(`.${initiatorHost}`) || initiatorHost.endsWith(`.${requestHost}`)))

    if ((condition.domainType === 'firstParty') !== firstParty) {
      return false
    }
  }

  return rule.urlFilter?.test(details.url) ?? true
}

export function compilePreviewUblockRules(rules: DnrRule[]): CompiledRule[] {
  return rules.flatMap(rule => {
    const action = rule.action?.type

    if ((action !== 'allow' && action !== 'block') || !rule.condition) {
      return []
    }

    const urlFilter = rule.condition.urlFilter ? compileUrlFilter(rule.condition.urlFilter) : null

    if (rule.condition.urlFilter && !urlFilter) {
      return []
    }

    return [{ action, condition: rule.condition, priority: rule.priority ?? 1, urlFilter }]
  })
}

function filterTrigrams(filter: string | undefined): string[] {
  if (!filter) {
    return []
  }

  const trigrams = new Set<string>()

  for (const literal of filter.toLowerCase().split(/[|*^]+/)) {
    for (let index = 0; index + 3 <= literal.length; index += 1) {
      trigrams.add(literal.slice(index, index + 3))
    }
  }

  return [...trigrams]
}

function appendRule(index: Map<string, CompiledRule[]>, key: string, rule: CompiledRule): void {
  const existing = index.get(key)

  if (existing) {
    existing.push(rule)
  } else {
    index.set(key, [rule])
  }
}

function indexPreviewUblockRules(rules: CompiledRule[]): PreviewUblockRuleIndex {
  const domainRules = new Map<string, CompiledRule[]>()
  const trigramRules = new Map<string, CompiledRule[]>()
  const fallbackRules: CompiledRule[] = []
  const ruleTrigrams = new Map<CompiledRule, string[]>()
  const trigramFrequency = new Map<string, number>()

  for (const rule of rules) {
    const domains = rule.condition.requestDomains

    if (domains?.length) {
      for (const domain of domains) {
        appendRule(domainRules, domain.toLowerCase(), rule)
      }

      continue
    }

    const trigrams = filterTrigrams(rule.condition.urlFilter)

    if (trigrams.length === 0) {
      fallbackRules.push(rule)

      continue
    }

    ruleTrigrams.set(rule, trigrams)
    for (const trigram of trigrams) {
      trigramFrequency.set(trigram, (trigramFrequency.get(trigram) ?? 0) + 1)
    }
  }

  for (const [rule, trigrams] of ruleTrigrams) {
    let selected = trigrams[0]

    for (const trigram of trigrams.slice(1)) {
      if ((trigramFrequency.get(trigram) ?? Infinity) < (trigramFrequency.get(selected) ?? Infinity)) {
        selected = trigram
      }
    }

    appendRule(trigramRules, selected, rule)
  }

  return { domainRules, fallbackRules, trigramRules }
}

function candidatesForRequest(index: PreviewUblockRuleIndex, details: PreviewUblockRequestDetails): Set<CompiledRule> {
  const candidates = new Set(index.fallbackRules)
  const requestHost = normalizeHost(details.url)

  if (requestHost) {
    const labels = requestHost.split('.')

    for (let offset = 0; offset < labels.length; offset += 1) {
      for (const rule of index.domainRules.get(labels.slice(offset).join('.')) ?? []) {
        candidates.add(rule)
      }
    }
  }

  const normalizedUrl = details.url.toLowerCase()
  const seenTrigrams = new Set<string>()

  for (let offset = 0; offset + 3 <= normalizedUrl.length; offset += 1) {
    const trigram = normalizedUrl.slice(offset, offset + 3)

    if (seenTrigrams.has(trigram)) {
      continue
    }

    seenTrigrams.add(trigram)
    for (const rule of index.trigramRules.get(trigram) ?? []) {
      candidates.add(rule)
    }
  }

  return candidates
}

export function shouldBlockPreviewUblockRequest(
  rules: CompiledRule[] | PreviewUblockRuleIndex,
  details: PreviewUblockRequestDetails
): boolean {
  let winningRule: CompiledRule | null = null

  const candidates = Array.isArray(rules) ? rules : candidatesForRequest(rules, details)

  for (const rule of candidates) {
    if (!ruleMatches(rule, details)) {
      continue
    }

    if (
      !winningRule ||
      rule.priority > winningRule.priority ||
      (rule.priority === winningRule.priority && rule.action === 'allow' && winningRule.action === 'block')
    ) {
      winningRule = rule
    }
  }

  return winningRule?.action === 'block'
}

function safeRuleResourcePath(extensionPath: string, resourcePath: string): string | null {
  const root = path.resolve(extensionPath)
  const candidate = path.resolve(root, resourcePath.replace(/^\/+/, ''))

  return candidate.startsWith(`${root}${path.sep}`) ? candidate : null
}

export function loadPreviewUblockStaticRules(extensionPath: string): PreviewUblockRuleIndex | null {
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(extensionPath, 'manifest.json'), 'utf8')) as ExtensionManifest
    const resources = manifest.declarative_net_request?.rule_resources ?? []
    const rules: DnrRule[] = []

    for (const resource of resources) {
      if (resource.enabled !== true || typeof resource.path !== 'string') {
        continue
      }

      const filePath = safeRuleResourcePath(extensionPath, resource.path)

      if (!filePath) {
        return null
      }

      const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'))

      if (!Array.isArray(parsed)) {
        return null
      }

      rules.push(...(parsed as DnrRule[]))
    }

    return indexPreviewUblockRules(compilePreviewUblockRules(rules))
  } catch {
    return null
  }
}

export interface PreviewUblockRequestBlocker {
  dispose(): void
  getBlockedRequestCount(): number
  loadRules(extensionPath: string): boolean
  registerGuest(webContentsId: number, ownerWebContentsId: number): void
  setActive(active: boolean): void
  unregisterGuest(webContentsId: number): void
}

export function createPreviewUblockRequestBlocker({
  onBlocked,
  session
}: {
  onBlocked?: (details: Pick<PreviewUblockRequestDetails, 'id' | 'webContentsId'>) => void
  session: PreviewUblockRequestBlockerSession
}): PreviewUblockRequestBlocker {
  let active = false
  let blockedRequestCount = 0
  let disposed = false
  let rules: PreviewUblockRuleIndex | null = null
  const guestsByRequestWebContentsId = new Map<number, number>()
  const onBeforeRequest = (details: PreviewUblockRequestDetails, callback: (response: { cancel: boolean }) => void): void => {
    const previewGuestId =
      typeof details.webContentsId === 'number' ? guestsByRequestWebContentsId.get(details.webContentsId) : undefined
    const cancel =
      !disposed &&
      active &&
      typeof previewGuestId === 'number' &&
      rules !== null &&
      shouldBlockPreviewUblockRequest(rules, details)

    if (cancel) {
      blockedRequestCount += 1
      onBlocked?.({ id: details.id, webContentsId: previewGuestId })
    }

    callback({ cancel })
  }

  session.webRequest.onBeforeRequest(onBeforeRequest)

  return {
    dispose() {
      if (disposed) {
        return
      }

      disposed = true
      active = false
      blockedRequestCount = 0
      rules = null
      guestsByRequestWebContentsId.clear()
      session.webRequest.removeListener?.(onBeforeRequest)
    },
    getBlockedRequestCount() {
      return blockedRequestCount
    },
    loadRules(extensionPath) {
      const nextRules = loadPreviewUblockStaticRules(extensionPath)

      if (!nextRules) {
        return false
      }

      rules = nextRules
      // Electron permits only one listener per webRequest stage. Loading the
      // extension can replace the listener installed at construction, so
      // re-attach after every successful extension load.
      session.webRequest.onBeforeRequest(onBeforeRequest)

      return true
    },
    registerGuest(webContentsId, ownerWebContentsId) {
      guestsByRequestWebContentsId.set(webContentsId, webContentsId)
      guestsByRequestWebContentsId.set(ownerWebContentsId, webContentsId)
    },
    setActive(nextActive) {
      active = nextActive
    },
    unregisterGuest(webContentsId) {
      for (const [requestWebContentsId, guestId] of guestsByRequestWebContentsId) {
        if (guestId === webContentsId) {
          guestsByRequestWebContentsId.delete(requestWebContentsId)
        }
      }
    }
  }
}
