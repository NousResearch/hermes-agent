export interface LinkTitleFetchDependencies {
  admitUrl(value: string): boolean
  cache: {
    get(key: string): string | undefined
    has(key: string): boolean
  }
  cacheKey(value: string): string
  fetchWithCurl(url: string): Promise<string>
  fetchWithRenderer(url: string): Promise<string>
  inflight: {
    delete(key: string): unknown
    get(key: string): Promise<string> | undefined
    set(key: string, value: Promise<string>): unknown
  }
  normalizeTitle(value: string): string
  storeCachedTitle(key: string, value: string): void
}

export function createLinkTitleFetcher(deps: LinkTitleFetchDependencies): (rawUrl: unknown) => Promise<string> {
  return rawUrl => {
    const url = String(rawUrl || '').trim()

    if (!deps.admitUrl(url)) {
      return Promise.resolve('')
    }

    const key = deps.cacheKey(url)

    if (!key) {
      return Promise.resolve('')
    }

    if (deps.cache.has(key)) {
      return Promise.resolve(deps.cache.get(key) ?? '')
    }

    const inflight = deps.inflight.get(key)

    if (inflight) {
      return inflight
    }

    const pending = deps
      .fetchWithCurl(url)
      .catch(() => '')
      .then(value => deps.normalizeTitle((value || '').slice(0, 240)))
      .then(async value => {
        if (value) {
          return value
        }

        const rendered = await deps.fetchWithRenderer(url).catch(() => '')

        return deps.normalizeTitle((rendered || '').slice(0, 240))
      })
      .then(clean => {
        deps.storeCachedTitle(key, clean)
        deps.inflight.delete(key)

        return clean
      })

    deps.inflight.set(key, pending)

    return pending
  }
}
