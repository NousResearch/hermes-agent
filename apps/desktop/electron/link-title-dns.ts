import { lookup as dnsLookup } from 'node:dns'
import { isIP } from 'node:net'

export type LinkTitleAddress = { address: string; family: 4 | 6 }

export interface LinkTitlePinnedResolver {
  clear(): void
  resolve(hostname: string): Promise<readonly LinkTitleAddress[]>
}

interface CachedAddresses {
  addresses: readonly LinkTitleAddress[]
  expiresAt: number
}

type LinkTitleLookup = (hostname: string) => Promise<readonly LinkTitleAddress[]>

function lookupAll(hostname: string): Promise<readonly LinkTitleAddress[]> {
  return new Promise((resolve, reject) => {
    dnsLookup(hostname, { all: true, verbatim: true }, (error, addresses) => {
      if (error) {
        reject(error)

        return
      }

      resolve(
        addresses.map(value => ({
          address: value.address,
          family: value.family as 4 | 6
        }))
      )
    })
  })
}

function normalizeHostname(hostname: string): string {
  let normalized = hostname.trim().toLowerCase()

  if (normalized.startsWith('[') && normalized.endsWith(']')) {
    normalized = normalized.slice(1, -1)
  }

  if (normalized.endsWith('.')) {
    normalized = normalized.slice(0, -1)
  }

  if (!normalized) {
    throw new Error('Link title DNS hostname is empty')
  }

  return normalized
}

function immutableAddresses(addresses: readonly LinkTitleAddress[]): readonly LinkTitleAddress[] {
  return Object.freeze(addresses.map(value => Object.freeze({ ...value })))
}

function approveAnswers(
  answers: readonly LinkTitleAddress[],
  isPublicAddress: (value: string) => boolean
): readonly LinkTitleAddress[] {
  if (answers.length === 0) {
    throw new Error('Link title DNS returned an empty answer set')
  }

  if (answers.some(answer => !isPublicAddress(answer.address))) {
    throw new Error('Link title DNS returned a non-public address')
  }

  const approved: LinkTitleAddress[] = []
  const seen = new Set<string>()

  for (const answer of answers) {
    const key = `${answer.family}:${answer.address}`

    if (!seen.has(key)) {
      seen.add(key)
      approved.push(answer)
    }
  }

  return immutableAddresses(approved)
}

export function createLinkTitlePinnedResolver(options: {
  isPublicAddress(value: string): boolean
  lookup?: LinkTitleLookup
  maxEntries?: number
  now?: () => number
  ttlMs: number
}): LinkTitlePinnedResolver {
  const cache = new Map<string, CachedAddresses>()
  const inflight = new Map<string, Promise<readonly LinkTitleAddress[]>>()
  const lookup = options.lookup ?? lookupAll
  const maxEntries = Math.max(1, Math.floor(options.maxEntries ?? 512))
  const now = options.now ?? Date.now
  let generation = 0

  const pruneCache = (currentTime: number) => {
    for (const [hostname, cached] of cache) {
      if (currentTime >= cached.expiresAt) {
        cache.delete(hostname)
      }
    }

    while (cache.size > maxEntries) {
      const oldest = cache.keys().next().value

      if (oldest === undefined) {
        break
      }

      cache.delete(oldest)
    }
  }

  return {
    clear() {
      generation += 1
      cache.clear()
      inflight.clear()
    },
    async resolve(hostname) {
      const normalized = normalizeHostname(hostname)
      const family = isIP(normalized)

      if (family !== 0) {
        if (!options.isPublicAddress(normalized)) {
          throw new Error('Link title DNS literal is a non-public address')
        }

        return immutableAddresses([{ address: normalized, family: family === 4 ? 4 : 6 }])
      }

      const currentTime = now()
      pruneCache(currentTime)
      const cached = cache.get(normalized)

      if (cached) {
        cache.delete(normalized)
        cache.set(normalized, {
          addresses: cached.addresses,
          expiresAt: currentTime + options.ttlMs
        })

        return cached.addresses
      }

      const pending = inflight.get(normalized)

      if (pending) {
        return pending
      }

      const startedGeneration = generation

      const resolution = (async () => {
        const approved = approveAnswers(await lookup(normalized), options.isPublicAddress)

        if (generation === startedGeneration) {
          const resolvedAt = now()
          pruneCache(resolvedAt)

          while (cache.size >= maxEntries) {
            const oldest = cache.keys().next().value

            if (oldest === undefined) {
              break
            }

            cache.delete(oldest)
          }

          cache.set(normalized, { addresses: approved, expiresAt: resolvedAt + options.ttlMs })
        }

        return approved
      })()

      inflight.set(normalized, resolution)

      try {
        return await resolution
      } finally {
        if (inflight.get(normalized) === resolution) {
          inflight.delete(normalized)
        }
      }
    }
  }
}
