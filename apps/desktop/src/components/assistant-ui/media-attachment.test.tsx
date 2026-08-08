import { describe, expect, it } from 'vitest'

const createCache = () => {
  const srcCache = new Map<string, string>()
  const refCount = new Map<string, number>()

  const acquire = (path: string): string | undefined => {
    refCount.set(path, (refCount.get(path) ?? 0) + 1)
    return srcCache.get(path)
  }

  const store = (path: string, value: string) => {
    srcCache.set(path, value)
  }

  const release = (path: string) => {
    const count = (refCount.get(path) ?? 1) - 1
    if (count <= 0) {
      refCount.delete(path)
      srcCache.delete(path)
    } else {
      refCount.set(path, count)
    }
  }

  return { acquire, store, release, srcCache, refCount }
}

describe('MediaAttachment path-keyed cache', () => {
  it('survives remount: cache cleared after last consumer releases', () => {
    const { acquire, store, release, srcCache } = createCache()
    const first = acquire('image.png')
    expect(first).toBeUndefined()
    store('image.png', '/resolved/image.png')
    release('image.png')
    expect(srcCache.has('image.png')).toBe(false)
  })

  it('does not show previous src when path changes', () => {
    const { acquire, store, release, srcCache } = createCache()
    acquire('image-a.png')
    store('image-a.png', '/resolved/image-a.png')
    release('image-a.png')
    const cachedB = acquire('image-b.png')
    expect(cachedB).toBeUndefined()
    expect(srcCache.has('image-a.png')).toBe(false)
  })

  it('ref count: cache persists while multiple consumers hold path', () => {
    const { acquire, store, release, srcCache } = createCache()
    acquire('image.png')
    acquire('image.png')
    store('image.png', '/resolved/image.png')
    release('image.png')
    expect(srcCache.has('image.png')).toBe(true)
    release('image.png')
    expect(srcCache.has('image.png')).toBe(false)
  })
})
