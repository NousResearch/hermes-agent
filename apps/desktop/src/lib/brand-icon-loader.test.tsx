import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { BRAND_ICON_HOSTNAMES } from './brand-icon'
import type { BrandIcon, BrandIconLoader } from './brand-icon-loader'
import {
  createBrandIconLoader,
  DEFERRED_BRAND_ICON_HOSTNAMES,
  hasBrandIconHost,
  useBrandIcon
} from './brand-icon-loader'

const GithubIcon = (() => null) as BrandIcon
const GoogleDocsIcon = (() => null) as BrandIcon

const resolver = {
  resolveBrandIcon: (host: string) => {
    if (host.endsWith('github.com')) {
      return GithubIcon
    }

    if (host === 'docs.google.com') {
      return GoogleDocsIcon
    }

    return null
  }
}

describe('hasBrandIconHost', () => {
  it('keeps the cheap admission table aligned with the deferred catalog', () => {
    expect(BRAND_ICON_HOSTNAMES.length).toBeGreaterThan(0)
    expect(DEFERRED_BRAND_ICON_HOSTNAMES.length).toBeGreaterThan(0)
    expect(new Set(DEFERRED_BRAND_ICON_HOSTNAMES)).toEqual(new Set(BRAND_ICON_HOSTNAMES))
  })

  it('admits exact and inherited branded hosts without importing the catalog', () => {
    expect(hasBrandIconHost('github.com')).toBe(true)
    expect(hasBrandIconHost('api.github.com')).toBe(true)
    expect(hasBrandIconHost('docs.google.com')).toBe(true)
  })

  it('rejects unknown hosts and bare public suffixes', () => {
    expect(hasBrandIconHost('example.com')).toBe(false)
    expect(hasBrandIconHost('localhost')).toBe(false)
    expect(hasBrandIconHost('com')).toBe(false)
  })
})

describe('createBrandIconLoader', () => {
  it('does not import the icon catalog for unknown hosts', async () => {
    const importer = vi.fn()
    const loader = createBrandIconLoader(importer)

    await expect(loader.load('example.com')).resolves.toBeNull()
    expect(importer).not.toHaveBeenCalled()
  })

  it('deduplicates concurrent imports and reuses the completed catalog', async () => {
    const importer = vi.fn(async () => resolver)
    const loader = createBrandIconLoader(importer)

    const [first, second] = await Promise.all([loader.load('github.com'), loader.load('api.github.com')])

    expect(first).toBe(GithubIcon)
    expect(second).toBe(GithubIcon)
    await expect(loader.load('docs.google.com')).resolves.toBe(GoogleDocsIcon)
    expect(importer).toHaveBeenCalledOnce()
  })

  it('does not re-request a catalog module URL that Chromium has already poisoned', async () => {
    const failure = new Error('chunk failed')
    const importer = vi.fn().mockRejectedValue(failure)
    const loader = createBrandIconLoader(importer)

    await expect(loader.load('github.com')).rejects.toBe(failure)
    await expect(loader.load('github.com')).rejects.toBe(failure)
    expect(importer).toHaveBeenCalledOnce()
  })
})

describe('useBrandIcon', () => {
  it('does not show a previous host icon while the next host is loading', () => {
    const loader: BrandIconLoader = {
      load: () => new Promise(() => undefined),
      peek: host => (host === 'github.com' ? GithubIcon : null)
    }

    const { result, rerender } = renderHook(({ host }) => useBrandIcon(host, loader), {
      initialProps: { host: 'github.com' }
    })

    expect(result.current).toBe(GithubIcon)

    rerender({ host: 'docs.google.com' })

    expect(result.current).toBeNull()
  })

  it('publishes the icon after a recognized host loads the deferred catalog', async () => {
    const importer = vi.fn(async () => resolver)
    const loader = createBrandIconLoader(importer)

    const { result, rerender } = renderHook(({ host }) => useBrandIcon(host, loader), {
      initialProps: { host: 'example.com' }
    })

    expect(result.current).toBeNull()
    expect(importer).not.toHaveBeenCalled()

    rerender({ host: 'github.com' })

    await waitFor(() => expect(result.current).toBe(GithubIcon))
    expect(importer).toHaveBeenCalledOnce()
  })
})
