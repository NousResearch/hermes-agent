import { useEffect, useMemo, useState } from 'react'

import {
  getDashboardPluginComponent,
  notifyDashboardPluginRegistry,
  setDashboardPluginLoadError
} from './registry'
import {
  dashboardPluginAssetUrl,
  exposeDesktopDashboardPluginSdk
} from './sdk'
import type { DashboardPluginManifest } from './types'

const loadedScriptUrls = new Set<string>()
const loadedStyleUrls = new Set<string>()

export interface DashboardPluginsState {
  error: Error | null
  loading: boolean
  manifests: DashboardPluginManifest[]
}

export function useDashboardPluginManifests(enabled = true): DashboardPluginsState {
  const [manifests, setManifests] = useState<DashboardPluginManifest[]>([])
  const [loading, setLoading] = useState(enabled)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!enabled) {
      setLoading(false)

      return
    }

    let cancelled = false

    setLoading(true)
    window.hermesDesktop
      .api<DashboardPluginManifest[]>({ path: '/api/dashboard/plugins' })
      .then(list => {
        if (cancelled) {return}
        setManifests(Array.isArray(list) ? list : [])
        setError(null)
      })
      .catch(err => {
        if (cancelled) {return}
        setError(err instanceof Error ? err : new Error(String(err)))
        setManifests([])
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [enabled])

  return { error, loading, manifests }
}

export function useDashboardPluginManifest(name: string, enabled = true) {
  const state = useDashboardPluginManifests(enabled)

  const manifest = useMemo(
    () => state.manifests.find(item => item.name === name) ?? null,
    [name, state.manifests]
  )

  return { ...state, manifest }
}

export interface DashboardPluginAssetState {
  error: Error | null
  loading: boolean
}

export function useDashboardPluginAssets(manifest: DashboardPluginManifest | null): DashboardPluginAssetState {
  const [loading, setLoading] = useState(Boolean(manifest && !getDashboardPluginComponent(manifest.name)))
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!manifest) {
      setLoading(false)

      return
    }

    if (getDashboardPluginComponent(manifest.name)) {
      setLoading(false)
      setError(null)

      return
    }

    let cancelled = false

    setLoading(true)
    setError(null)
    exposeDesktopDashboardPluginSdk()

    window.hermesDesktop
      .getConnection()
      .then(connection => loadDashboardPluginAssets(manifest, connection.baseUrl))
      .then(() => {
        if (cancelled) {return}
        setError(null)
        setLoading(false)
      })
      .catch(err => {
        if (cancelled) {return}
        const next = err instanceof Error ? err : new Error(String(err))
        setDashboardPluginLoadError(manifest.name, next.message || 'LOAD_FAILED')
        setError(next)
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [manifest])

  return { error, loading }
}

export async function loadDashboardPluginAssets(manifest: DashboardPluginManifest, baseUrl: string): Promise<void> {
  exposeDesktopDashboardPluginSdk()

  if (manifest.css) {
    loadPluginStylesheet(dashboardPluginAssetUrl(manifest, manifest.css, baseUrl), manifest.name)
  }

  if (getDashboardPluginComponent(manifest.name)) {
    return
  }

  const scriptUrl = dashboardPluginAssetUrl(manifest, manifest.entry, baseUrl)

  if (loadedScriptUrls.has(scriptUrl)) {
    notifyDashboardPluginRegistry()

    return
  }

  await loadPluginScript(scriptUrl, manifest.name)
}

function loadPluginStylesheet(url: string, name: string): void {
  if (loadedStyleUrls.has(url) || document.querySelector(`link[data-hermes-dashboard-plugin-css="${name}"]`)) {
    return
  }

  const link = document.createElement('link')
  link.dataset.hermesDashboardPluginCss = name
  link.href = url
  link.rel = 'stylesheet'
  document.head.appendChild(link)
  loadedStyleUrls.add(url)
}

function loadPluginScript(url: string, name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    const scriptUrl = import.meta.env.DEV ? `${url}${url.includes('?') ? '&' : '?'}hermes_desktop=${Date.now()}` : url

    script.async = true
    script.dataset.hermesDashboardPlugin = name
    script.src = scriptUrl

    script.onerror = () => {
      setDashboardPluginLoadError(name, 'LOAD_FAILED')
      reject(new Error(`Failed to load dashboard plugin ${name}`))
    }

    script.onload = () => {
      loadedScriptUrls.add(url)
      notifyDashboardPluginRegistry()
      queueMicrotask(() => {
        if (!getDashboardPluginComponent(name)) {
          setDashboardPluginLoadError(name, 'NO_REGISTER')
        }
      })
      resolve()
    }

    document.body.appendChild(script)
  })
}
