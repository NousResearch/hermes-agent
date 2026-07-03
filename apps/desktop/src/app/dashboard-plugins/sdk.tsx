import * as React from 'react'
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { registerDashboardPlugin } from './registry'
import type { DashboardPluginManifest } from './types'

const SESSION_HEADER = 'X-Hermes-Session-Token'
const SDK_CONTRACT_VERSION = '1.1.0-desktop'

type PluginComponent = React.ComponentType<Record<string, never>>
type PluginFetchInit = Omit<RequestInit, 'body'> & { body?: BodyInit | null | unknown }

interface DesktopDashboardPluginSdk {
  api: Record<string, (...args: never[]) => unknown>
  authedFetch: (url: string, init?: RequestInit) => Promise<Response>
  buildWsAuthParam: () => Promise<[string, string]>
  buildWsUrl: (path: string, params?: Record<string, string>) => Promise<string>
  components: Record<string, React.ComponentType<never>>
  fetchJSON: <T = unknown>(url: string, init?: PluginFetchInit) => Promise<T>
  hooks: {
    createContext: typeof React.createContext
    useCallback: typeof useCallback
    useContext: typeof useContext
    useEffect: typeof useEffect
    useMemo: typeof useMemo
    useRef: typeof useRef
    useState: typeof useState
  }
  React: typeof React
  sdkVersion: string
  useI18n: typeof useI18n
  utils: {
    cn: typeof cn
    isoTimeAgo: (iso: string) => string
    timeAgo: (timestamp: number | string) => string
  }
}

declare global {
  interface Window {
    __HERMES_PLUGINS__?: {
      register: (name: string, component: PluginComponent) => void
      registerSlot: (slot: string, name: string, component: React.ComponentType) => void
    }
    __HERMES_PLUGIN_SDK__?: DesktopDashboardPluginSdk
  }
}

export function dashboardPluginPath(name: string): string {
  return name.split('/').map(encodeURIComponent).join('/')
}

export function dashboardPluginAssetUrl(
  manifest: Pick<DashboardPluginManifest, 'name'>,
  filePath: string,
  baseUrl: string
): string {
  const cleanBase = baseUrl.replace(/\/+$/, '')
  const cleanFile = filePath.replace(/^\/+/, '')

  return `${cleanBase}/dashboard-plugins/${dashboardPluginPath(manifest.name)}/${cleanFile}`
}

export function dashboardPluginRequestPath(url: string): string {
  if (url.startsWith('/')) {
    return url
  }

  const parsed = new URL(url)

  return `${parsed.pathname}${parsed.search}`
}

function normalizeJsonBody(body: PluginFetchInit['body']): unknown {
  if (body === null || body === undefined) {
    return undefined
  }

  if (typeof body === 'string') {
    const trimmed = body.trim()

    if (!trimmed) {
      return undefined
    }

    try {
      return JSON.parse(trimmed)
    } catch {
      return body
    }
  }

  if (typeof URLSearchParams !== 'undefined' && body instanceof URLSearchParams) {
    return Object.fromEntries(body.entries())
  }

  if (typeof FormData !== 'undefined' && body instanceof FormData) {
    throw new Error('Dashboard plugin JSON requests cannot send FormData; use SDK.authedFetch instead.')
  }

  return body
}

export async function dashboardPluginFetchJSON<T = unknown>(url: string, init: PluginFetchInit = {}): Promise<T> {
  return window.hermesDesktop.api<T>({
    body: normalizeJsonBody(init.body),
    method: init.method,
    path: dashboardPluginRequestPath(url)
  })
}

function absoluteBackendUrl(url: string, baseUrl: string): string {
  return new URL(url, `${baseUrl.replace(/\/+$/, '')}/`).toString()
}

export async function dashboardPluginAuthedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const connection = await window.hermesDesktop.getConnection()
  const headers = new Headers(init.headers)

  if (connection.authMode !== 'oauth' && connection.token && !headers.has(SESSION_HEADER)) {
    headers.set(SESSION_HEADER, connection.token)
  }

  return fetch(absoluteBackendUrl(url, connection.baseUrl), {
    ...init,
    credentials: init.credentials ?? 'include',
    headers
  })
}

export async function dashboardPluginBuildWsAuthParam(): Promise<[string, string]> {
  const connection = await window.hermesDesktop.getConnection()

  if (connection.authMode === 'oauth') {
    const { ticket } = await window.hermesDesktop.api<{ ticket: string }>({
      method: 'POST',
      path: '/api/auth/ws-ticket'
    })

    return ['ticket', ticket]
  }

  return ['token', connection.token ?? '']
}

export async function dashboardPluginBuildWsUrl(path: string, params: Record<string, string> = {}): Promise<string> {
  const connection = await window.hermesDesktop.getConnection()
  const [authName, authValue] = await dashboardPluginBuildWsAuthParam()
  const url = new URL(path, `${connection.baseUrl.replace(/\/+$/, '')}/`)

  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'

  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value)
  }

  url.searchParams.set(authName, authValue)

  return url.toString()
}

function PluginCard({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn('rounded-lg border border-border bg-card text-card-foreground shadow-sm', className)}
      data-slot="dashboard-plugin-card"
      {...props}
    />
  )
}

function PluginCardContent({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('p-4', className)} data-slot="dashboard-plugin-card-content" {...props} />
}

function PluginLabel({ className, ...props }: React.ComponentProps<'label'>) {
  return <label className={cn('text-xs font-medium text-muted-foreground', className)} {...props} />
}

function PluginSelect({
  className,
  onChange,
  onValueChange,
  ...props
}: React.ComponentProps<'select'> & { onValueChange?: (value: string) => void }) {
  return (
    <select
      className={cn(
        'h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50',
        className
      )}
      onChange={event => {
        onChange?.(event)
        onValueChange?.(event.target.value)
      }}
      {...props}
    />
  )
}

function PluginSelectOption(props: React.ComponentProps<'option'>) {
  return <option {...props} />
}

function PluginSlot() {
  return null
}

function timeAgo(value: number | string): string {
  const numeric = typeof value === 'string' ? Date.parse(value) : Number(value)
  const timestamp = Number.isFinite(numeric) ? (numeric < 10_000_000_000 ? numeric * 1000 : numeric) : Date.now()
  const deltaSeconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000))

  if (deltaSeconds < 60) {return 'just now'}

  const minutes = Math.floor(deltaSeconds / 60)

  if (minutes < 60) {return `${minutes}m ago`}

  const hours = Math.floor(minutes / 60)

  if (hours < 24) {return `${hours}h ago`}

  const days = Math.floor(hours / 24)

  if (days < 30) {return `${days}d ago`}

  const months = Math.floor(days / 30)

  if (months < 12) {return `${months}mo ago`}

  return `${Math.floor(months / 12)}y ago`
}

function isoTimeAgo(iso: string): string {
  return timeAgo(iso)
}

export function exposeDesktopDashboardPluginSdk(): void {
  window.__HERMES_PLUGINS__ = {
    register: registerDashboardPlugin,
    registerSlot: () => undefined
  }

  window.__HERMES_PLUGIN_SDK__ = {
    React,
    api: {
      getPlugins: () => dashboardPluginFetchJSON('/api/dashboard/plugins')
    },
    authedFetch: dashboardPluginAuthedFetch,
    buildWsAuthParam: dashboardPluginBuildWsAuthParam,
    buildWsUrl: dashboardPluginBuildWsUrl,
    components: {
      Badge,
      Button: Button as React.ComponentType<never>,
      Card: PluginCard as React.ComponentType<never>,
      CardContent: PluginCardContent as React.ComponentType<never>,
      Checkbox: Checkbox as React.ComponentType<never>,
      Input: Input as React.ComponentType<never>,
      Label: PluginLabel as React.ComponentType<never>,
      PluginSlot: PluginSlot as React.ComponentType<never>,
      Select: PluginSelect as React.ComponentType<never>,
      SelectOption: PluginSelectOption as React.ComponentType<never>
    },
    fetchJSON: dashboardPluginFetchJSON,
    hooks: {
      createContext: React.createContext,
      useCallback,
      useContext,
      useEffect,
      useMemo,
      useRef,
      useState
    },
    sdkVersion: SDK_CONTRACT_VERSION,
    useI18n,
    utils: { cn, isoTimeAgo, timeAgo }
  }
}
