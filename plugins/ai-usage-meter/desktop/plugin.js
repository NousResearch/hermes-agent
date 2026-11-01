/**
 * ai-usage-meter — live provider quota and tracked-model usage for Hermes.
 * Plain ESM loaded uncompiled: jsx() calls only, no JSX syntax.
 */
import {
  Badge,
  Button,
  ConfirmDialog,
  EmptyState,
  ErrorState,
  host,
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  STATUSBAR_AREAS,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'
import React, { useEffect, useState, useMemo, useRef } from 'react'

const ID = 'ai-usage-meter'
const ALERT_THRESHOLD = 90
const ALERT_STORAGE_KEY = `${ID}:active-alerts`
const ORDER_STORAGE_KEY = `${ID}:provider-order`
const EXPECTED_BACKEND_VERSION = '0.2.0'
const MAX_RESTART_WAIT_MS = 20_000
const POLL_REQUEST_TIMEOUT_MS = 4_000
const AUTO_CHECK_INTERVAL_MS = 30_000

const COMPACT_FORMATTER = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  maximumFractionDigits: 1
})

const MONEY_FORMATTER = new Intl.NumberFormat(undefined, {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 4
})

const FALLBACK_STORE = {
  get: () => undefined,
  getValue: () => undefined,
  subscribe: () => () => {}
}

let rest
let sharedRefreshRevision = 0
let cachedUsageData = null
const cacheListeners = new Set()

function publishUsageData(nextData) {
  if (!nextData || typeof nextData !== 'object') return
  cachedUsageData = nextData
  cacheListeners.forEach(listener => {
    try {
      listener(nextData)
    } catch (err) {
      console.warn('Cache listener error', err)
    }
  })
}

function useCachedUsageData() {
  const [snapshot, setSnapshot] = useState(() => cachedUsageData)
  useEffect(() => {
    setSnapshot(cachedUsageData)
    cacheListeners.add(setSnapshot)
    return () => {
      cacheListeners.delete(setSnapshot)
    }
  }, [])
  return snapshot
}

const palette = {
  card: 'linear-gradient(135deg, #25262c 0%, #1c1d22 55%, #18191d 100%)',
  panel: 'rgba(255, 255, 255, 0.055)',
  panelStrong: 'rgba(255, 255, 255, 0.08)',
  border: 'rgba(255, 255, 255, 0.10)',
  borderStrong: 'rgba(255, 255, 255, 0.15)',
  text: '#f4f4f5',
  muted: '#a1a1aa',
  track: 'rgba(0, 0, 0, 0.38)',
  keyframes: `
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    @keyframes pulseDot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.92); }
    }
  `,
  limeCyan: 'linear-gradient(90deg, #bef264 0%, #4ade80 52%, #2dd4bf 100%)',
  rose: 'linear-gradient(90deg, #fb7185 0%, #f43f5e 52%, #fb923c 100%)',
  purple: 'linear-gradient(90deg, #a78bfa 0%, #c084fc 52%, #e879f9 100%)'
}

function safeRest(path) {
  if (typeof rest !== 'function') {
    return Promise.reject(new Error('Plugin REST client is not initialized yet.'))
  }
  return rest(path)
}

function withTimeout(promise, ms) {
  let timerId
  const timeout = new Promise((_, reject) => {
    timerId = setTimeout(() => reject(new Error(`Request timed out after ${ms}ms`)), ms)
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timerId))
}

function safeNotify(payload) {
  try {
    host?.notify?.(payload)
  } catch (err) {
    console.warn('Notification failed', err)
  }
}

function safeNavigate(path) {
  try {
    host?.navigate?.(path)
  } catch (err) {
    console.warn('Navigation failed', err)
  }
}

function backendNeedsRestart(data) {
  return Boolean(
    typeof data?.backend_version === 'string' &&
    data.backend_version.trim() !== '' &&
    data.backend_version !== EXPECTED_BACKEND_VERSION
  )
}

function toFiniteNumber(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') {
    return null
  }
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function toArray(value) {
  if (Array.isArray(value)) return value
  if (typeof value === 'string' && value.trim()) return [value]
  return []
}

function toDisplayString(value) {
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') {
    return String(value.message || value.detail || value.error || JSON.stringify(value))
  }
  return String(value)
}

function truncateText(value, maxLength = 240) {
  const str = toDisplayString(value)
  return str.length > maxLength ? `${str.slice(0, maxLength)}…` : str
}

function getProviderId(provider) {
  const raw = provider?.id ?? provider?.name
  if (raw !== undefined && raw !== null && String(raw).trim() !== '') {
    return String(raw).trim()
  }
  const source = toDisplayString(provider?.source || 'unknown')
  const plan = toDisplayString(provider?.plan || 'default')
  const modelCount = toArray(provider?.models).length
  return `provider:${source}:${plan}:${modelCount}`
}

function getWindowUsedPercent(window) {
  if (!window) return null
  const used = toFiniteNumber(window.used_percent)
  if (used !== null) return Math.max(0, Math.min(100, used))
  const remaining = toFiniteNumber(window.remaining_percent)
  if (remaining !== null) return Math.max(0, Math.min(100, 100 - remaining))
  return null
}

function getWindowRemainingPercent(window) {
  if (!window) return null
  const remaining = toFiniteNumber(window.remaining_percent)
  if (remaining !== null) return Math.max(0, Math.min(100, remaining))
  const used = toFiniteNumber(window.used_percent)
  if (used !== null) return Math.max(0, Math.min(100, 100 - used))
  return null
}

function hasUsageBar(provider) {
  if (!provider) return false
  return toArray(provider.windows).some(window => getWindowUsedPercent(window) !== null)
}

function loadSavedOrder() {
  try {
    const saved = globalThis.localStorage?.getItem(ORDER_STORAGE_KEY)
    if (!saved) return []
    const parsed = JSON.parse(saved)
    if (!Array.isArray(parsed)) return []
    return [...new Set(parsed.filter(item => typeof item === 'string' && item.trim() !== ''))]
  } catch (err) {
    console.warn('Failed to parse provider order', err)
    return []
  }
}

function saveOrderToStorage(order) {
  try {
    globalThis.localStorage?.setItem(ORDER_STORAGE_KEY, JSON.stringify(order))
  } catch (err) {
    console.warn('Failed to save provider order', err)
  }
}

const styles = {
  page: {
    height: '100%',
    maxHeight: 'calc(100vh - 24px)',
    overflowY: 'auto',
    overflowX: 'hidden',
    padding: '12px',
    boxSizing: 'border-box',
    color: palette.text
  },
  shell: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    paddingBottom: '20px'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '10px',
    flexWrap: 'wrap',
    flexShrink: 0
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap'
  },
  title: { margin: 0, fontSize: '20px', lineHeight: 1.2, fontWeight: 700, letterSpacing: '-0.02em' },
  autoCheckBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '11px',
    color: palette.muted,
    background: palette.panel,
    border: `1px solid ${palette.border}`,
    padding: '3px 10px',
    borderRadius: '999px'
  },
  pulseDot: {
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    background: '#34d399',
    animation: 'pulseDot 2s ease-in-out infinite'
  },
  controls: { display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' },
  statusPanel: {
    padding: '10px 14px',
    borderRadius: '12px',
    background: palette.panel,
    border: `1px solid ${palette.border}`,
    flexShrink: 0
  },
  statusHeaderRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    flexWrap: 'wrap',
    marginBottom: '8px'
  },
  card: {
    border: `1px solid ${palette.border}`,
    borderRadius: '18px',
    background: palette.card,
    padding: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
    minWidth: 0,
    height: '100%',
    boxSizing: 'border-box',
    boxShadow: '0 14px 34px rgba(0, 0, 0, 0.18)'
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    minHeight: '32px'
  },
  cardTitle: { margin: 0, fontSize: '14px', fontWeight: 700, letterSpacing: '-0.01em' },
  muted: { color: palette.muted, fontSize: '11px' },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
    gap: '10px',
    alignItems: 'stretch'
  },
  quotaSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    minHeight: '160px',
    flex: 1
  },
  window: {
    display: 'flex',
    flexDirection: 'column',
    gap: '3px',
    padding: '8px',
    border: `1px solid ${palette.border}`,
    borderRadius: '12px',
    background: palette.panel
  },
  row: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', minWidth: 0 },
  track: {
    width: '100%',
    height: '8px',
    overflow: 'hidden',
    borderRadius: '999px',
    background: palette.track,
    border: `1px solid ${palette.border}`,
    boxSizing: 'border-box',
    boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.28)'
  },
  modelList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    paddingTop: '6px',
    borderTop: '1px solid rgba(255,255,255,0.06)',
    marginTop: 'auto'
  },
  modelRows: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px'
  },
  modelRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '8px',
    minHeight: 16,
    padding: '1px 0',
    borderBottom: '1px dashed rgba(255,255,255,0.07)',
    fontSize: '10px',
    boxSizing: 'border-box'
  },
  unavailable: {
    minHeight: '146px',
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '12px',
    border: `1px dashed ${palette.borderStrong}`,
    borderRadius: '14px',
    color: palette.muted,
    fontSize: '11px',
    textAlign: 'center',
    boxSizing: 'border-box'
  },
  error: {
    border: '1px solid rgba(244, 63, 94, 0.55)',
    borderRadius: '10px',
    padding: '10px 12px',
    color: '#fb7185',
    fontSize: '12px',
    wordBreak: 'break-word',
    flexShrink: 0
  }
}

function compactNumber(value) {
  const safe = toFiniteNumber(value) ?? 0
  return COMPACT_FORMATTER.format(safe)
}

function money(value) {
  const safe = toFiniteNumber(value) ?? 0
  return MONEY_FORMATTER.format(safe)
}

function localTime(value) {
  if (value === null || value === undefined || value === '') return 'No reset reported'
  let parsedInput = value
  if (typeof value === 'number' || (typeof value === 'string' && /^\d+(\.\d+)?$/.test(value.trim()))) {
    const num = Number(value)
    if (Number.isFinite(num)) {
      parsedInput = num > 1e9 && num < 1e11 ? num * 1000 : num
    }
  }
  const date = new Date(parsedInput)
  return Number.isNaN(date.getTime()) ? 'Reset time unavailable' : date.toLocaleString()
}

function Gauge({ percent, scheme = 'limeCyan' }) {
  const numeric = toFiniteNumber(percent)
  const safe = numeric !== null ? Math.max(0, Math.min(100, numeric)) : 0
  const resolvedScheme = safe >= ALERT_THRESHOLD ? 'rose' : scheme
  const glow = resolvedScheme === 'rose'
    ? '0 0 12px rgba(244, 63, 94, 0.55)'
    : resolvedScheme === 'purple'
      ? '0 0 12px rgba(192, 132, 252, 0.48)'
      : '0 0 12px rgba(45, 212, 191, 0.48)'
  return jsx('div', {
    style: styles.track,
    role: 'progressbar',
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    'aria-valuenow': safe,
    children: jsx('div', {
      style: {
        width: `${safe}%`,
        height: '100%',
        borderRadius: '999px',
        background: palette[resolvedScheme] || palette.limeCyan,
        boxShadow: glow,
        transition: 'width 240ms ease'
      }
    })
  })
}

function SourceBadge({ value }) {
  const label = truncateText(value, 32) || 'UNKNOWN'
  const unavailable = label === 'UNAVAILABLE'
  return jsx(Badge, {
    variant: 'secondary',
    style: {
      color: unavailable ? '#fb7185' : '#d4d4d8',
      background: unavailable ? 'rgba(244, 63, 94, 0.12)' : palette.panel,
      border: `1px solid ${unavailable ? 'rgba(244, 63, 94, 0.24)' : palette.border}`
    },
    children: label
  })
}

function UsageWindow({ window, scheme }) {
  if (!window) return null
  const label = truncateText(window.label, 80) || 'Quota'
  const displayUsed = getWindowUsedPercent(window)
  const remaining = getWindowRemainingPercent(window)

  return jsxs('div', {
    style: styles.window,
    children: [
      jsxs('div', {
        style: styles.row,
        children: [
          jsx('strong', {
            style: { minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '12px' },
            title: label,
            children: label
          }),
          jsx('span', {
            style: { ...styles.muted, flexShrink: 0, whiteSpace: 'nowrap' },
            children: displayUsed !== null ? `${displayUsed.toFixed(1)}% used` : 'Usage unavailable'
          })
        ]
      }),
      jsx(Gauge, { percent: displayUsed, scheme }),
      jsx('div', {
        style: { ...styles.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
        title: [
          remaining !== null ? `${remaining.toFixed(1)}% remaining` : 'Remaining unavailable',
          localTime(window.reset_at),
          truncateText(window.detail, 160),
          window.scope ? `Scope: ${truncateText(window.scope, 40)}` : null
        ].filter(Boolean).join(' • '),
        children: `${remaining !== null ? `${remaining.toFixed(1)}% left` : 'Remaining unavailable'} • ${localTime(window.reset_at)}`
      })
    ]
  })
}

function ModelRow({ model }) {
  if (!model) return null
  const modelName = truncateText(model.name, 120) || 'Unknown model'
  const costSource = typeof model.cost_source === 'string' && model.cost_source.trim()
    ? model.cost_source.trim()
    : 'UNAVAILABLE'
  const costLabel = costSource === 'UNAVAILABLE'
    ? 'No cost recorded'
    : `${money(model.display_cost)} ${costSource.toLowerCase()}`
  const metadata = model.api_metadata
  const inputLimit = toFiniteNumber(metadata?.input_token_limit)
  const outputLimit = toFiniteNumber(metadata?.output_token_limit)
  const metadataLabel = (inputLimit !== null && inputLimit > 0) || (outputLimit !== null && outputLimit > 0)
    ? `${compactNumber(inputLimit)} input • ${compactNumber(outputLimit)} output`
    : null
  const matchedModels = toArray(model.matched_models).map(item => truncateText(item, 80)).filter(Boolean)

  return jsxs('div', {
    style: { ...styles.modelRow, minHeight: metadataLabel ? '52px' : styles.modelRow.minHeight },
    children: [
      jsxs('div', {
        style: { minWidth: 0, overflow: 'hidden' },
        title: [modelName, truncateText(metadata?.display_name, 80), ...matchedModels].filter(Boolean).join(' • '),
        children: [
          jsx('div', {
            style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 600 },
            children: modelName
          }),
          metadataLabel
            ? jsx('div', {
                style: { ...styles.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: '2px' },
                children: metadataLabel
              })
            : null
        ]
      }),
      jsx('div', {
        style: { textAlign: 'right', whiteSpace: 'nowrap' },
        title: costLabel,
        children: `${compactNumber(model.total_tokens)} tokens`
      })
    ]
  })
}

function ProviderCard({ provider, onMoveLeft, onMoveRight, isFirst, isLast }) {
  if (!provider) return null
  const scheme = provider.id === 'openai-codex' ? 'purple' : 'limeCyan'
  const visibleModels = toArray(provider.models)
    .filter(model => model && (toFiniteNumber(model.total_tokens) ?? 0) > 0)
    .sort((left, right) =>
      ((toFiniteNumber(right.total_tokens) ?? 0) - (toFiniteNumber(left.total_tokens) ?? 0)) ||
      String(left.name || '').localeCompare(String(right.name || ''))
    )
    .slice(0, 3)

  const windows = toArray(provider.windows).filter(window => getWindowUsedPercent(window) !== null)
  const details = toArray(provider.details).map(item => truncateText(item, 160)).filter(Boolean)
  const providerName = truncateText(provider.name || provider.id, 80) || 'Unknown Provider'

  return jsxs('section', {
    style: styles.card,
    children: [
      jsxs('div', {
        style: styles.cardHeader,
        children: [
          jsxs('div', {
            style: { display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: 0 },
            children: [
              jsx(Button, {
                size: 'sm',
                variant: 'ghost',
                disabled: isFirst,
                style: {
                  padding: '2px 4px',
                  fontSize: '10px',
                  color: palette.muted,
                  opacity: isFirst ? 0.35 : 1
                },
                onClick: onMoveLeft,
                children: '←'
              }),
              jsxs('div', {
                style: { display: 'flex', flexDirection: 'column', minWidth: 0 },
                children: [
                  jsx('h2', {
                    style: { ...styles.cardTitle, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                    title: providerName,
                    children: providerName
                  }),
                  provider.plan
                    ? jsx('div', { style: styles.muted, children: `Provider plan: ${truncateText(provider.plan, 60)}` })
                    : null
                ]
              }),
              jsx(Button, {
                size: 'sm',
                variant: 'ghost',
                disabled: isLast,
                style: {
                  padding: '2px 4px',
                  fontSize: '10px',
                  color: palette.muted,
                  opacity: isLast ? 0.35 : 1
                },
                onClick: onMoveRight,
                children: '→'
              })
            ]
          }),
          jsx(SourceBadge, { value: provider.source })
        ]
      }),
      jsxs('div', {
        style: styles.quotaSection,
        children: [
          windows.map((window, index) =>
            jsx(UsageWindow, {
              window,
              scheme,
              key: `${toDisplayString(window.label) || 'window'}-${index}`
            })
          ),
          details.length
            ? jsx('div', {
                style: { ...styles.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                title: details.join(' • '),
                children: details.join(' • ')
              })
            : null
        ]
      }),
      jsxs('div', {
        style: styles.modelList,
        children: [
          jsx('div', {
            style: { fontSize: '8px', color: '#a1a1aa', fontWeight: 600, letterSpacing: '0.03em', paddingBottom: '1px' },
            children: 'MODELS · 7 DAYS'
          }),
          jsx('div', {
            style: styles.modelRows,
            children: visibleModels.length
              ? visibleModels.map((model, index) =>
                  jsx(ModelRow, {
                    model,
                    key: `${toDisplayString(model.name) || 'model'}-${index}`
                  })
                )
              : jsx('div', {
                  style: { ...styles.muted, padding: '6px 0 2px', fontSize: '9px' },
                  children: 'No token usage in the last 7 days.'
                })
          })
        ]
      })
    ]
  })
}

function UsagePage() {
  const [refreshRevision, setRefreshRevision] = useState(() => sharedRefreshRevision)
  const [providerOrder, setProviderOrder] = useState(() => loadSavedOrder())
  const [isRestarting, setIsRestarting] = useState(false)
  const [isManualRefreshing, setIsManualRefreshing] = useState(false)
  const [restartWarningOpen, setRestartWarningOpen] = useState(false)
  const sharedCache = useCachedUsageData()
  const pollRef = useRef(null)
  const mountedRef = useRef(true)
  const forceRefreshRef = useRef(false)

  const stopPolling = () => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      stopPolling()
    }
  }, [])

  const query = useQuery({
    queryKey: [ID, 'usage', 30, refreshRevision],
    queryFn: async () => {
      const shouldForceRefresh = forceRefreshRef.current
      forceRefreshRef.current = false
      try {
        const result = await safeRest(`/usage?days=30&refresh=${shouldForceRefresh ? 'true' : 'false'}`)
        publishUsageData(result)
        return result
      } finally {
        if (mountedRef.current) {
          setIsManualRefreshing(false)
        }
      }
    },
    refetchInterval: AUTO_CHECK_INTERVAL_MS
  })

  useEffect(() => {
    if (query.data && typeof query.data === 'object') {
      publishUsageData(query.data)
    }
  }, [query.data])

  const activeCache = sharedCache || query.data
  const data = activeCache || { providers: [], errors: [], windows: [] }
  const rawProviders = useMemo(
    () => toArray(data.providers).filter(Boolean),
    [data.providers]
  )

  const sortedProviders = useMemo(() => {
    if (!rawProviders.length) return []
    if (!Array.isArray(providerOrder) || !providerOrder.length) return rawProviders

    return [...rawProviders].sort((a, b) => {
      const idA = getProviderId(a)
      const idB = getProviderId(b)
      const idxA = providerOrder.indexOf(idA)
      const idxB = providerOrder.indexOf(idB)
      if (idxA === -1 && idxB === -1) return 0
      if (idxA === -1) return 1
      if (idxB === -1) return -1
      return idxA - idxB
    })
  }, [rawProviders, providerOrder])

  const quotaProviders = useMemo(
    () => sortedProviders.filter(hasUsageBar),
    [sortedProviders]
  )

  const noQuotaProviders = useMemo(
    () => sortedProviders.filter(provider => !hasUsageBar(provider)),
    [sortedProviders]
  )

  const connectedNoQuotaCount = useMemo(
    () => noQuotaProviders.filter(provider => Boolean(provider.available)).length,
    [noQuotaProviders]
  )

  const offlineCount = noQuotaProviders.length - connectedNoQuotaCount

  const statusOrderedProviders = useMemo(
    () => [...quotaProviders, ...noQuotaProviders],
    [quotaProviders, noQuotaProviders]
  )

  if (query.isLoading && !activeCache) {
    return jsx('div', { style: styles.page, children: jsx(EmptyState, { title: 'Loading usage data' }) })
  }
  if (query.isError && !activeCache) {
    return jsx('div', {
      style: styles.page,
      children: jsx(ErrorState, {
        title: 'Usage data unavailable',
        description: query.error?.message || 'The plugin backend could not be reached.'
      })
    })
  }

  const bumpRefreshRevision = () => {
    forceRefreshRef.current = true
    setIsManualRefreshing(true)
    sharedRefreshRevision += 1
    setRefreshRevision(sharedRefreshRevision)
  }

  const updateOrder = (newOrder) => {
    const cleanOrder = [...new Set(newOrder.filter(Boolean))]
    setProviderOrder(cleanOrder)
    saveOrderToStorage(cleanOrder)
  }

  const moveQuotaProviderByIndex = (fromIndex, delta) => {
    const visibleIds = quotaProviders.map(p => getProviderId(p))
    const hiddenIds = noQuotaProviders.map(p => getProviderId(p))
    const toIndex = fromIndex + delta
    if (fromIndex < 0 || toIndex < 0 || toIndex >= visibleIds.length) return
    ;[visibleIds[fromIndex], visibleIds[toIndex]] = [visibleIds[toIndex], visibleIds[fromIndex]]
    updateOrder([...visibleIds, ...hiddenIds])
  }

  const refreshProviders = () => {
    if (backendNeedsRestart(data)) {
      setRestartWarningOpen(true)
      return
    }
    bumpRefreshRevision()
  }

  const restartGatewayAndRefresh = async () => {
    if (typeof host?.restartGateway !== 'function') {
      safeNotify({
        kind: 'error',
        title: 'Gateway restart unavailable',
        message: 'Gateway restart is unavailable in this Hermes Desktop version.'
      })
      setRestartWarningOpen(false)
      return
    }

    stopPolling()
    setIsRestarting(true)
    setRestartWarningOpen(false)

    try {
      await host.restartGateway()
    } catch (err) {
      console.info('Gateway restart initiated', err)
    }

    if (!mountedRef.current) return

    const startedAt = Date.now()
    let inFlight = false

    pollRef.current = setInterval(async () => {
      if (!mountedRef.current) return
      if (Date.now() - startedAt >= MAX_RESTART_WAIT_MS) {
        stopPolling()
        setIsRestarting(false)
        safeNotify({
          kind: 'error',
          title: 'Gateway reconnect timed out',
          message: 'Hermes Gateway did not respond in time. Please try refreshing again.'
        })
        return
      }
      if (inFlight) return
      inFlight = true

      try {
        const res = await withTimeout(safeRest('/usage?days=30&refresh=false'), POLL_REQUEST_TIMEOUT_MS)
        if (!mountedRef.current) return
        if (res && (!res.backend_version || res.backend_version === EXPECTED_BACKEND_VERSION)) {
          stopPolling()
          publishUsageData(res)
          setIsRestarting(false)
          bumpRefreshRevision()
        }
      } catch {
        // Expected while gateway is restarting
      } finally {
        inFlight = false
      }
    }, 1000)
  }

  const errorList = toArray(data.errors).map(item => truncateText(item, 300)).filter(Boolean)
  const isScanning = isManualRefreshing || (query.isLoading && !activeCache) || isRestarting
  const statusSummaryParts = [
    `${quotaProviders.length} with quota bars`,
    `${connectedNoQuotaCount} connected without quota (cards hidden)`
  ]
  if (offlineCount > 0) {
    statusSummaryParts.push(`${offlineCount} offline`)
  }

  return jsxs('div', {
    style: styles.page,
    children: [
      jsx('style', { children: palette.keyframes }),
      jsxs('main', {
        style: styles.shell,
        children: [
          jsxs('header', {
            style: styles.header,
            children: [
              jsxs('div', {
                style: styles.headerLeft,
                children: [
                  jsx('h1', { style: styles.title, children: 'AI Usage Meter' }),
                  jsxs('span', {
                    style: styles.autoCheckBadge,
                    children: [
                      jsx('span', { style: styles.pulseDot }),
                      jsx('span', { children: 'Auto-checking connections (30s)' })
                    ]
                  })
                ]
              }),
              jsx('div', {
                style: styles.controls,
                children: jsx(Button, {
                  size: 'sm',
                  variant: 'outline',
                  style: {
                    color: '#d4d4d8',
                    background: palette.panel,
                    borderColor: palette.border,
                    fontSize: '11px',
                    padding: '3px 8px'
                  },
                  disabled: isScanning,
                  onClick: refreshProviders,
                  children: isScanning ? 'Scanning...' : 'Refresh'
                })
              })
            ]
          }),
          jsxs('div', {
            style: styles.statusPanel,
            children: [
              jsxs('div', {
                style: styles.statusHeaderRow,
                children: [
                  jsx('div', {
                    style: {
                      fontSize: '10px',
                      fontWeight: 700,
                      color: '#d4d4d8',
                      letterSpacing: '0.04em',
                      whiteSpace: 'nowrap'
                    },
                    children: 'PROVIDER CONNECTION STATUS'
                  }),
                  jsx('div', {
                    style: styles.muted,
                    children: statusSummaryParts.join(' · ')
                  })
                ]
              }),
              jsx('div', {
                style: { display: 'flex', flexWrap: 'wrap', gap: '6px' },
                children: statusOrderedProviders.map((provider, index) => {
                  const providerId = getProviderId(provider)
                  const providerName = truncateText(provider.name || provider.id, 80) || providerId
                  const hasBar = hasUsageBar(provider)
                  const isAvailable = Boolean(provider.available)
                  return jsx('span', {
                    key: `status-${providerId}-${index}`,
                    title: hasBar
                      ? `${providerName} — Active usage bar`
                      : isAvailable
                        ? `${providerName} — Connected, but no usage bar exposed (card hidden)`
                        : `${providerName} — Offline / unavailable (card hidden)`,
                    style: {
                      padding: '4px 9px',
                      borderRadius: '999px',
                      fontSize: '10px',
                      fontWeight: 600,
                      maxWidth: '280px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      display: 'inline-block',
                      border: '1px solid ' + (hasBar ? 'rgba(45, 212, 191, 0.35)' : palette.border),
                      background: hasBar ? 'rgba(45, 212, 191, 0.08)' : palette.panel,
                      color: hasBar ? '#34d399' : '#71717a'
                    },
                    children: hasBar
                      ? `● ${providerName}`
                      : isAvailable
                        ? `◌ ${providerName} (no quota)`
                        : `○ ${providerName} (offline)`
                  })
                })
              })
            ]
          }),
          errorList.length
            ? jsx('div', {
                style: styles.error,
                children: errorList.map((message, index) => jsx('div', { key: `err-${index}`, children: message }))
              })
            : null,
          isRestarting
            ? jsxs('div', {
                style: {
                  ...styles.grid,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '300px',
                  color: '#a1a1aa',
                  fontSize: '14px'
                },
                children: [
                  jsx('div', {
                    style: {
                      width: '24px',
                      height: '24px',
                      border: '2px solid rgba(255,255,255,0.1)',
                      borderTopColor: '#bef264',
                      borderRadius: '50%',
                      animation: 'spin 0.8s linear infinite'
                    }
                  }),
                  jsx('div', {
                    style: { marginTop: '12px', fontWeight: 500 },
                    children: 'Restarting Gateway...'
                  })
                ]
              })
            : quotaProviders.length
              ? jsx('div', {
                  style: styles.grid,
                  children: quotaProviders.map((provider, index) => {
                    const providerId = getProviderId(provider)
                    return jsx(ProviderCard, {
                      provider,
                      key: `card-${providerId}-${index}`,
                      isFirst: index === 0,
                      isLast: index === quotaProviders.length - 1,
                      onMoveLeft: () => moveQuotaProviderByIndex(index, -1),
                      onMoveRight: () => moveQuotaProviderByIndex(index, 1)
                    })
                  })
                })
              : jsx('div', {
                  style: styles.unavailable,
                  children: 'No connected providers are currently exposing an account usage bar. Auto-checking for new quota connections…'
                })
        ]
      }),
      jsx(ConfirmDialog, {
        open: restartWarningOpen,
        onClose: () => setRestartWarningOpen(false),
        onConfirm: restartGatewayAndRefresh,
        title: 'Restart Hermes Gateway?',
        description: 'AI Usage Meter’s backend has changed and must be reloaded. Restarting the Gateway will briefly disconnect active chats and tools. Hermes Desktop should reconnect automatically.',
        confirmLabel: 'Restart Gateway & Refresh',
        busyLabel: 'Restarting Gateway…',
        doneLabel: 'Restarted',
        destructive: true,
        dismissOnConfirm: true
      })
    ]
  })
}

function UsageChip() {
  const focusedSessionId = useValue(host?.state?.focusedSessionId ?? FALLBACK_STORE)
  const focusedUsage = useValue(host?.state?.focusedUsage ?? FALLBACK_STORE)
  const sharedCache = useCachedUsageData()
  const query = useQuery({
    queryKey: [ID, 'usage', 30],
    queryFn: async () => {
      const result = await safeRest('/usage?days=30')
      publishUsageData(result)
      return result
    },
    refetchInterval: AUTO_CHECK_INTERVAL_MS
  })

  useEffect(() => {
    if (query.data && typeof query.data === 'object') {
      publishUsageData(query.data)
    }
  }, [query.data])

  const activeData = sharedCache || query.data
  const providers = useMemo(
    () => toArray(activeData?.providers).filter(Boolean),
    [activeData]
  )

  const highest = useMemo(() => {
    const accountValues = providers.flatMap(provider =>
      toArray(provider.windows)
        .filter(window => window && (!window.scope || window.scope === 'account'))
        .map(getWindowUsedPercent)
        .filter(val => val !== null)
    )
    if (accountValues.length) return Math.max(...accountValues)

    const fallbackValues = providers.flatMap(provider =>
      toArray(provider.windows)
        .map(getWindowUsedPercent)
        .filter(val => val !== null)
    )
    return fallbackValues.length ? Math.max(...fallbackValues) : null
  }, [providers])

  const providerAlerts = useMemo(() => {
    return providers.flatMap(provider => {
      const providerId = truncateText(getProviderId(provider), 64)
      const providerName = truncateText(provider.name || provider.id, 64) || providerId
      return toArray(provider.windows)
        .filter(window => window && (!window.scope || window.scope === 'account'))
        .map(window => {
          const used = getWindowUsedPercent(window)
          if (used === null || used < ALERT_THRESHOLD) return null
          const label = truncateText(window.label, 64) || 'Quota'
          return {
            key: `provider:${providerId}:${label}`,
            message: `${providerName} — ${label}: ${used.toFixed(1)}% used`
          }
        })
        .filter(Boolean)
    })
  }, [providers])

  const contextMax = toFiniteNumber(focusedUsage?.context_max) ?? 0
  const contextUsed = toFiniteNumber(focusedUsage?.context_used) ?? 0
  const rawContextPercent = toFiniteNumber(focusedUsage?.context_percent)
  const computedContextPercent = contextMax > 0
    ? rawContextPercent ?? (contextUsed / contextMax) * 100
    : null
  const contextPercent = computedContextPercent !== null
    ? Math.max(0, Math.min(100, computedContextPercent))
    : null

  const hasLoadedData = Boolean(activeData)
  const activeAlerts = useMemo(() => {
    const list = [...providerAlerts]
    if (contextPercent !== null && contextPercent >= ALERT_THRESHOLD) {
      list.push({
        key: `context:${truncateText(focusedSessionId || 'unknown', 64)}`,
        message: `Current session context: ${contextPercent.toFixed(1)}% used`
      })
    }
    return list
  }, [providerAlerts, contextPercent, focusedSessionId])

  const alertKeySignature = useMemo(
    () => JSON.stringify(activeAlerts.map(alert => alert.key)),
    [activeAlerts]
  )

  useEffect(() => {
    if (!hasLoadedData) return
    let previous = []
    try {
      const parsed = JSON.parse(globalThis.localStorage?.getItem(ALERT_STORAGE_KEY) || '[]')
      previous = Array.isArray(parsed) ? parsed : []
    } catch {
      previous = []
    }
    const previousKeys = new Set(previous.filter(item => typeof item === 'string'))
    const fresh = activeAlerts.filter(alert => !previousKeys.has(alert.key))
    if (fresh.length) {
      safeNotify({
        kind: 'warning',
        title: 'AI usage reached 90%',
        message: fresh.map(alert => alert.message).join(' • ')
      })
    }
    try {
      globalThis.localStorage?.setItem(ALERT_STORAGE_KEY, alertKeySignature)
    } catch (err) {
      console.warn('Unable to persist alert state', err)
    }
  }, [alertKeySignature, hasLoadedData])

  return jsx(Button, {
    size: 'sm',
    variant: 'ghost',
    onClick: () => safeNavigate('/ai-usage-meter'),
    children: highest === null ? 'Usage unavailable' : `Usage ${Math.round(highest)}%`
  })
}

export default {
  id: ID,
  name: 'AI Usage Meter',
  description: 'Live provider quota, focused-session context, and per-provider model usage.',
  defaultEnabled: true,
  register(ctx) {
    rest = ctx?.rest
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/ai-usage-meter' },
        title: 'AI Usage Meter',
        render: () => jsx(UsagePage, {})
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 57,
        data: { path: '/ai-usage-meter', label: 'AI Usage Meter', codicon: 'graph' }
      },
      {
        id: 'chip',
        area: STATUSBAR_AREAS.right,
        order: 90,
        render: () => jsx(UsageChip, {})
      },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'ai-usage-meter.open',
          label: 'AI Usage Meter: Open',
          keywords: ['usage', 'quota', 'tokens', 'anthropic', 'codex', 'openrouter'],
          run: () => safeNavigate('/ai-usage-meter')
        }
      }
    ])
  }
}
