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
import React, { useEffect, useState, useMemo } from 'react'

const ID = 'ai-usage-meter'
const ALERT_THRESHOLD = 90
const ALERT_STORAGE_KEY = `${ID}:active-alerts`
const ORDER_STORAGE_KEY = `${ID}:provider-order`
const EXPECTED_BACKEND_VERSION = '0.2.0'

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
  `,
  limeCyan: 'linear-gradient(90deg, #bef264 0%, #4ade80 52%, #2dd4bf 100%)',
  rose: 'linear-gradient(90deg, #fb7185 0%, #f43f5e 52%, #fb923c 100%)',
  purple: 'linear-gradient(90deg, #a78bfa 0%, #c084fc 52%, #e879f9 100%)'
}

let rest

function backendNeedsRestart(data) {
  return data?.backend_version !== EXPECTED_BACKEND_VERSION
}

const styles = {
  page: {
    height: '100%',
    overflow: 'hidden',
    padding: '12px',
    color: palette.text
  },
  shell: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    maxHeight: 'calc(100vh - 32px)',
    overflow: 'hidden'
  },
  header: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: '10px',
    flexWrap: 'wrap'
  },
  title: { margin: 0, fontSize: '20px', lineHeight: 1.2, fontWeight: 700, letterSpacing: '-0.02em' },
  controls: { display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' },
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
    gridAutoRows: '1fr',
    gap: '10px',
    alignItems: 'stretch'
  },
  quotaSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    minHeight: '160px'
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
    borderTop: `1px solid rgba(255,255,255,0.06)`,
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
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '12px',
    border: `1px dashed ${palette.borderStrong}`,
    borderRadius: '14px',
    color: palette.muted,
    fontSize: '11px',
    textAlign: 'center'
  },
  error: {
    border: '1px solid rgba(244, 63, 94, 0.55)',
    borderRadius: '10px',
    padding: '10px 12px',
    color: '#fb7185',
    fontSize: '12px'
  }
}

function compactNumber(value) {
  const number = Number(value || 0)
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(number)
}

function money(value) {
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 4
  }).format(Number(value || 0))
}

function localTime(value) {
  if (!value) return 'No reset reported'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Reset time unavailable' : date.toLocaleString()
}

function Gauge({ percent, scheme = 'limeCyan' }) {
  const safe = typeof percent === 'number' && Number.isFinite(percent)
    ? Math.max(0, Math.min(100, percent))
    : 0
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
        background: palette[resolvedScheme],
        boxShadow: glow,
        transition: 'width 240ms ease'
      }
    })
  })
}

function SourceBadge({ value }) {
  const unavailable = value === 'UNAVAILABLE'
  return jsx(Badge, {
    variant: 'secondary',
    style: {
      color: unavailable ? '#fb7185' : '#d4d4d8',
      background: unavailable ? 'rgba(244, 63, 94, 0.12)' : palette.panel,
      border: `1px solid ${unavailable ? 'rgba(244, 63, 94, 0.24)' : palette.border}`
    },
    children: value
  })
}

function UsageWindow({ window, scheme }) {
  const used = window.used_percent
  const remaining = window.remaining_percent
  return jsxs('div', {
    style: styles.window,
    children: [
      jsxs('div', {
        style: styles.row,
        children: [
          jsx('strong', {
            style: { minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '12px' },
            title: window.label,
            children: window.label
          }),
          jsx('span', {
            style: { ...styles.muted, flexShrink: 0, whiteSpace: 'nowrap' },
            children: typeof used === 'number' ? `${used.toFixed(1)}% used` : 'Usage unavailable'
          })
        ]
      }),
      jsx(Gauge, { percent: used, scheme }),
      jsx('div', {
        style: { ...styles.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
        title: [
          typeof remaining === 'number' ? `${remaining.toFixed(1)}% remaining` : 'Remaining unavailable',
          localTime(window.reset_at),
          window.detail,
          `Scope: ${window.scope}`
        ].filter(Boolean).join(' • '),
        children: `${typeof remaining === 'number' ? `${remaining.toFixed(1)}% left` : 'Remaining unavailable'} • ${localTime(window.reset_at)}`
      })
    ]
  })
}

function ModelRow({ model }) {
  const costLabel = model.cost_source === 'UNAVAILABLE'
    ? 'No cost recorded'
    : `${money(model.display_cost)} ${model.cost_source.toLowerCase()}`
  const metadata = model.api_metadata
  const metadataLabel = metadata
    ? `${compactNumber(metadata.input_token_limit)} input • ${compactNumber(metadata.output_token_limit)} output`
    : null
  return jsxs('div', {
    style: { ...styles.modelRow, minHeight: metadata ? '52px' : styles.modelRow.minHeight },
    children: [
      jsxs('div', {
        style: { minWidth: 0, overflow: 'hidden' },
        title: [model.name, metadata?.display_name, ...model.matched_models].filter(Boolean).join(' • '),
        children: [
          jsx('div', {
            style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 600 },
            children: model.name
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

function ProviderCard({ provider, onMoveLeft, onMoveRight }) {
  if (!provider) return null
  const scheme = provider.id === 'openai-codex' ? 'purple' : 'limeCyan'
  const visibleModels = [...(provider?.models || [])]
    .filter(model => Number(model.total_tokens || 0) > 0)
    .sort((left, right) =>
      Number(right.total_tokens || 0) - Number(left.total_tokens || 0) ||
      String(left.name || '').localeCompare(String(right.name || ''))
    )
    .slice(0, 3)
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
                style: { padding: '2px 4px', fontSize: '10px', color: palette.muted },
                onClick: onMoveLeft,
                children: '←'
              }),
              jsxs('div', {
                style: { display: 'flex', flexDirection: 'column', minWidth: 0 },
                children: [
                  jsx('h2', { style: { ...styles.cardTitle, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }, children: provider.name }),
                  provider.plan ? jsx('div', { style: styles.muted, children: `Provider plan: ${provider.plan}` }) : null
                ]
              }),
              jsx(Button, {
                size: 'sm',
                variant: 'ghost',
                style: { padding: '2px 4px', fontSize: '10px', color: palette.muted },
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
          provider?.windows?.length
            ? (provider.windows || []).map((window, index) => jsx(UsageWindow, { window, scheme, key: `${window.label}-${index}` }))
            : jsx('div', {
                style: provider.available
                  ? {
                      ...styles.unavailable,
                      color: '#d4d4d8',
                      background: 'rgba(45, 212, 191, 0.055)',
                      border: '1px solid rgba(45, 212, 191, 0.22)'
                    }
                  : styles.unavailable,
                children: provider.unavailable_reason || 'No account limit was returned by this provider.'
              }),
          provider.details?.length
            ? jsx('div', {
                style: { ...styles.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                title: provider.details.join(' • '),
                children: provider.details.join(' • ')
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
              ? visibleModels.map(model => jsx(ModelRow, { model, key: model.name }))
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
  const [refreshRevision, setRefreshRevision] = useState(0)
  const [providerOrder, setProviderOrder] = useState([])
  const [isRestarting, setIsRestarting] = useState(false)
  const [restartWarningOpen, setRestartWarningOpen] = useState(false)
  const query = useQuery({
    queryKey: [ID, 'usage', 30, refreshRevision],
    queryFn: () => rest(`/usage?days=30&refresh=${refreshRevision > 0 ? 'true' : 'false'}`)
  })

  const data = query.data || { providers: [], errors: [], windows: [] }

  useEffect(() => {
    if (query.data && (data.providers || []).length > 0) {
      try {
        const saved = globalThis.localStorage?.getItem(ORDER_STORAGE_KEY)
        if (saved) {
          setProviderOrder(JSON.parse(saved))
        }
      } catch (e) {
        console.error('Failed to parse provider order', e)
      }
    }
  }, [query.data])

  const sortedProviders = useMemo(() => {
    const providers = data.providers || []
    if (!providers.length) return []
    if (!providerOrder || !providerOrder.length) return providers

    return [...providers].sort((a, b) => {
      if (!a || !b) return 0
      const idxA = providerOrder.indexOf(a.id)
      const idxB = providerOrder.indexOf(b.id)
      if (idxA === -1 && idxB === -1) return 0
      if (idxA === -1) return 1
      if (idxB === -1) return -1
      return idxA - idxB
    })
  }, [data.providers, providerOrder])

  useEffect(() => {
    if (isRestarting && query.data && !query.isLoading) {
      setIsRestarting(false)
    }
  }, [isRestarting, query.data, query.isLoading])

  if (query.isLoading) {
    return jsx('div', { style: styles.page, children: jsx(EmptyState, { title: 'Loading usage data' }) })
  }
  if (query.isError) {
    return jsx('div', {
      style: styles.page,
      children: jsx(ErrorState, {
        title: 'Usage data unavailable',
        description: query.error?.message || 'The plugin backend could not be reached.'
      })
    })
  }

  const updateOrder = (newOrder) => {
    setProviderOrder(newOrder)
    globalThis.localStorage?.setItem(ORDER_STORAGE_KEY, JSON.stringify(newOrder))
  }
  const moveLeft = (id) => {
    const order = providerOrder.length ? [...providerOrder] : sortedProviders.map(p => p.id)
    const idx = order.indexOf(id)
    if (idx === -1) {
      updateOrder([id, ...order])
      return
    }
    if (idx <= 0) return
    ;[order[idx], order[idx - 1]] = [order[idx - 1], order[idx]]
    updateOrder(order)
  }
  const moveRight = (id) => {
    const order = providerOrder.length ? [...providerOrder] : sortedProviders.map(p => p.id)
    const idx = order.indexOf(id)
    if (idx === -1) {
      updateOrder([...order, id])
      return
    }
    if (idx >= order.length - 1) return
    ;[order[idx], order[idx + 1]] = [order[idx + 1], order[idx]]
    updateOrder(order)
  }

  const refreshProviders = () => {
    if (backendNeedsRestart(data)) {
      setRestartWarningOpen(true)
      return
    }
    setRefreshRevision(value => value + 1)
  }
  const restartGatewayAndRefresh = async () => {
    if (typeof host.restartGateway !== 'function') {
      throw new Error('Gateway restart is unavailable in this Hermes Desktop version.')
    }
    setIsRestarting(true)
    setRestartWarningOpen(false)
    try {
      await host.restartGateway()
    } catch (e) {
      console.log('Gateway is restarting...')
    }

    let attempts = 0
    const poll = setInterval(async () => {
      attempts++
      try {
        const res = await rest('/usage?days=30&refresh=true')
        if (res) {
          clearInterval(poll)
          setRefreshRevision(value => value + 1)
        }
      } catch (e) {
        if (attempts > 10) {
          clearInterval(poll)
          console.error('Gateway failed to reconnect in time.')
        }
      }
    }, 1000)
  }

  return jsxs('div', {
    style: styles.page,
    children: [
      jsxs('main', {
        style: styles.shell,
        children: [
          jsxs('header', {
            style: styles.header,
            children: [
              jsx('h1', { style: styles.title, children: 'AI Usage Meter' }),
              jsxs('div', {
                style: styles.controls,
                children: [
                  jsx(Button, {
                    size: 'sm',
                    variant: 'outline',
                    style: {
                      color: '#d4d4d8',
                      background: palette.panel,
                      borderColor: palette.border,
                      fontSize: '11px',
                      padding: '3px 8px'
                    },
                    disabled: query.isFetching,
                    onClick: refreshProviders,
                    children: query.isFetching ? 'Scanning...' : 'Refresh'
                  })
                ]
              })
            ]
          }),
          data.errors?.length
            ? jsx('div', {
                style: styles.error,
                children: data.errors.map((message, index) => jsx('div', { key: index, children: message }))
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
            : jsx('div', {
                style: {
                  ...styles.grid,
                  gridTemplateColumns: `repeat(5, minmax(0, 1fr))`
                },
                children: sortedProviders.map(provider => jsx(ProviderCard, {
                  provider,
                  key: provider.id,
                  onMoveLeft: () => moveLeft(provider.id),
                  onMoveRight: () => moveRight(provider.id)
                }))
              }),
          jsxs('div', {
            style: {
              marginTop: '10px',
              padding: '10px 14px',
              borderRadius: '12px',
              background: palette.panel,
              border: '1px solid ' + palette.border
            },
            children: [
              jsx('div', { style: { fontSize: '10px', fontWeight: 700, color: '#d4d4d8', letterSpacing: '0.04em', marginBottom: '8px', whiteSpace: 'nowrap' }, children: 'PROVIDER CONNECTION STATUS' }),
              jsx('div', {
                style: { display: 'flex', flexWrap: 'wrap', gap: '6px' },
                children: [
                  ...(data.providers || []).map(provider => jsxs('span', {
                    key: provider.id,
                    style: {
                      padding: '4px 9px',
                      borderRadius: '999px',
                      fontSize: '10px',
                      fontWeight: 600,
                      border: '1px solid ' + (provider.available ? 'rgba(45, 212, 191, 0.35)' : palette.border),
                      background: provider.available ? 'rgba(45, 212, 191, 0.08)' : palette.panel,
                      color: provider.available ? '#34d399' : '#a1a1aa'
                    },
                    children: [
                      provider.available ? '●' : '○',
                      ' ' + provider.name
                    ]
                  }))
                ]
              })
            ]
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
  const focusedSessionId = useValue(host.state.focusedSessionId)
  const focusedUsage = useValue(host.state.focusedUsage)
  const query = useQuery({
    queryKey: [ID, 'usage', 30],
    queryFn: () => rest('/usage?days=30'),
    refetchInterval: 60_000
  })
  const values = (query.data?.providers || []).flatMap(provider =>
    (provider.windows || []).filter(window => window.scope === 'account' && typeof window.used_percent === 'number')
      .map(window => window.used_percent)
  )
  const highest = values.length ? Math.max(...values) : null

  const providerAlerts = (query.data?.providers || []).flatMap(provider =>
    (provider.windows || [])
      .filter(window => window.scope === 'account' && typeof window.used_percent === 'number' && window.used_percent >= ALERT_THRESHOLD)
      .map(window => ({
        key: `provider:${provider.id}:${window.label}`,
        message: `${provider.name} — ${window.label}: ${window.used_percent.toFixed(1)}% used`
      }))
  )
  const contextMax = Number(focusedUsage?.context_max || 0)
  const contextUsed = Number(focusedUsage?.context_used || 0)
  const contextPercent = contextMax > 0
    ? Math.max(0, Math.min(100, Number(focusedUsage?.context_percent ?? (contextUsed / contextMax) * 100)))
    : null
  const alertSignature = JSON.stringify([
    ...providerAlerts,
    contextPercent !== null && contextPercent >= ALERT_THRESHOLD
      ? { key: `context:${focusedSessionId || 'unknown'}`, message: `Current session context: ${contextPercent.toFixed(1)}% used` }
      : null
  ].filter(Boolean))

  useEffect(() => {
    const current = JSON.parse(alertSignature)
    let previous = []
    try {
      previous = JSON.parse(globalThis.localStorage?.getItem(ALERT_STORAGE_KEY) || '[]')
    } catch {
      previous = []
    }
    const previousKeys = new Set(Array.isArray(previous) ? previous : [])
    const fresh = current.filter(alert => !previousKeys.has(alert.key))
    if (fresh.length) {
      host.notify({
        kind: 'warning',
        title: 'AI usage reached 90%',
        message: fresh.map(alert => alert.message).join(' • ')
      })
    }
    try {
      globalThis.localStorage?.setItem(ALERT_STORAGE_KEY, JSON.stringify(current.map(alert => alert.key)))
    } catch {
      // Notification still works when persistent browser storage is unavailable.
    }
  }, [alertSignature])

  return jsx(Button, {
    size: 'sm',
    variant: 'ghost',
    onClick: () => host.navigate('/ai-usage-meter'),
    children: highest === null ? 'Usage unavailable' : `Usage ${Math.round(highest)}%`
  })
}

export default {
  id: ID,
  name: 'AI Usage Meter',
  description: 'Live provider quota, focused-session context, and per-provider model usage.',
  defaultEnabled: true,
  register(ctx) {
    rest = ctx.rest
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
          run: () => host.navigate('/ai-usage-meter')
        }
      }
    ])
  }
}
