/**
 * Mission control — one pane to watch all ongoing Hermes work.
 *
 * Bundled SDK-consumer plugin (the Radio shape): plain ESM, `jsx()` calls
 * only — no JSX syntax — importing only '@hermes/plugin-sdk', 'react', and
 * 'react/jsx-runtime'. The same file loads unchanged through the runtime
 * plugin door at $HERMES_HOME/desktop-plugins/mission-control/plugin.js.
 *
 * Shows, live: every recent session (working state, model, click-to-open),
 * the cron roster with next runs, and the gateway line. Copy ships as
 * ctx.i18n bundles (en / ja / zh / zh-hant); styles are scoped to
 * [data-mission-control], injected at register, and removed on dispose.
 */
import { haptic, host, queryClient, usePluginI18n, useQuery, useValue } from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID = 'mission-control'
const KEY_SESSIONS = [ID, 'sessions']
const KEY_ACTIVE = [ID, 'active']
const KEY_CRON = [ID, 'cron']
const SESSION_LIMIT = 80

// LiveSessionStatus values that mean "doing something right now".
const WORKING_STATUSES = new Set(['starting', 'working', 'streaming', 'resuming'])
// Waiting on the user (approval / queued input) — worth an eye, not a pulse.
const WAITING_STATUSES = new Set(['waiting'])

const EN = {
  title: 'Mission control',
  refresh: 'Refresh',
  refreshTip: 'Refetch sessions, live state and cron now',
  sessions: n => `${n} ${n === 1 ? 'session' : 'sessions'}`,
  workingCount: n => `${n} working`,
  updatedNow: 'updated just now',
  updatedAgo: age => `updated ${age} ago`,
  loading: 'loading…',
  working: 'working',
  recent: 'recent',
  msgCount: n => `${n} msgs`,
  activeAgo: age => `active ${age}`,
  startedAgo: age => `started ${age}`,
  untitled: '(untitled)',
  sessionsLabel: 'sessions',
  sessionListUnavailable: 'session list unavailable',
  loadingSessions: 'loading sessions…',
  noSessions: 'no sessions yet',
  openFailed: 'Could not open that session',
  cronLoading: 'cron — loading…',
  cronUnavailable: 'cron — unavailable',
  cronScheduled: n => `cron — ${n} scheduled`,
  cronPaused: n => `${n} paused`,
  cronNoJobs: 'cron — no jobs',
  cronTotal: n => `${n} total`,
  jobFallback: 'job',
  gateway: 'gateway',
  socket: 'socket',
  running: 'running',
  notRunning: 'not running'
}

const JA = {
  title: 'ミッションコントロール',
  refresh: '更新',
  refreshTip: 'セッション・稼働状態・cron を今すぐ再取得',
  sessions: n => `${n} セッション`,
  workingCount: n => `${n} 稼働中`,
  updatedNow: 'たった今更新',
  updatedAgo: age => `${age}前に更新`,
  loading: '読み込み中…',
  working: '稼働中',
  recent: '最近',
  msgCount: n => `${n} 件`,
  activeAgo: age => `アクティブ ${age}前`,
  startedAgo: age => `開始 ${age}前`,
  untitled: '（無題）',
  sessionsLabel: 'セッション',
  sessionListUnavailable: 'セッション一覧を取得できません',
  loadingSessions: 'セッションを読み込み中…',
  noSessions: 'セッションはまだありません',
  openFailed: 'このセッションを開けませんでした',
  cronLoading: 'cron — 読み込み中…',
  cronUnavailable: 'cron — 利用できません',
  cronScheduled: n => `cron — 予定 ${n} 件`,
  cronPaused: n => `一時停止 ${n} 件`,
  cronNoJobs: 'cron — ジョブなし',
  cronTotal: n => `全 ${n} 件`,
  jobFallback: 'ジョブ',
  gateway: 'ゲートウェイ',
  socket: 'ソケット',
  running: '稼働中',
  notRunning: '停止中'
}

const ZH = {
  title: '任务控制',
  refresh: '刷新',
  refreshTip: '立即重新获取会话、运行状态与定时任务',
  sessions: n => `${n} 个会话`,
  workingCount: n => `${n} 个进行中`,
  updatedNow: '刚刚更新',
  updatedAgo: age => `${age}前更新`,
  loading: '加载中…',
  working: '进行中',
  recent: '最近',
  msgCount: n => `${n} 条消息`,
  activeAgo: age => `活跃 ${age}前`,
  startedAgo: age => `开始于 ${age}前`,
  untitled: '（未命名）',
  sessionsLabel: '会话',
  sessionListUnavailable: '无法获取会话列表',
  loadingSessions: '正在加载会话…',
  noSessions: '暂无会话',
  openFailed: '无法打开该会话',
  cronLoading: 'cron — 加载中…',
  cronUnavailable: 'cron — 不可用',
  cronScheduled: n => `cron — ${n} 个已排期`,
  cronPaused: n => `${n} 个已暂停`,
  cronNoJobs: 'cron — 无任务',
  cronTotal: n => `共 ${n} 个`,
  jobFallback: '任务',
  gateway: '网关',
  socket: '套接字',
  running: '运行中',
  notRunning: '未运行'
}

const ZH_HANT = {
  title: '任務控制',
  refresh: '重新整理',
  refreshTip: '立即重新取得工作階段、執行狀態與排程工作',
  sessions: n => `${n} 個工作階段`,
  workingCount: n => `${n} 個進行中`,
  updatedNow: '剛剛更新',
  updatedAgo: age => `${age}前更新`,
  loading: '載入中…',
  working: '進行中',
  recent: '最近',
  msgCount: n => `${n} 則訊息`,
  activeAgo: age => `活躍 ${age}前`,
  startedAgo: age => `開始於 ${age}前`,
  untitled: '（未命名）',
  sessionsLabel: '工作階段',
  sessionListUnavailable: '無法取得工作階段清單',
  loadingSessions: '正在載入工作階段…',
  noSessions: '尚無工作階段',
  openFailed: '無法開啟該工作階段',
  cronLoading: 'cron — 載入中…',
  cronUnavailable: 'cron — 無法使用',
  cronScheduled: n => `cron — ${n} 個已排程`,
  cronPaused: n => `${n} 個已暫停`,
  cronNoJobs: 'cron — 沒有工作',
  cronTotal: n => `共 ${n} 個`,
  jobFallback: '工作',
  gateway: '閘道',
  socket: '通訊端',
  running: '執行中',
  notRunning: '未執行'
}

/** Registered via `ctx.i18n.register` at plugin load (disposer tracked). */
const LOCALES = { en: EN, ja: JA, zh: ZH, 'zh-hant': ZH_HANT }

const CSS = `
[data-mission-control] { display:flex; flex-direction:column; height:100%; min-height:0; overflow:hidden; color:var(--ui-text-secondary); font-size:.75rem; line-height:1.35; }
[data-mission-control] [data-mc-head] { display:flex; align-items:center; gap:8px; padding:9px 10px 3px; }
[data-mission-control] [data-mc-head] b { color:var(--ui-text-primary); font-weight:650; }
[data-mission-control] [data-mc-spacer] { flex:1; }
[data-mission-control] [data-mc-btn] { appearance:none; border:0; background:transparent; color:var(--ui-text-quaternary); font:inherit; font-size:.6875rem; padding:2px 7px; border-radius:6px; cursor:pointer; }
[data-mission-control] [data-mc-btn]:hover { color:var(--ui-text-primary); background:color-mix(in srgb, var(--ui-text-quaternary) 12%, transparent); }
[data-mission-control] [data-mc-stats] { padding:0 10px 8px; color:var(--ui-text-quaternary); font-size:.6875rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
[data-mission-control] [data-mc-scroll] { flex:1; min-height:0; overflow:auto; padding:0 5px 6px; }
[data-mission-control] [data-mc-section] { display:flex; justify-content:space-between; gap:8px; padding:8px 6px 3px; color:var(--ui-text-quaternary); font-size:.625rem; font-weight:650; letter-spacing:.09em; text-transform:uppercase; }
[data-mission-control] [data-mc-row] { display:grid; grid-template-columns:11px minmax(0,1fr) auto; gap:7px; align-items:start; padding:5px 6px; border-radius:7px; cursor:pointer; }
[data-mission-control] [data-mc-row]:hover { background:color-mix(in srgb, var(--ui-text-quaternary) 9%, transparent); }
[data-mission-control] [data-mc-dot] { width:7px; height:7px; margin-top:4px; border-radius:50%; background:var(--ui-text-quaternary); opacity:.45; }
[data-mission-control] [data-mc-dot='working'] { background:var(--ui-accent); opacity:1; animation:mc-pulse 1.5s ease-in-out infinite; }
[data-mission-control] [data-mc-dot='waiting'] { background:var(--ui-text-secondary); opacity:1; }
[data-mission-control] [data-mc-main] { min-width:0; display:flex; flex-direction:column; gap:1px; }
[data-mission-control] [data-mc-title] { color:var(--ui-text-primary); font-weight:600; display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
[data-mission-control] [data-mc-row][data-mc-state='plain'] [data-mc-title] { font-weight:500; color:var(--ui-text-secondary); }
[data-mission-control] [data-mc-meta] { color:var(--ui-text-quaternary); font-size:.6875rem; display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
[data-mission-control] [data-mc-age] { color:var(--ui-text-quaternary); font-size:.6875rem; font-variant-numeric:tabular-nums; white-space:nowrap; padding-top:1px; }
[data-mission-control] [data-mc-note] { padding:8px; color:var(--ui-text-quaternary); font-size:.6875rem; }
[data-mission-control] [data-mc-note][data-mc-err] { color:var(--ui-accent); }
[data-mission-control] [data-mc-foot] { border-top:1px solid var(--ui-stroke-secondary); padding:7px 10px 9px; display:flex; flex-direction:column; gap:3px; }
[data-mission-control] [data-mc-foot-h] { display:flex; justify-content:space-between; gap:8px; color:var(--ui-text-quaternary); font-size:.625rem; font-weight:650; letter-spacing:.09em; text-transform:uppercase; }
[data-mission-control] [data-mc-cron] { display:grid; grid-template-columns:9px minmax(0,1fr) auto; gap:7px; align-items:baseline; font-size:.6875rem; color:var(--ui-text-tertiary); }
[data-mission-control] [data-mc-cron] [data-mc-cname] { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
[data-mission-control] [data-mc-cdot] { width:6px; height:6px; border-radius:50%; background:var(--ui-text-quaternary); opacity:.8; }
[data-mission-control] [data-mc-cdot='error'] { background:var(--ui-accent); opacity:1; }
[data-mission-control] [data-mc-cdot='paused'] { opacity:.25; }
[data-mission-control] [data-mc-gw] { color:var(--ui-text-quaternary); font-size:.6875rem; display:flex; gap:10px; flex-wrap:wrap; }
[data-mission-control] [data-mc-gw] b { color:var(--ui-text-tertiary); font-weight:600; }
@keyframes mc-pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
`

/* ── small formatters ─────────────────────────────────────────────────── */

const pad = n => (n < 10 ? '0' + n : String(n))

/** Epoch seconds → compact age ("now", "12m", "3h", "2d"). */
const ago = epochSec => {
  if (!epochSec) return ''
  const s = Math.max(0, Math.round(Date.now() / 1000 - epochSec))
  if (s < 45) return 'now'
  const m = Math.round(s / 60)
  if (m < 60) return m + 'm'
  const h = Math.floor(m / 60)
  if (h < 24) return h + 'h'
  return Math.floor(h / 24) + 'd'
}

/** ISO datetime → "HH:MM" local, or '' when missing/invalid. */
const clockOf = iso => {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return pad(d.getHours()) + ':' + pad(d.getMinutes())
}

const clean = value =>
  String(value == null ? '' : value)
    .replace(/\s+/g, ' ')
    .trim()

const modelShort = value => {
  const s = clean(value)
  return s ? s.split('/').pop() : ''
}

const titleOf = (row, untitled) => clean(row.title) || clean(row.preview).slice(0, 80) || row.id || untitled

const isWorkingStatus = value => WORKING_STATUSES.has(String(value || ''))

const isWaitingStatus = value => WAITING_STATUSES.has(String(value || ''))

/** 'ok' | 'error' | 'paused' for one cron job row. */
const cronState = job => {
  const state = String(job.state || (job.enabled === false ? 'paused' : 'scheduled'))
  if (state === 'paused') return 'paused'
  const last = String(job.last_status || '')
  if (last && !/^ok/i.test(last)) return 'error'
  return 'ok'
}

/** Re-render on an interval so "12m" style ages keep ticking. */
function useNowTick(ms) {
  const [, setTick] = useState(0)
  useEffect(() => {
    const timer = setInterval(() => setTick(x => x + 1), ms)
    return () => clearInterval(timer)
  }, [ms])
}

/** The plugin bundle's translator for the active locale. */
function useText() {
  const t = usePluginI18n(ID)

  return useCallback((key, ...args) => t(key, ...args), [t])
}

/** Tab-strip label that follows the locale (the register-time title cannot). */
function TabTitle() {
  const tx = useText()

  return jsx('span', { children: tx('title') })
}

/* ── pieces ───────────────────────────────────────────────────────────── */

function SectionHead({ label, count }) {
  return jsxs('div', {
    'data-mc-section': true,
    children: [jsx('span', { children: label }), jsx('span', { children: count })]
  })
}

function SessionRow({ row, onOpen }) {
  const tx = useText()
  const live = row.live
  const status = live ? String(live.status || '') : ''
  const working = isWorkingStatus(status) || row.busy === true
  const waiting = isWaitingStatus(status)
  const state = working ? 'working' : waiting ? 'waiting' : 'plain'

  const meta = []
  if (live && live.model) meta.push(modelShort(live.model))
  if (row.source) meta.push(String(row.source))
  meta.push(tx('msgCount', Number(row.message_count) || 0))
  if (working && live && live.last_active) meta.push(tx('activeAgo', ago(live.last_active)))
  else if (row.started_at) meta.push(tx('startedAgo', ago(row.started_at)))

  const rightLabel =
    working && live && live.last_active ? ago(live.last_active) : row.started_at ? ago(row.started_at) : ''

  const open = () => onOpen(row)

  return jsxs('div', {
    'data-mc-row': true,
    'data-mc-state': state,
    role: 'button',
    tabIndex: 0,
    title: clean(row.preview) || clean(row.title),
    onClick: open,
    onKeyDown: event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        open()
      }
    },
    children: [
      jsx('span', { 'data-mc-dot': state }),
      jsxs('span', {
        'data-mc-main': true,
        children: [
          jsx('span', { 'data-mc-title': true, children: titleOf(row, tx('untitled')) }),
          jsx('span', { 'data-mc-meta': true, children: meta.join(' · ') })
        ]
      }),
      jsx('span', { 'data-mc-age': true, children: rightLabel })
    ]
  })
}

function CronBlock({ query, socketState }) {
  const tx = useText()
  const data = query.data
  const jobs = (data && data.jobs) || []
  const paused = jobs.filter(job => cronState(job) === 'paused')
  const scheduled = jobs.filter(job => cronState(job) !== 'paused')
  const upcoming = scheduled
    .filter(job => job.next_run_at)
    .sort((a, b) => new Date(a.next_run_at).getTime() - new Date(b.next_run_at).getTime())
    .slice(0, 3)

  const head =
    query.isLoading && !data
      ? tx('cronLoading')
      : query.isError && !data
        ? tx('cronUnavailable')
        : jobs.length
          ? tx('cronScheduled', scheduled.length) + (paused.length ? ' · ' + tx('cronPaused', paused.length) : '')
          : tx('cronNoJobs')

  const gateway = !data ? '—' : tx(data.gateway_running ? 'running' : 'notRunning')

  return jsxs('div', {
    'data-mc-foot': true,
    children: [
      jsxs('div', {
        'data-mc-foot-h': true,
        children: [
          jsx('span', { children: head }),
          jsx('span', { children: jobs.length ? tx('cronTotal', jobs.length) : '' })
        ]
      }),
      ...upcoming.map(job =>
        jsxs('div', {
          'data-mc-cron': true,
          key: job.job_id || job.name,
          children: [
            jsx('span', { 'data-mc-cdot': cronState(job) }),
            jsx('span', {
              'data-mc-cname': true,
              title: clean(job.name),
              children: clean(job.name) || job.job_id || tx('jobFallback')
            }),
            jsx('span', { children: clockOf(job.next_run_at) })
          ]
        })
      ),
      jsxs('div', {
        'data-mc-gw': true,
        children: [
          jsxs('span', { children: [tx('gateway') + ': ', jsx('b', { children: gateway })] }),
          jsxs('span', { children: [tx('socket') + ': ', jsx('b', { children: String(socketState || '—') })] })
        ]
      })
    ]
  })
}

/* ── the pane ─────────────────────────────────────────────────────────── */

function MissionControl() {
  const tx = useText()
  const busies = useValue(host.state.busyBySession)
  const socketState = useValue(host.state.gateway)
  useNowTick(15000)

  const sessionsQuery = useQuery({
    queryKey: KEY_SESSIONS,
    queryFn: () => host.request('session.list', { limit: SESSION_LIMIT }),
    refetchInterval: 20000,
    retry: 1
  })

  const activeQuery = useQuery({
    queryKey: KEY_ACTIVE,
    queryFn: () =>
      host.request('session.active_list', { current_session_id: host.state.focusedSessionId.get() || null }),
    refetchInterval: 10000,
    retry: 1
  })

  const cronQuery = useQuery({
    queryKey: KEY_CRON,
    queryFn: () => host.request('cron.manage', { action: 'list' }),
    refetchInterval: 60000,
    retry: 1
  })

  // Roster = recent stored sessions (most recent first), enriched with live
  // info when the active-session list knows the same session.
  const rows = useMemo(() => {
    const listed = (sessionsQuery.data && sessionsQuery.data.sessions) || []
    const actives = (activeQuery.data && activeQuery.data.sessions) || []
    const liveByKey = new Map()
    for (const item of actives) {
      if (!item) continue
      if (item.id) liveByKey.set(item.id, item)
      if (item.session_key) liveByKey.set(item.session_key, item)
    }
    const seen = new Set()
    const out = []
    for (const row of listed) {
      if (!row || !row.id) continue
      seen.add(row.id)
      out.push({ ...row, live: liveByKey.get(row.id) || null })
    }
    for (const item of actives) {
      if (!item) continue
      if (seen.has(item.id) || (item.session_key && seen.has(item.session_key))) continue
      const id = item.session_key || item.id
      if (!id || seen.has(id)) continue
      out.unshift({
        id,
        title: item.title || '',
        preview: item.preview || '',
        started_at: item.started_at,
        message_count: item.message_count,
        source: item.source || 'live',
        live: item
      })
      seen.add(id)
    }
    return out
  }, [sessionsQuery.data, activeQuery.data])

  const decorated = useMemo(
    () =>
      rows.map(row => ({
        row,
        working:
          isWorkingStatus(row.live && row.live.status) ||
          busies[row.id] === true ||
          Boolean(row.live && busies[row.live.id] === true)
      })),
    [rows, busies]
  )

  const workingRows = decorated.filter(entry => entry.working)
  const recentRows = decorated.filter(entry => !entry.working)

  const lastUpdated = Math.max(
    sessionsQuery.dataUpdatedAt || 0,
    activeQuery.dataUpdatedAt || 0,
    cronQuery.dataUpdatedAt || 0
  )

  const stats = []
  stats.push(tx('sessions', rows.length))
  if (workingRows.length) stats.push(tx('workingCount', workingRows.length))
  if (lastUpdated) {
    const label = ago(lastUpdated / 1000)
    stats.push(label === 'now' ? tx('updatedNow') : tx('updatedAgo', label))
  } else {
    stats.push(tx('loading'))
  }

  const sessionError =
    sessionsQuery.isError && !sessionsQuery.data
      ? clean(
          (sessionsQuery.error && sessionsQuery.error.message) || sessionsQuery.error || tx('sessionListUnavailable')
        )
      : ''

  const onOpen = useCallback(
    row => {
      haptic('tap')
      try {
        const pending = host.openSession(row.id, { intent: 'stack' })
        if (pending && typeof pending.then === 'function') {
          pending.catch(error => host.notifyError(error, tx('openFailed')))
        }
      } catch (error) {
        host.notifyError(error, tx('openFailed'))
      }
    },
    [tx]
  )

  const refresh = useCallback(() => {
    haptic('tap')
    void queryClient.invalidateQueries({ queryKey: [ID] })
  }, [])

  return jsxs('div', {
    'data-mission-control': true,
    children: [
      jsxs('div', {
        'data-mc-head': true,
        children: [
          jsx('b', { children: '⚡ ' + tx('title') }),
          jsx('span', { 'data-mc-spacer': true }),
          jsx('button', {
            type: 'button',
            'data-mc-btn': true,
            onClick: refresh,
            title: tx('refreshTip'),
            children: '↻ ' + tx('refresh')
          })
        ]
      }),
      jsx('div', { 'data-mc-stats': true, children: stats.join(' · ') }),
      jsxs('div', {
        'data-mc-scroll': true,
        children: [
          sessionError
            ? jsx('div', {
                'data-mc-note': true,
                'data-mc-err': true,
                children: tx('sessionsLabel') + ': ' + sessionError
              })
            : null,
          workingRows.length ? jsx(SectionHead, { label: tx('working'), count: String(workingRows.length) }) : null,
          ...workingRows.map(entry => jsx(SessionRow, { key: entry.row.id, row: entry.row, onOpen })),
          jsx(SectionHead, { label: tx('recent'), count: String(recentRows.length) }),
          ...recentRows.map(entry => jsx(SessionRow, { key: entry.row.id, row: entry.row, onOpen })),
          !rows.length && sessionsQuery.isLoading
            ? jsx('div', { 'data-mc-note': true, children: tx('loadingSessions') })
            : null,
          !rows.length && !sessionsQuery.isLoading && !sessionError
            ? jsx('div', { 'data-mc-note': true, children: tx('noSessions') })
            : null
        ]
      }),
      jsx(CronBlock, { query: cronQuery, socketState })
    ]
  })
}

/* ── registration ─────────────────────────────────────────────────────── */

export default {
  id: ID,
  name: 'Mission Control',
  description:
    'One pane to watch all ongoing work: live sessions with click-to-open, the cron roster with next runs, and the gateway line.',
  defaultEnabled: false,
  register(ctx) {
    ctx.i18n.register(LOCALES)

    // Disk plugins are not scanned by Tailwind; layout CSS is scoped to
    // [data-mission-control] and removed on disable.
    const style = document.createElement('style')

    style.textContent = CSS
    document.head.append(style)
    ctx.onDispose(() => style.remove())

    const bumpSessions = () => {
      void queryClient.invalidateQueries({ queryKey: KEY_SESSIONS })
      void queryClient.invalidateQueries({ queryKey: KEY_ACTIVE })
    }
    const bumpCron = () => void queryClient.invalidateQueries({ queryKey: KEY_CRON })

    // Tracked listeners — removed with the plugin on disable/reload.
    ctx.onEvent('sessions.changed', bumpSessions)
    ctx.onEvent('message.complete', bumpSessions)
    ctx.onEvent('session.title', bumpSessions)
    ctx.onEvent('cron.changed', bumpCron)
    ctx.onEvent('gateway.ready', () => {
      bumpSessions()
      bumpCron()
    })

    ctx.register({
      id: 'pane',
      area: 'panes',
      // Sampled at register (possibly before locales load); `data.tabTitle`
      // keeps the tab strip on the active locale afterwards.
      title: 'Mission control',
      order: 50,
      data: {
        placement: 'right',
        // Land on the right edge of the conversation (a split), not as a tab
        // hidden behind the sessions sidebar. Draggable afterwards.
        dock: { pane: 'workspace', pos: 'right' },
        width: '340px',
        tabTitle: () => jsx(TabTitle, {})
      },
      render: () => jsx(MissionControl, {})
    })
  }
}
