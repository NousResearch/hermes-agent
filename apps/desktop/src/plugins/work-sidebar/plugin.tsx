import {
  atom,
  cn,
  haptic,
  type HermesPlugin,
  host,
  icons,
  queryClient,
  relativeTime,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useRef, useState } from 'react'

// ── 会话阶段：纯逻辑 + 显示映射 ──────────────────────────────────────────
// ⚠️ 单文件约束：desktop-plugins 运行期加载器把 plugin.js 打成单个 blob
// import()，不支持相对模块导入，故本段内联自 phase.js（work/sources 下的
// 测试规范源）。修改必须两边同步：phase.js + 本段，并跑 test/phase.test.mjs
// （含一致性断言，防漂移）。
const ACTIVITY_FRESH_MS = 180000

type PhaseKey = 'running' | 'planning' | 'testing' | 'active' | 'done' | 'idle'

// 单一枚举源：状态 → { label, className }。derivePhase 只返回状态 key，
// 显示层一律查这张表，禁止再手写第二套分支（曾因 phaseLabel 漏 planning
// 分支，导致「badge 高亮但文字显示空闲」）。
const PHASES: Record<PhaseKey, { label: string; className: string }> = {
  running: { label: '执行中', className: 'text-(--ui-accent)' },
  planning: { label: '计划中', className: 'text-(--ui-accent)' },
  testing: { label: '测试中', className: 'text-(--ui-accent)' },
  active: { label: '活跃中', className: 'text-(--ui-text-secondary)' },
  done: { label: '已完成', className: 'text-(--ui-text-secondary)' },
  idle: { label: '空闲', className: 'text-(--ui-text-tertiary)' }
}

function phaseLabel(phase: PhaseKey): string {
  return (PHASES[phase] || PHASES.idle).label
}

function phaseClassName(phase: PhaseKey): string {
  return (PHASES[phase] || PHASES.idle).className
}

function derivePhase(todos: TodoItem[] | null | undefined, activity: ActivityItem[]): PhaseKey {
  const list = Array.isArray(activity) ? activity : []

  // 统一新鲜度过滤：所有分支只认最近 ACTIVITY_FRESH_MS 内的活动。
  // 无 todo 分支过去不过期判断，旧测试记录会让静止会话永久显示
  // 「测试中/活跃中」——与有 todo 分支的行为不一致。
  const freshActivity = list.filter(item => Date.now() - item.time < ACTIVITY_FRESH_MS)

  // todos 是主信号——有进行中/待处理时优先表达 todo 状态；
  // 全部完成/取消后再看 activity：agent 可能仍在收尾（测试/产出）。
  if (Array.isArray(todos) && todos.length) {
    if (todos.some(item => item.status === 'in_progress')) {return 'running'}

    if (todos.some(item => item.status === 'pending')) {return 'planning'}

    if (freshActivity.some(item => item.type === 'test')) {return 'testing'}

    if (freshActivity.length) {return 'active'}

    return 'done'
  }

  if (freshActivity.some(item => item.type === 'test')) {return 'testing'}

  if (freshActivity.length) {return 'active'}

  return 'idle'
}
// ── 会话阶段 end ─────────────────────────────────────────────────────────

const PLUGIN_ID = 'work-sidebar'
const TODO_STATUSES = ['pending', 'in_progress', 'completed', 'cancelled']

const todosAtom = atom<TodoItem[] | null>(null)
const activityAtom = atom<ActivityItem[]>([])
const titleAtom = atom('')

// 快照后端返回的活动/产物条目（后端 plugin_api.py 结构，见
// plugins/work-sidebar/dashboard/plugin_api.py —— 与前端必须同步）。
type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'

interface TodoItem {
  id: string
  content: string
  status: TodoStatus
}

interface ActivityItem {
  id: string
  time: number
  type: string
  text: string
  title: string
  seeded?: boolean
  count?: number
  // 折叠行附加字段（渲染层由 groupActivity 注入）
  _groupCount?: number
  _groupTexts?: string[]
  _groupItems?: ActivityItem[]
}

interface OutputItem {
  kind: string
  value: string
  label: string
  ts?: number
  purpose?: string
}

interface SnapshotData {
  todos?: TodoItem[]
  outputs?: OutputItem[]
  activity?: Array<{ time: number; type: string; text: string }>
  title?: string
  messageCount?: number
  snapshotVersion?: string
  storedId?: string
  resolved?: boolean
  _sid?: string
}

// ctx 最小表面：SDK 的 PluginContext 未全量导出，插件只用到这几个门。
interface PluginContextLike {
  rest: (path: string) => Promise<SnapshotData>
  storage: {
    get: (key: string, fallback: unknown) => unknown
    set: (key: string, value: unknown) => void
    remove: (key: string) => void
  }
  os?: {
    revealPath: (path: string) => void
    openExternal: (url: string) => void
  }
  onDispose?: (fn: () => void) => void
}

let pluginCtx: PluginContextLike | null = null
let eventDisposer: (() => void) | null = null

// 按会话缓存的快照/活动数据：切会话时优先显示目标会话的真实缓存，
// 避免先清空再等快照导致的闪空。desktop 恢复会话时首次请求可能
// 解析失败（_sessions 尚未就绪），缓存 + 轮询会逐轮收敛。
const snapshotCache = new Map<string, SnapshotData>() // sid -> snapshot data
const activityCache = new Map<string, ActivityItem[]>() // sid -> activity[]
// 已被快照活动种子填充的会话（被动事件驱动兜底）。
// Map: sid -> { fingerprint: 上次种子的 snapshotVersion }。
// 真实工具事件到达时 pushActivity 会 delete 对应条目（.has/.delete 与 Set
// 语义一致），种子随即被清出缓存，防与实时流重复；key 存在 = 该会话仍处于
// 「纯种子」态（事件流缺位），快照轮询可以刷新种子（见快照回填 effect）。
const seededSessions = new Map<string, { fingerprint: string }>()
// B2：runtime sid → stored id 映射（从快照响应学习）。desktop 的 runtime
// sid 会漂移（同一 stored 会话可对应多个 runtime sid）——事件过滤不能只
// 比实时 sid（快照侧已删掉同类比较）。事件 sid ≠ activeSessionId 时，若
// 两者映射到同一 stored id 则视为同一会话放行；否则丢弃（跨会话污染）。
// 有界，防长生命周期内存增长。
const sidToStored = new Map<string, string>()

// #9：缓存 Map 条目上限——按插入序淘汰最旧的，防长生命周期下内存无限增长
function boundedCacheSet<T>(map: Map<string, T>, key: string, value: T, max = 30): void {
  map.set(key, value)

  if (map.size > max) {
    const oldest = map.keys().next().value

    if (oldest !== undefined) {map.delete(oldest)}
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function parseTodos(value: unknown, depth = 0): TodoItem[] | null {
  if (depth > 2) {return null}

  if (Array.isArray(value)) {
    const todos = value
      .filter(item => isRecord(item) && TODO_STATUSES.includes(String(item.status)))
      .map(item => ({
        id: String(item.id ?? '').trim(),
        content: String(item.content ?? '').trim(),
        status: item.status as TodoStatus
      }))
      .filter(item => item.id && item.content)

    return todos.length ? todos : null
  }

  if (typeof value === 'string' && value.trim()) {
    try {
      return parseTodos(JSON.parse(value), depth + 1)
    } catch {
      return null
    }
  }

  if (isRecord(value) && Object.hasOwn(value, 'todos')) {
    return parseTodos(value.todos, depth + 1)
  }

  return null
}

function shorten(text: string | null | undefined, max = 68): string {
  const compact = String(text || '').replace(/\s+/g, ' ').trim()

  return compact.length > max ? `${compact.slice(0, max)}...` : compact
}

function getActivityPreview(text: string): { text: string; title: string } {
  const raw = String(text || '').replace(/\s+/g, ' ').trim()

  const cleaned = raw
    .replace(/^已修复并验证。刚才这一轮，/, '已修复并验证。')
    .replace(/^刚才这一轮，/, '')
    .replace(/表现已/g, '已')

  return {
    text: shorten(cleaned || raw, 30),
    title: raw
  }
}

function resolveTodoData({
  liveTodos,
  snapshotTodos,
  snapshotLoaded,
  onlyOpen
}: {
  liveTodos: TodoItem[] | null
  snapshotTodos: TodoItem[] | undefined
  snapshotLoaded: boolean
  onlyOpen: boolean
}): { todos: TodoItem[] | null; visibleTodos: TodoItem[]; isLoading: boolean } {
  const todos = Array.isArray(liveTodos)
    ? liveTodos
    : Array.isArray(snapshotTodos)
      ? snapshotTodos
      : snapshotLoaded
        ? []
        : null

  const visibleTodos = Array.isArray(todos)
    ? (
        onlyOpen
          ? todos.filter(item => item.status !== 'completed' && item.status !== 'cancelled')
          : todos
      )
    : []

  return {
    todos,
    visibleTodos,
    isLoading: todos === null
  }
}

function getTodoPanelLayout({
  todoCount,
  activityCount,
  outputCount
}: {
  todoCount: number
  activityCount: number
  outputCount: number
}): { todoClassName: string; activityClassName: string; outputsClassName: string } {
  if (todoCount > 0) {
    return {
      todoClassName: 'flex min-h-0 flex-1 flex-col',
      activityClassName: 'max-h-44 shrink-0',
      outputsClassName: outputCount > 0 ? 'max-h-40' : 'max-h-32'
    }
  }

  return {
    todoClassName: 'shrink-0',
    activityClassName: activityCount > 0 ? 'flex-1 min-h-[140px]' : 'max-h-44 shrink-0',
    outputsClassName: 'max-h-32'
  }
}

function pushActivity(next: ActivityItem): void {
  // 单一来源：host.state.activeSessionId（desktop 的会话状态由它驱动）
  const sid = host.state.activeSessionId.get()
  let list = sid ? (activityCache.get(sid) || []) : activityAtom.get()

  // 真实工具事件到达 → 快照种子已完成兜底使命，移除防重复。
  // 仅工具类事件清种子；message.complete（info）照常叠加，不清。
  if (sid && next.type !== 'info' && seededSessions.has(sid)) {
    seededSessions.delete(sid)
    list = list.filter(item => !item.seeded)
  }

  const newest = list[0]

  if (newest && newest.type === next.type && newest.text === next.text) {
    // 连续同类型同文本 → 合并计数（同一工具反复调用不再刷屏），
    // 时间戳刷新到最近一次，relativeTime 显示的是最后一次调用
    newest.count = (newest.count || 1) + 1
    newest.time = next.time
  } else {
    list.unshift(next)
  }

  const capped = list.slice(0, 50)

  if (sid) {
    boundedCacheSet(activityCache, sid, capped)
  }

  // activityAtom 镜像当前会话缓存；合并时更新的是缓存里的同一对象，
  // 拷贝数组引用即可触发订阅方重渲染
  activityAtom.set([...capped])
}

// 类型标签（显示层）：内部 type 保持 test/write 原值（phase.js derivePhase 依赖），
// 渲染时经 TYPE_LABELS 映射成用户可见的固定标签
const TYPE_LABELS: Record<string, string> = { terminal: 'terminal', skill: 'skill', test: 'check', write: 'output', read: 'read', tool: 'tool' }

interface ActivityGroup {
  label: string
  items: ActivityItem[]
  count: number
  texts: string[]
}

// 同类型折叠（渲染层）：按 badge 标签分组，组间按首字母排序；
// ≥2 条折叠成一行（×N = 组内总次数），hover 看组内全部文本
function groupActivity(items: ActivityItem[]): ActivityGroup[] {
  const groups = new Map<string, ActivityItem[]>()

  for (const item of items) {
    const key = TYPE_LABELS[item.type] || item.type || 'tool'

    if (!groups.has(key)) {groups.set(key, [])}
    groups.get(key)!.push(item)
  }

  return [...groups.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([label, list]) => ({
      label,
      items: list,
      count: list.reduce((s, it) => s + (it.count || 1), 0),
      texts: list.map(it => it.text)
    }))
}

function inferActivityType(toolName: string): string {
  const name = String(toolName || '')

  if (/terminal/i.test(name)) {return 'terminal'}

  if (/skill/i.test(name)) {return 'skill'}

  if (/test|pytest|vitest|check|verify/i.test(name)) {return 'test'}

  if (/read|search|list|get|view|inspect/i.test(name)) {return 'read'}

  if (/write|patch|edit|create|install|build/i.test(name)) {return 'write'}

  return 'tool'
}

// 工具活动文案：仅有工具名太干，尽力从 payload 里带出上下文
// （args 字符串 / 常见路径字段）。取不到就退回工具名。
function toolActivityText(payload: Record<string, unknown> | undefined | null): string {
  const name = String(payload?.name || 'tool')
  const args = payload?.args

  if (typeof args === 'string' && args.trim()) {
    const compact = args.replace(/\s+/g, ' ').trim()

    if (compact.length <= 40) {return `${name}: ${compact}`}
  }

  if (isRecord(args)) {
    // 字段白名单要够宽：白名单外的不同参数（如不同文件）会被并成
    // 一条「read_file ×N」，丢失具体参数——这是 tool.progress 合并的细节损失来源
    for (const key of ['path', 'file_path', 'resolved_path', 'file', 'filename', 'target', 'url', 'command', 'query', 'pattern', 'keyword']) {
      const value = args[key]

      if (typeof value === 'string' && value.trim() && value.trim().length <= 60) {
        return `${name}: ${value.trim()}`
      }
    }
  }

  return name
}

function isExplicitTodoPayload(value: unknown): boolean {
  return Array.isArray(value) || (isRecord(value) && Object.hasOwn(value, 'todos'))
}

interface GatewayEvent {
  session_id?: string
  type?: string
  payload?: Record<string, unknown>
}

function ensureAdapter(): void {
  if (eventDisposer) {return}

  eventDisposer = host.onEvent('*', event => {
    // 单一来源：直接读 host.state.activeSessionId，避免 sessionAtom 双源竞争
    const activeSessionId = host.state.activeSessionId.get()

    // B2：漂移容错过滤。runtime sid 会漂移（快照回填处已删掉实时值比较，
    // 见 queryFn 注释）——事件 sid 与 activeSessionId 不同时，不再直接
    // return，而是看两者是否解析到同一 stored id（映射从快照响应学习）：
    // 同一会话的旧 sid 事件仍放行，跨会话事件仍被丢弃。
    if (event?.session_id && activeSessionId) {
      const storedEvent = sidToStored.get(event.session_id)
      const storedActive = sidToStored.get(activeSessionId)

      const sameSession =
        event.session_id === activeSessionId || !!(storedEvent && storedEvent === storedActive)

      if (!sameSession) {return}
    }

    const payload = (event?.payload || {}) as Record<string, unknown>
    const type = String(event?.type || '')
    const isToolEvent = type === 'tool.start' || type === 'tool.progress' || type === 'tool.complete'

    if (isToolEvent && (payload.name === 'todo' || (!payload.name && Object.hasOwn(payload, 'todos')))) {
      const rawTodoValue = payload.todos ?? payload.result ?? payload.args ?? payload
      const parsed = parseTodos(rawTodoValue)

      let nextTodos: TodoItem[] | null = null

      if (parsed) {
        nextTodos = parsed
      } else if (isExplicitTodoPayload(rawTodoValue)) {
        nextTodos = []
      }

      if (nextTodos !== null && activeSessionId) {
        todosAtom.set(nextTodos)
        // 写穿到快照缓存：todo 事件即最新真值，不等下次轮询回填。
        // 否则切走再切回时 snapshotCache 里还是旧 todos，闪回旧列表。
        const cached = snapshotCache.get(activeSessionId)

        if (isRecord(cached)) {
          boundedCacheSet(snapshotCache, activeSessionId, { ...cached, todos: nextTodos } as SnapshotData)
        }
      }

      queryClient.invalidateQueries({ queryKey: ['work-sidebar', 'snapshot', activeSessionId] })

      return
    }

    if (isToolEvent) {
      const toolName = String(payload.name || 'tool')

      if (toolName !== 'todo') {
        const text = toolActivityText(payload)
        pushActivity({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          time: Date.now(),
          type: inferActivityType(toolName),
          text,
          title: text
        })
      }

      return
    }

    if (type === 'message.complete') {
      const text = typeof payload.text === 'string' ? payload.text : payload.content

      if (typeof text === 'string' && text.trim()) {
        const preview = getActivityPreview(text)
        pushActivity({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          time: Date.now(),
          type: 'info',
          text: preview.text,
          // tooltip 用截断版：完整原文可能数千字符，悬停铺满屏无意义
          title: shorten(preview.title, 200)
        })
      }

      // message.complete 不受 tool_progress 开关控制 —— tool_progress 关闭时
      // tool 事件缺失，这条消息就是自然的刷新信号：立即重拉快照，不等轮询
      if (activeSessionId) {
        queryClient.invalidateQueries({ queryKey: ['work-sidebar', 'snapshot', activeSessionId] })
      }
    }
  })
}

function useStoredState<T>(key: string, initialValue: T): [T, (next: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      return (pluginCtx?.storage.get(key, initialValue) as T | null) ?? initialValue
    } catch {
      return initialValue
    }
  })

  const update = (next: T): void => {
    setValue(next)

    try {
      pluginCtx?.storage.set(key, next)
    } catch {
      // storage 不可用时静默降级为纯内存状态
    }
  }

  return [value, update]
}

function statusIcon(todo: TodoItem) {
  if (todo.status === 'completed') {
    return <icons.CheckCircle2 className="mt-0.5 shrink-0 text-(--ui-accent)" size={13} />
  }

  if (todo.status === 'in_progress') {
    return <icons.Loader2 className="mt-0.5 shrink-0 animate-spin text-(--ui-accent)" size={13} />
  }

  if (todo.status === 'cancelled') {
    return <icons.X className="mt-0.5 shrink-0 text-(--ui-text-quaternary)" size={13} />
  }

  // #12：pending 用空心圆点，不再复用 Check（勾 = 已完成，视觉误导）
  return <span className="mt-[7px] size-1.5 shrink-0 rounded-full border border-(--ui-text-quaternary)" />
}

function TodoRow({ todo }: { todo: TodoItem }) {
  const ref = useRef<HTMLLIElement | null>(null)
  const [tip, setTip] = useState<{ left: number; top: number } | null>(null)

  const showTip = () => {
    const el = ref.current

    if (!el) {return}
    const r = el.getBoundingClientRect()
    // 浮层右缘对齐行右缘、向下展开；宽 280 向左收，防溢出视口
    setTip({ left: Math.max(8, r.right - 280), top: r.bottom + 2 })
  }

  return (
    <li
      className="flex items-start gap-2 px-3 py-1.5"
      onMouseEnter={showTip}
      onMouseLeave={() => setTip(null)}
      ref={ref}
    >
      {statusIcon(todo)}
      <span
        className={cn(
          'min-w-0 flex-1 truncate leading-snug',
          todo.status === 'completed' && 'text-(--ui-text-quaternary) line-through',
          todo.status === 'in_progress' && 'text-(--ui-text-secondary)',
          todo.status === 'cancelled' && 'text-(--ui-text-quaternary) line-through opacity-60'
        )}
      >
        {todo.content}
      </span>
      {/* hover 预览浮层：完整任务文本（原生 title 在 Electron 环境不可靠，自绘） */}
      {tip && (
        <div
          className="fixed z-50 max-h-[200px] max-w-[280px] overflow-auto rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-2.5 py-2 text-[0.75rem] leading-snug text-(--ui-text-secondary) shadow-lg"
          style={{ left: tip.left, top: tip.top }}
        >
          {todo.content}
        </div>
      )}
    </li>
  )
}

function ActivityRow({ item }: { item: ActivityItem }) {
  const ref = useRef<HTMLLIElement | null>(null)
  const [tip, setTip] = useState<{ left: number; top: number } | null>(null)

  const tone =
    item.type === 'write' || item.type === 'test'
      ? 'bg-(--ui-accent)'
      : item.type === 'info'
        ? 'bg-(--ui-text-quaternary)'
        : 'bg-(--ui-text-secondary)'

  const isGrouped = Array.isArray(item._groupItems) && item._groupItems.length > 1

  const openTip = () => {
    const el = ref.current

    if (!el) {return}
    const r = el.getBoundingClientRect()
    // 浮层右缘对齐行右缘、向下展开；宽 320 向左收，防溢出视口
    setTip({ left: Math.max(8, r.right - 320), top: r.bottom + 2 })
  }

  // 组详情：最新改动在上（time 倒序）
  const groupSorted = isGrouped
    ? [...(item._groupItems as ActivityItem[])].sort((a, b) => (b.time || 0) - (a.time || 0))
    : []

  return (
    <li
      className="flex items-start gap-2 px-3 py-1.5 text-[0.75rem]"
      ref={ref}
      title={item._groupTexts ? item._groupTexts.join('\n') : item.title || item.text}
    >
      <span className={cn('mt-1 size-1.5 shrink-0 rounded-full', tone)} />
      {item.type !== 'info' ? (
        <span className="shrink-0 min-w-16 rounded bg-(--ui-stroke-secondary) px-1 py-px text-center text-[0.625rem] leading-tight text-(--ui-text-tertiary)">
          {TYPE_LABELS[item.type] || item.type}
        </span>
      ) : null}
      <span className="min-w-0 flex-1 truncate leading-snug text-(--ui-text-secondary)">
        {item.text}
      </span>
      {/* 计数徽标：普通行 = 连续合并计数；折叠行 = 组内总次数，点击弹组详情 */}
      {(item._groupCount || item.count || 0) > 1 ? (
        isGrouped ? (
          <button
            className="shrink-0 self-center cursor-pointer rounded bg-(--ui-stroke-secondary) px-1 text-[0.625rem] text-(--ui-text-quaternary) hover:text-(--ui-accent)"
            onClick={() => (tip ? setTip(null) : openTip())}
            type="button"
          >
            {`×${item._groupCount}`}
          </button>
        ) : (
          <span className="shrink-0 self-center rounded bg-(--ui-stroke-secondary) px-1 text-[0.625rem] text-(--ui-text-quaternary)">
            {`×${item.count}`}
          </span>
        )
      ) : null}
      <span className="shrink-0 pt-0.5 text-[0.625rem] text-(--ui-text-quaternary)">
        {relativeTime(item.time)}
      </span>
      {/* 组详情浮层：点击 ×N 弹出，最新改动在上 */}
      {tip && isGrouped && (
        <div
          className="fixed z-50 max-h-[240px] w-[320px] overflow-auto rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) py-1 shadow-lg"
          style={{ left: tip.left, top: tip.top }}
        >
          {groupSorted.map(g => (
            <div className="flex items-start gap-2 px-2.5 py-1 text-[0.75rem]" key={g.id}>
              <span className="mt-1.5 size-1 shrink-0 rounded-full bg-(--ui-text-tertiary)" />
              <span className="min-w-0 flex-1 truncate text-(--ui-text-secondary)" title={g.text}>
                {g.text}
              </span>
              {(g.count || 1) > 1 ? (
                <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">
                  {`×${g.count}`}
                </span>
              ) : null}
              <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">
                {relativeTime(g.time)}
              </span>
            </div>
          ))}
        </div>
      )}
      {/* 点击浮层外关闭 */}
      {tip && isGrouped && <div className="fixed inset-0 z-40" onClick={() => setTip(null)} />}
    </li>
  )
}

function OutputRow({ item }: { item: OutputItem }) {
  const icon =
    item.kind === 'image' ? (
      <icons.FileImage className="shrink-0 text-(--ui-text-secondary)" size={13} />
    ) : item.kind === 'link' ? (
      <icons.Link2 className="shrink-0 text-(--ui-text-secondary)" size={13} />
    ) : (
      <icons.FileText className="shrink-0 text-(--ui-text-secondary)" size={13} />
    )

  // 文件卡片：主行 = 文件名（点击打开/定位），副行 = 完整路径/URL 摘要。
  // 后端快照只有 {kind, value, label, ts}——摘要诚实显示位置，不编造用途。
  const labelEl = item.value.startsWith('data:') ? (
    <span className="block truncate text-[0.75rem] text-(--ui-text-quaternary)" title="内联数据，无法打开">
      {item.label}
    </span>
  ) : (
    <button
      className="block w-full truncate text-left text-[0.75rem] text-(--ui-text-secondary) hover:text-(--ui-accent)"
      onClick={() => {
        haptic('tap')

        if (item.kind === 'file') {
          pluginCtx?.os?.revealPath(item.value)
        } else {
          pluginCtx?.os?.openExternal(item.value)
        }
      }}
      title={item.value}
      type="button"
    >
      {item.label}
    </button>
  )

  const subEl = item.value.startsWith('data:') ? (
    <div className="truncate text-[0.6875rem] text-(--ui-text-quaternary)">内联数据（无法打开）</div>
  ) : (
    <div className="truncate text-[0.6875rem] text-(--ui-text-quaternary)" title={item.value}>
      {item.value}
    </div>
  )

  // 用途摘要行：后端结构化提取带 tool_name → 用途短语；正则兜底无来源则隐藏
  const purposeEl = item.purpose ? (
    <div className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">{item.purpose}</div>
  ) : null

  return (
    <li className="group flex items-start gap-2 rounded-md px-3 py-2 hover:bg-(--ui-stroke-secondary)">
      <div className="mt-0.5 shrink-0">{icon}</div>
      <div className="min-w-0 flex-1">
        {labelEl}
        {subEl}
        {purposeEl}
      </div>
    </li>
  )
}

function WorkSidebarPane() {
  const activeSessionId = useValue(host.state.activeSessionId)
  const liveTodos = useValue(todosAtom)
  const activity = useValue(activityAtom)
  const title = useValue(titleAtom)
  const [collapsed, setCollapsed] = useStoredState('collapsed', false)
  const [onlyOpen, setOnlyOpen] = useStoredState('onlyOpen', false)
  // #13：Activity/Outputs 展开状态（默认折叠，展开看全量）
  const [activityExpanded, setActivityExpanded] = useStoredState('activityExpanded', false)
  const [outputsExpanded, setOutputsExpanded] = useStoredState('outputsExpanded', false)
  // 当前快照数据归属的会话：切换会话时旧值先保留，新快照回填后才切过去，
  // 避免短暂空白/闪烁（等新数据期间显示骨架屏遮罩）。
  const [shownSessionId, setShownSessionId] = useState<string | null>(null)

  useEffect(() => {
    if (activeSessionId) {
      // 切到目标会话：
      // 1. 有快照缓存 → 立即显示该会话的真实数据（不遮罩、不闪空）
      // 2. 无缓存 → 保留旧值 + 遮罩等待，快照到达后再切换
      // 快照拉取统一由 useQuery 负责：同 queryKey 缓存共享，key 变化自动
      // fetch（tool_progress 关闭时也立即拉，不等轮询）；不再手动
      // fetchQuery + invalidateQueries 造成同 key 双通道冗余请求。
      const cached = snapshotCache.get(activeSessionId)

      if (cached) {
        setShownSessionId(activeSessionId)
        todosAtom.set(Array.isArray(cached.todos) ? cached.todos : [])
        titleAtom.set(cached.title || '')
      } else {
        setShownSessionId(null)
      }

      // activity 按会话切换：有缓存显示缓存；无缓存保留旧值
      // （等新会话事件/快照自然替换，避免先清空导致的闪空）
      const cachedActivity = activityCache.get(activeSessionId)

      if (cachedActivity) {
        activityAtom.set([...cachedActivity])
      } else {
        // #8：无缓存时清空而非保留旧会话值——避免 Activity 串会话误导
        activityAtom.set([])
      }
    } else {
      // 无活跃会话：清空是合理的（真的没有会话了）
      setShownSessionId(null)
      todosAtom.set(null)
      activityAtom.set([])
      titleAtom.set('')
    }
  }, [activeSessionId])

  const snapshot = useQuery({
    queryKey: ['work-sidebar', 'snapshot', activeSessionId],
    enabled: !!activeSessionId,
    queryFn: async () => {
      // 请求 sid 随数据返回。注意：`_sid` 与回填 effect 里的 activeSessionId
      // 同源于同一渲染闭包，恒相等 —— 它不是漂移防护。真正的修复是删除
      // 原先与 host.state.activeSessionId（实时值）的比较：desktop 的
      // runtime sid 会漂移（同一 stored 会话可对应多个 runtime sid），
      // 实时值 ≠ 渲染捕获值时误 return → 快照永不回填，遮罩永存。
      // 保留 `_sid` 仅为显式表达「快照归属于请求时的 sid」这一不变式防回归。
      const sid = activeSessionId ?? ''
      const data = await pluginCtx!.rest(`/snapshot?session_id=${encodeURIComponent(sid)}`)

      return { ...data, _sid: sid }
    },
    staleTime: 2000,
    // 失败时暂停轮询：后端持续 500（db 锁/损坏）时停止每 2s 打点，
    // 避免与 gateway 写锁争用放大；恢复路径 = 手动「重试」或会话切换。
    refetchInterval: query => {
      // 折叠时暂停轮询（#6）：面板不可见还每 2s 打 DB 是纯浪费
      if (collapsed) {return false}

      return query.state.error ? false : 2000
    },
    retry: 2
  })

  // 按会话记录的 snapshotVersion（后端 = total|last_active_id|title），
  // 用作「数据是否变化」的变化指示器。
  // 之前是单值 ref，跨会话共享：两个 >400 条的会话 messageCount 都被
  // 后端截断为固定值，切换后新会话首个快照会被误判为「未变化」而跳过
  // 回填 → 新会话数据冻结。改为按 activeSessionId 分键，只与本会话
  // 上次值比较，不跨会话污染；条目数受限防无限增长。
  // 旧后端没有 snapshotVersion 时退化为 messageCount 版本串（原行为）。
  const lastVersionBySessionRef = useRef(new Map<string, string>())

  useEffect(() => {
    if (!snapshot.data || !activeSessionId) {return}

    // 归属校验（恒真：`_sid` 与 activeSessionId 同源于 queryFn 所在渲染
    // 闭包）。保留它仅为防回归 —— 真正的修复是删除与 host.state.
    // activeSessionId 实时值的比较（runtime sid 漂移会让实时值 ≠ 渲染捕获
    // 值，误杀回填）。react-query 按 queryKey 隔离数据，本就不会把旧会话
    // 快照带进当前 key。
    if (snapshot.data._sid !== activeSessionId) {return}

    // 快照版本未变化且当前已展示该会话数据 → 跳过回填。
    // 2s 轮询期间数据没变就反复 set atoms 会引发不必要重渲染/闪动；
    // snapshotVersion（total|last_active_id|title）比 messageCount 覆盖更全：
    // 新增消息 / rewind（soft-delete 行数不变但 active 窗口收缩）/ 改标题
    // 都会让版本变化——messageCount 只覆盖「新增消息」一种。
    // 旧后端无 snapshotVersion 时退化为 messageCount 版本串（原行为）。
    const version = snapshot.data.snapshotVersion ?? `m${snapshot.data.messageCount}`
    const lastVersion = lastVersionBySessionRef.current.get(activeSessionId)

    if (
      lastVersion === version &&
      shownSessionId === activeSessionId
    ) {
      return
    }

    lastVersionBySessionRef.current.set(activeSessionId, version)

    // 限制条目数：老会话的版本不再有用，删掉最早插入的键
    if (lastVersionBySessionRef.current.size > 50) {
      const oldest = lastVersionBySessionRef.current.keys().next().value

      if (oldest !== undefined) {lastVersionBySessionRef.current.delete(oldest)}
    }

    // 快照回填：写缓存 + 切到该会话的数据（旧值只保留到这一刻）
    // 解析失败的快照不缓存（避免切回时永久显示空且无提示）
    if (snapshot.data.resolved !== false) {
      boundedCacheSet(snapshotCache, activeSessionId, snapshot.data)

      // B2：学习 runtime sid → stored id 映射（事件过滤的漂移容错锚点）。
      // 解析失败的快照 storedId=原 sid，记录无害（同 sid 相等判定不受影响）。
      if (typeof snapshot.data.storedId === 'string' && snapshot.data.storedId) {
        boundedCacheSet(sidToStored, activeSessionId, snapshot.data.storedId, 50)
      }
    }

    setShownSessionId(activeSessionId)
    todosAtom.set(Array.isArray(snapshot.data.todos) ? snapshot.data.todos : [])
    // 无条件回填 title：切到无标题会话时回到 '当前会话'，
    // 不再残留上一个会话的标题（旧实现只在非空时更新，导致标题错位）
    titleAtom.set(snapshot.data.title || '')

    // 被动事件驱动兜底：事件流缺位（tool_progress 关闭 / agent 不 emit）时，
    // 用快照里的最近活动种子填充 Activity，不再永远「暂无活动」。
    // 真实工具事件到达时由 pushActivity 清除种子（seededSessions.delete），
    // 防与实时流重复。种子不是一次性的：事件流持续缺位时，快照轮询拿到的
    // 更新活动要能覆盖旧种子（否则 Activity 冻结在首次恢复值）。
    // seed 行 id 不含 messageCount（稳定 key），刷新时 React 复用 DOM 不闪动。
    const makeSeed = (activityList: SnapshotData['activity']): ActivityItem[] =>
      (activityList || []).map((entry, index) => ({
        id: `seed-${activeSessionId}-${index}`,
        time: entry.time,
        type: entry.type,
        text: entry.text,
        title: entry.text,
        seeded: true
      }))

    // 首次播种：缓存为空且快照有活动
    if (
      !activityCache.get(activeSessionId) &&
      Array.isArray(snapshot.data.activity) &&
      snapshot.data.activity.length
    ) {
      const seeded = makeSeed(snapshot.data.activity)
      seededSessions.set(activeSessionId, { fingerprint: version })
      activityCache.set(activeSessionId, seeded)
      activityAtom.set([...seeded])
    }

    // 种子刷新：仅在「纯种子态」下用新快照覆盖旧种子。三个条件全部满足才写：
    //   1. 当前缓存全是 seed（有 live 项混入 → 实时流优先，快照不覆盖）
    //   2. 会话仍处于种子态（seededSessions 未被真实工具事件 delete）
    //   3. 快照版本戳变化（复用上方 version：total/rewind/改标题任一变化即刷新；
    //      version 未变则跳过，避免每 2s 轮询重写相同内容造成闪动）
    const seededList = activityCache.get(activeSessionId)
    const allSeeded = Array.isArray(seededList) && seededList.length && seededList.every(item => item.seeded)

    if (
      allSeeded &&
      seededSessions.has(activeSessionId) &&
      Array.isArray(snapshot.data.activity) &&
      snapshot.data.activity.length &&
      version !== seededSessions.get(activeSessionId)!.fingerprint
    ) {
      const refreshed = makeSeed(snapshot.data.activity)
      seededSessions.set(activeSessionId, { fingerprint: version })
      activityCache.set(activeSessionId, refreshed)
      activityAtom.set([...refreshed])
    }
  }, [snapshot.data, activeSessionId, shownSessionId])

  const switching = activeSessionId !== shownSessionId

  const todoData = resolveTodoData({
    liveTodos,
    snapshotTodos: snapshot.data?.todos,
    // B6：无活跃会话时快照查询被禁用（enabled:false）且从未 fetch——
    // isFetched 恒为 false，snapshotLoaded 恒为 false → todo 区永远显示
    // 加载骨架。无会话本身就是「加载完的空态」，直接视为已加载：
    // 与 Activity/Outputs 一致显示空态，而不是无限骨架。
    snapshotLoaded: !!snapshot.data || snapshot.isFetched || !activeSessionId,
    onlyOpen
  })

  const phase = derivePhase(todoData.todos || [], activity || [])
  const total = Array.isArray(todoData.todos) ? todoData.todos.length : 0

  const cancelled = Array.isArray(todoData.todos)
    ? todoData.todos.filter(item => item.status === 'cancelled').length
    : 0

  const completed = Array.isArray(todoData.todos)
    ? todoData.todos.filter(item => item.status === 'completed').length
    : 0

  // 进度分母排除 cancelled：取消的任务不该拉低完成率
  const activeTotal = total - cancelled
  const percent = activeTotal ? Math.round((completed / activeTotal) * 100) : 0

  // 布局用的 activity 数取当前会话缓存（activityAtom 可能保留旧会话值，
  // 用全局长度判断弹性布局会不准）
  const currentActivityCount = (activeSessionId && activityCache.get(activeSessionId))?.length || 0

  const panelLayout = useMemo(
    () => getTodoPanelLayout({
      todoCount: total,
      activityCount: currentActivityCount,
      outputCount: snapshot.data?.outputs?.length || 0
    }),
    [currentActivityCount, snapshot.data?.outputs?.length, total]
  )

  if (collapsed) {
    return (
      <div className="flex h-full flex-col items-center gap-3 border-l border-(--ui-stroke-secondary) bg-(--ui-editor-surface-background) py-3">
        <button
          className="rounded p-1 text-(--ui-text-secondary) hover:text-(--ui-accent)"
          onClick={() => {
            haptic('tap')
            setCollapsed(false)
          }}
          title="展开 Work Sidebar"
          type="button"
        >
          <icons.ChevronLeft size={16} />
        </button>
        <icons.Clipboard className="text-(--ui-text-quaternary)" size={16} title={`TODO ${total}`} />
        <icons.Activity className="text-(--ui-text-quaternary)" size={16} title={`活动 ${currentActivityCount}`} />
        <icons.FileText className="text-(--ui-text-quaternary)" size={16} title={`产物 ${snapshot.data?.outputs?.length || 0}`} />
      </div>
    )
  }

  return (
    <div className="relative flex h-full min-w-[280px] max-w-[360px] flex-col border-l border-(--ui-stroke-secondary) bg-(--ui-editor-surface-background) text-sm">
      {/* isError 时不显示遮罩：否则「加载会话数据…」会盖住下方「快照加载
          失败 [重试]」横幅（z-10 > 横幅默认层级），重试不可达 → 用户误以为
          卡死。错误态让横幅在正常文档流里露出，恢复后遮罩自动收敛。 */}
      {switching && !snapshot.isError ? (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-(--ui-editor-surface-background)/70 text-[0.75rem] text-(--ui-text-tertiary)">
          加载会话数据…
        </div>
      ) : null}
      {snapshot.isError || (snapshot.data && snapshot.data.resolved === false) ? (
        <div className="flex items-center gap-2 border-b border-(--ui-stroke-secondary) px-3 py-1.5 text-[0.6875rem]">
          <span className="flex-1 text-red-400">
            {snapshot.isError ? '快照加载失败' : '会话数据不可用（已删除或已过期）'}
          </span>
          <button
            className="rounded px-1.5 py-0.5 text-(--ui-accent) hover:opacity-80"
            onClick={() => {
              haptic('tap')
              snapshot.refetch()
            }}
            title="重新加载"
            type="button"
          >
            重试
          </button>
        </div>
      ) : null}
      <div className="flex flex-col gap-2 border-b border-(--ui-stroke-secondary) px-3 py-2.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className="min-w-0 truncate font-medium text-(--ui-text-secondary)">
              {title || '当前会话'}
            </span>
            <span
              className={cn(
                'shrink-0 rounded-full border border-(--ui-stroke-secondary) px-1.5 py-px text-[0.625rem]',
                phaseClassName(phase)
              )}
            >
              {phaseLabel(phase)}
            </span>
          </div>
          <button
            className="shrink-0 rounded p-1 text-(--ui-text-tertiary) hover:text-(--ui-accent)"
            onClick={() => {
              haptic('tap')
              setCollapsed(true)
            }}
            title="折叠 Work Sidebar"
            type="button"
          >
            <icons.ChevronRight size={15} />
          </button>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-(--ui-stroke-secondary)">
            <div
              className="h-full rounded-full bg-(--ui-accent) transition-[width] duration-300"
              style={{ width: `${percent}%` }}
            />
          </div>
          <span
            className="text-[0.6875rem] tabular-nums text-(--ui-text-tertiary)"
            // 分母排除 cancelled 时完成率≠直觉总数：悬停给出上下文，
            // 不让「5/10」在共有 12 项且 2 项已取消时显得算错
            title={cancelled > 0 ? `共 ${total} 项，其中 ${cancelled} 项已取消` : undefined}
          >
            {`${completed}/${activeTotal}`}
          </span>
        </div>
      </div>
      <div className={panelLayout.todoClassName}>
        <div className="flex items-center justify-between px-3 pt-2 pb-1">
          <span className="text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-(--ui-text-quaternary)">
            Todo
          </span>
          <button
            className={cn(
              'text-[0.6875rem]',
              onlyOpen ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)'
            )}
            onClick={() => setOnlyOpen(!onlyOpen)}
            type="button"
          >
            {onlyOpen ? '显示全部' : '只看未完成'}
          </button>
        </div>
        <div className={cn('overflow-y-auto', total > 0 ? 'min-h-0 flex-1' : 'px-3 pb-3')}>
          {todoData.isLoading ? (
            <div className="mx-3 mt-2 h-4 animate-pulse rounded bg-(--ui-stroke-secondary)" />
          ) : !todoData.visibleTodos.length ? (
            <div
              className={cn(
                'rounded border border-dashed border-(--ui-stroke-secondary) text-center text-[0.75rem] text-(--ui-text-quaternary)',
                total > 0 ? 'mx-3 my-4 px-3 py-6' : 'px-3 py-4'
              )}
            >
              暂无待办
            </div>
          ) : (
            <ul className="divide-y divide-(--ui-stroke-secondary)">
              {todoData.visibleTodos.map(todo => <TodoRow key={todo.id} todo={todo} />)}
            </ul>
          )}
        </div>
      </div>
      <div className={cn(panelLayout.activityClassName, 'overflow-y-auto border-t border-(--ui-stroke-secondary)')}>
        <div className="px-3 pt-2 pb-1 text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-(--ui-text-quaternary)">
          Activity
        </div>
        {activity.length ? (
          <div>
            <ul>
              {(activityExpanded
                ? activity
                : groupActivity(activity)
                    .slice(0, 5)
                    .map(g => ({
                      ...g.items[0],
                      _groupCount: g.count,
                      _groupTexts: g.texts,
                      _groupItems: g.items
                    }))
              ).map(item => <ActivityRow item={item} key={item.id} />)}
            </ul>
            {activity.length > 5 ? (
              <button
                className="w-full px-3 py-1 text-left text-[0.6875rem] text-(--ui-text-tertiary) hover:text-(--ui-accent)"
                onClick={() => {
                  haptic('tap')
                  setActivityExpanded(!activityExpanded)
                }}
                type="button"
              >
                {activityExpanded ? '收起' : `展开全部 ${activity.length} 条`}
              </button>
            ) : null}
          </div>
        ) : (
          <div className="px-3 py-2 text-[0.75rem] text-(--ui-text-quaternary)">
            暂无活动
          </div>
        )}
      </div>
      <div className={cn(panelLayout.outputsClassName, 'shrink-0 overflow-y-auto border-t border-(--ui-stroke-secondary)')}>
        <div className="px-3 pt-2 pb-1 text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-(--ui-text-quaternary)">
          Outputs
        </div>
        {snapshot.data?.outputs?.length ? (
          <div>
            <ul className="divide-y divide-(--ui-stroke-secondary)">
              {(outputsExpanded ? snapshot.data.outputs : snapshot.data.outputs.slice(0, 8)).map(
                (item, index) => <OutputRow item={item} key={`${item.value}-${index}`} />
              )}
            </ul>
            {snapshot.data.outputs.length > 8 ? (
              <button
                className="w-full px-3 py-1 text-left text-[0.6875rem] text-(--ui-text-tertiary) hover:text-(--ui-accent)"
                onClick={() => {
                  haptic('tap')
                  setOutputsExpanded(!outputsExpanded)
                }}
                type="button"
              >
                {outputsExpanded ? '收起' : `展开全部 ${snapshot.data.outputs.length} 条`}
              </button>
            ) : null}
          </div>
        ) : (
          <div className="px-3 py-3 flex flex-col gap-0.5">
            {snapshot.isError ? (
              <div className="text-[0.75rem] text-(--ui-text-quaternary)">加载失败</div>
            ) : snapshot.isLoading ? (
              <div className="text-[0.75rem] text-(--ui-text-quaternary)">加载中...</div>
            ) : (
              <div className="flex flex-col gap-0.5">
                <div className="text-[0.75rem] text-(--ui-text-quaternary)">暂无产物</div>
                {/* B6：无活跃会话时隐藏「agent 写入文件…」提示语——
                    没有会话却暗示有 agent 在工作会误导。有会话但确实
                    没产物时才展示引导文案。 */}
                {activeSessionId ? (
                  <div className="text-[0.6875rem] text-(--ui-text-tertiary)">
                    agent 写入文件、截图或下载后，会自动出现在这里
                  </div>
                ) : null}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

const plugin: HermesPlugin = {
  id: PLUGIN_ID,
  name: 'Work Sidebar',
  defaultEnabled: false,
  register(ctx) {
    pluginCtx = ctx as unknown as PluginContextLike
    ensureAdapter()

    // 插件卸载/热重载时销毁事件监听，避免旧监听器累积
    // （SDK: contrib/plugin.ts createPluginContext.onDispose）
    ctx.onDispose?.(() => {
      if (eventDisposer) {
        eventDisposer()
        eventDisposer = null
      }
    })

    ctx.register({
      id: 'workspace-pane-v3',
      area: 'panes',
      title: 'Work Sidebar',
      data: {
        placement: 'right',
        dock: { pane: 'workspace', pos: 'right' },
        width: '300px'
      },
      render: () => <WorkSidebarPane />
    })
  }
}

export default plugin
