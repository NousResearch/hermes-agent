import { type FormEvent, type PointerEvent as ReactPointerEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import type { DesktopAccountEdge, DesktopAccountTreeNode } from '@/global'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

import type { TeamResource } from './team-types'

const NODE_W = 176
const NODE_H = 58
const H_GAP = NODE_W + 44
const V_GAP = NODE_H + 56

const MAX_KB_SHOWN = 4

// 知识库条目鼠标悬停摘要(chunks / words / 状态),没有就只显示名字。
function kbHint(kb: TeamResource): string {
  const meta = kb.meta ?? {}
  const parts: string[] = []

  if (typeof meta.chunks === 'number') {
    parts.push(`${meta.chunks} 块`)
  }

  if (typeof meta.words === 'number') {
    parts.push(`${meta.words} 词`)
  }

  if (typeof meta.status === 'string' && meta.status) {
    parts.push(String(meta.status))
  }

  return parts.length ? `${kb.name} · ${parts.join(' · ')}` : kb.name
}

type XY = { x: number; y: number }

type Gesture =
  | { kind: 'drag'; id: string; offX: number; offY: number }
  | { kind: 'connect'; fromId: string }
  | { kind: 'pan'; sx: number; sy: number; px: number; py: number }

export interface TeamCanvasProps {
  nodes: DesktopAccountTreeNode[]
  edges: DesktopAccountEdge[]
  rootId: string
  busy: boolean
  onConnect: (managerId: string, memberId: string) => void
  onRefresh: () => void
  onCreateSubaccount: (payload: { email: string; name: string; password: string }) => Promise<{ ok: boolean; error?: string }>
  roles: string[]
  onSetRole: (userId: string, role: string) => void
  // 每个(子)节点拉取存到主本地的资源;用于在卡片上展示「该子有哪些知识库」。
  // 角色集管理 + 授权(角色→资源)已**拿出去**到团队账号页的常驻面板(team-roles.tsx),不在画布里。
  resourcesByNode: Record<string, TeamResource[]>
}

// parent_id 树自动布局:叶子从左到右排开,父节点居中在子节点上方;depth → y。
function autoLayout(nodes: DesktopAccountTreeNode[], rootId: string): Record<string, XY> {
  const ids = new Set(nodes.map(n => n.user_id))
  const childrenOf = new Map<string, string[]>()

  for (const n of nodes) {
    if (n.user_id === rootId) {
      continue
    }

    const parent = n.parent_id && ids.has(n.parent_id) && n.parent_id !== n.user_id ? n.parent_id : rootId
    const list = childrenOf.get(parent) ?? []
    list.push(n.user_id)
    childrenOf.set(parent, list)
  }

  const pos: Record<string, XY> = {}
  const placed = new Set<string>()
  let leaf = 0

  const place = (id: string, depth: number): number => {
    if (placed.has(id)) {
      return pos[id]?.x ?? 0
    }

    placed.add(id)
    const kids = childrenOf.get(id) ?? []

    if (kids.length === 0) {
      const x = leaf * H_GAP
      leaf += 1
      pos[id] = { x, y: depth * V_GAP }

      return x
    }

    const xs = kids.map(k => place(k, depth + 1))
    const x = (Math.min(...xs) + Math.max(...xs)) / 2
    pos[id] = { x, y: depth * V_GAP }

    return x
  }

  if (ids.has(rootId)) {
    place(rootId, 0)
  }

  for (const n of nodes) {
    if (!pos[n.user_id]) {
      pos[n.user_id] = { x: leaf * H_GAP, y: 0 }
      leaf += 1
    }
  }

  return pos
}

function loadSaved(rootId: string): Record<string, XY> {
  try {
    const raw = window.localStorage.getItem(`hermes:team-canvas:v1:${rootId}`)
    const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : {}
    const out: Record<string, XY> = {}

    // 只信形状合法、坐标有限的条目;NaN/缺字段/坏数据丢弃 → 回退自动布局(在屏内),避免节点隐形。
    for (const [id, p] of Object.entries(parsed)) {
      const xy = p as { x?: unknown; y?: unknown }

      if (xy && Number.isFinite(xy.x) && Number.isFinite(xy.y)) {
        out[id] = { x: xy.x as number, y: xy.y as number }
      }
    }

    return out
  } catch {
    return {}
  }
}

// 贝塞尔:A 底部中点 → B 顶部中点(hA = A 卡片实际高度)
function edgePath(a: XY, b: XY, hA: number): string {
  const ax = a.x + NODE_W / 2
  const ay = a.y + hA
  const bx = b.x + NODE_W / 2
  const by = b.y
  const midY = (ay + by) / 2

  return `M ${ax} ${ay} C ${ax} ${midY}, ${bx} ${midY}, ${bx} ${by}`
}

export function TeamCanvas({
  busy,
  edges,
  nodes,
  onConnect,
  onCreateSubaccount,
  onRefresh,
  onSetRole,
  resourcesByNode,
  roles,
  rootId
}: TeamCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const gestureRef = useRef<Gesture | null>(null)
  const hoverRef = useRef<string | null>(null)
  const panRef = useRef<XY>({ x: 56, y: 32 })
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({})
  const heightsRef = useRef<Record<string, number>>({})

  const [positions, setPositions] = useState<Record<string, XY>>({})
  const [heights, setHeights] = useState<Record<string, number>>({})
  const [pan, setPan] = useState<XY>({ x: 56, y: 32 })
  const [connect, setConnect] = useState<{ fromId: string; cx: number; cy: number } | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)

  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState('')
  const [linkMode, setLinkMode] = useState(false)
  const [linkSource, setLinkSource] = useState<string | null>(null)

  const positionsRef = useRef(positions)
  useEffect(() => {
    positionsRef.current = positions
  }, [positions])

  // 测量每张卡实际高度(角色下拉让子账号卡更高)——命中检测/连线端点用真实高度,而非固定 NODE_H。
  useLayoutEffect(() => {
    setHeights(prev => {
      const next: Record<string, number> = {}
      let changed = false

      for (const n of nodes) {
        const el = cardRefs.current[n.user_id]
        // offsetHeight 可能为 0(未布局/jsdom)——不要把 0 当真实高度提交,回退上次或 NODE_H。
        const h = el && el.offsetHeight > 0 ? el.offsetHeight : prev[n.user_id] || NODE_H
        next[n.user_id] = h

        if (prev[n.user_id] !== h) {
          changed = true
        }
      }

      return changed ? next : prev
    })
  }, [nodes, roles, resourcesByNode])

  useEffect(() => {
    heightsRef.current = heights
  }, [heights])

  const nodeById = useMemo(() => new Map(nodes.map(n => [n.user_id, n])), [nodes])

  // 保证每个节点有坐标:已有 > localStorage > 自动布局。
  useEffect(() => {
    setPositions(prev => {
      const saved = loadSaved(rootId)
      const auto = autoLayout(nodes, rootId)
      const next: Record<string, XY> = {}

      for (const n of nodes) {
        next[n.user_id] = prev[n.user_id] ?? saved[n.user_id] ?? auto[n.user_id] ?? { x: 0, y: 0 }
      }

      return next
    })
  }, [nodes, rootId])

  const persist = useCallback(() => {
    try {
      window.localStorage.setItem(`hermes:team-canvas:v1:${rootId}`, JSON.stringify(positionsRef.current))
    } catch {
      /* localStorage 不可用就放弃持久化,不影响使用 */
    }
  }, [rootId])

  const toCanvas = useCallback((clientX: number, clientY: number): XY => {
    const rect = canvasRef.current?.getBoundingClientRect()

    return {
      x: clientX - (rect?.left ?? 0) - panRef.current.x,
      y: clientY - (rect?.top ?? 0) - panRef.current.y
    }
  }, [])

  const nodeAt = useCallback(
    (cx: number, cy: number, exclude: string): string | null => {
      for (const n of nodes) {
        if (n.user_id === exclude) {
          continue
        }

        const p = positionsRef.current[n.user_id]
        const h = heightsRef.current[n.user_id] || NODE_H

        if (p && cx >= p.x && cx <= p.x + NODE_W && cy >= p.y && cy <= p.y + h) {
          return n.user_id
        }
      }

      return null
    },
    [nodes]
  )

  const startDrag = (event: ReactPointerEvent, id: string) => {
    if (event.button !== 0) {
      return
    }

    event.stopPropagation()

    // 连线模式:点选不拖动 —— 第一下选上级,第二下选下级即连上。
    if (linkMode) {
      if (!linkSource) {
        setLinkSource(id)
      } else if (id !== linkSource) {
        if (!edges.some(e => e.manager_id === linkSource && e.member_id === id)) {
          onConnect(linkSource, id)
        }

        setLinkSource(null)
      } else {
        setLinkSource(null)
      }

      return
    }

    const c = toCanvas(event.clientX, event.clientY)
    const p = positionsRef.current[id] ?? { x: 0, y: 0 }
    gestureRef.current = { kind: 'drag', id, offX: c.x - p.x, offY: c.y - p.y }
    canvasRef.current?.setPointerCapture(event.pointerId)
  }

  const startConnect = (event: ReactPointerEvent, id: string) => {
    if (event.button !== 0) {
      return
    }

    event.stopPropagation()
    const c = toCanvas(event.clientX, event.clientY)
    gestureRef.current = { kind: 'connect', fromId: id }
    hoverRef.current = null
    setHoverId(null)
    setConnect({ fromId: id, cx: c.x, cy: c.y })
    canvasRef.current?.setPointerCapture(event.pointerId)
  }

  const startPan = (event: ReactPointerEvent) => {
    if (event.button !== 0) {
      return
    }

    // 连线模式下点空白处 = 取消已选的上级,避免高亮卡住。
    if (linkSource) {
      setLinkSource(null)
    }

    gestureRef.current = { kind: 'pan', sx: event.clientX, sy: event.clientY, px: panRef.current.x, py: panRef.current.y }
    canvasRef.current?.setPointerCapture(event.pointerId)
  }

  const onPointerMove = (event: ReactPointerEvent) => {
    const g = gestureRef.current

    if (!g) {
      return
    }

    if (g.kind === 'drag') {
      const c = toCanvas(event.clientX, event.clientY)
      setPositions(prev => ({ ...prev, [g.id]: { x: c.x - g.offX, y: c.y - g.offY } }))
    } else if (g.kind === 'connect') {
      const c = toCanvas(event.clientX, event.clientY)
      const target = nodeAt(c.x, c.y, g.fromId)
      hoverRef.current = target
      setHoverId(target)
      setConnect({ fromId: g.fromId, cx: c.x, cy: c.y })
    } else {
      const next = { x: g.px + (event.clientX - g.sx), y: g.py + (event.clientY - g.sy) }
      panRef.current = next
      setPan(next)
    }
  }

  const endGesture = (event: ReactPointerEvent) => {
    const g = gestureRef.current
    gestureRef.current = null

    if (!g) {
      return
    }

    canvasRef.current?.releasePointerCapture(event.pointerId)

    if (g.kind === 'drag') {
      persist()
    } else if (g.kind === 'connect') {
      const target = hoverRef.current
      hoverRef.current = null
      setHoverId(null)
      setConnect(null)

      // 跳过自连与已存在的连接,避免后端重复报错
      if (target && target !== g.fromId && !edges.some(e => e.manager_id === g.fromId && e.member_id === target)) {
        onConnect(g.fromId, target)
      }
    }
  }

  const submitCreate = async (event: FormEvent) => {
    event.preventDefault()
    setSubmitting(true)
    setFormError('')

    const result = await onCreateSubaccount({ email, name, password })

    setSubmitting(false)

    if (!result.ok) {
      setFormError(result.error || '创建失败')

      return
    }

    setName('')
    setEmail('')
    setPassword('')
    setCreating(false)
  }

  const connectFromPos = connect ? positions[connect.fromId] : null

  return (
    <div className="relative">
      {/* 工具条 */}
      <div className="absolute left-2 top-2 z-10 flex items-center gap-1.5">
        <Button
          onClick={() => setCreating(v => !v)}
          size="sm"
          type="button"
          variant={creating ? 'secondary' : 'default'}
        >
          <Codicon name="add" size="0.85rem" />
          子账号
        </Button>
        <Button
          onClick={() => {
            setLinkMode(v => !v)
            setLinkSource(null)
            setCreating(false)
          }}
          size="sm"
          type="button"
          variant={linkMode ? 'secondary' : 'ghost'}
        >
          <Codicon name="link" size="0.85rem" />
          连线
        </Button>
        <Button disabled={busy} onClick={onRefresh} size="sm" type="button" variant="secondary">
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Codicon name="refresh" size="0.85rem" />}
          刷新
        </Button>
      </div>

      {/* 创建子账号表单 */}
      {creating ? (
        <form
          className="absolute left-2 top-12 z-10 grid w-60 gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-3 shadow-lg"
          onSubmit={submitCreate}
        >
          <div className="text-xs font-semibold">新建子账号</div>
          <Input onChange={e => setName(e.target.value)} placeholder="名称(可选)" value={name} />
          <Input onChange={e => setEmail(e.target.value)} placeholder="邮箱" type="email" value={email} />
          <Input onChange={e => setPassword(e.target.value)} placeholder="初始密码" type="password" value={password} />
          {formError ? <p className="text-[0.6875rem] text-destructive">{formError}</p> : null}
          <div className="flex gap-1.5">
            <Button className="flex-1" disabled={submitting || !email || !password} size="sm" type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : null}
              创建
            </Button>
            <Button onClick={() => setCreating(false)} size="sm" type="button" variant="ghost">
              取消
            </Button>
          </div>
        </form>
      ) : null}

      {/* 连线模式提示 */}
      {linkMode ? (
        <div className="absolute left-1/2 top-2 z-10 -translate-x-1/2 rounded-md bg-(--ui-accent,#7c83ff) px-2.5 py-1 text-[0.6875rem] font-medium text-[#0b0b14] shadow">
          {linkSource ? '再点「下级」账号即连上(点上级自己可取消)' : '连线模式:先点「上级」账号,再点「下级」账号'}
        </div>
      ) : null}

      {/* 图例 */}
      <div className="absolute bottom-2 right-2 z-10 flex items-center gap-3 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-2 py-1 text-[0.625rem] text-(--ui-text-tertiary)">
        <span className="flex items-center gap-1">
          <span className="inline-block h-px w-4 bg-(--ui-accent,#7c83ff)" /> 协同连接
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-px w-4 border-t border-dashed border-(--ui-text-tertiary)" /> 额度归属
        </span>
      </div>

      {/* 画布 */}
      <div
        className="relative h-[440px] touch-none overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) [background-image:radial-gradient(circle,var(--ui-stroke-tertiary)_1px,transparent_1px)] [background-size:18px_18px]"
        onPointerDown={startPan}
        onPointerMove={onPointerMove}
        onPointerUp={endGesture}
        ref={canvasRef}
        style={{ cursor: gestureRef.current?.kind === 'pan' ? 'grabbing' : 'default' }}
      >
        <div className="absolute left-0 top-0" style={{ transform: `translate(${pan.x}px, ${pan.y}px)` }}>
          <svg
            className="pointer-events-none absolute overflow-visible"
            height={4000}
            style={{ left: -2000, top: -2000 }}
            viewBox="-2000 -2000 4000 4000"
            width={4000}
          >
            <defs>
              <marker id="tc-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="5" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" fill="var(--ui-accent,#7c83ff)" />
              </marker>
            </defs>

            {/* 额度归属:parent_id 虚线 */}
            {nodes.map(n => {
              const parent = n.parent_id && nodeById.has(n.parent_id) ? n.parent_id : null

              if (!parent || parent === n.user_id) {
                return null
              }

              const a = positions[parent]
              const b = positions[n.user_id]

              if (!a || !b) {
                return null
              }

              return (
                <path
                  d={edgePath(a, b, heights[parent] || NODE_H)}
                  fill="none"
                  key={`own-${n.user_id}`}
                  stroke="var(--ui-text-tertiary)"
                  strokeDasharray="4 4"
                  strokeOpacity="0.5"
                  strokeWidth="1.5"
                />
              )
            })}

            {/* 协同连接:edges 实线带箭头 */}
            {edges.map(e => {
              const a = positions[e.manager_id]
              const b = positions[e.member_id]

              if (!a || !b) {
                return null
              }

              return (
                <path
                  d={edgePath(a, b, heights[e.manager_id] || NODE_H)}
                  fill="none"
                  key={`edge-${e.manager_id}-${e.member_id}`}
                  markerEnd="url(#tc-arrow)"
                  stroke="var(--ui-accent,#7c83ff)"
                  strokeWidth="2"
                />
              )
            })}

            {/* 正在拖拽的连线 */}
            {connect && connectFromPos ? (
              <path
                d={`M ${connectFromPos.x + NODE_W / 2} ${connectFromPos.y + (heights[connect.fromId] || NODE_H)} L ${connect.cx} ${connect.cy}`}
                fill="none"
                stroke="var(--ui-accent,#7c83ff)"
                strokeDasharray="5 4"
                strokeWidth="2"
              />
            ) : null}
          </svg>

          {/* 节点 */}
          {nodes.map(node => {
            const p = positions[node.user_id]

            if (!p) {
              return null
            }

            const isRoot = node.user_id === rootId
            const isHover = hoverId === node.user_id
            const label = node.name || node.email
            const kbs = (resourcesByNode[node.user_id] ?? []).filter(r => r.kind === 'knowledge')

            return (
              <div className="absolute left-0 top-0 select-none" key={node.user_id} style={{ transform: `translate(${p.x}px, ${p.y}px)`, width: NODE_W }}>
                <div
                  className={cn(
                    'cursor-grab rounded-lg border bg-(--ui-bg-elevated) px-3 py-2 shadow-sm transition active:cursor-grabbing',
                    isRoot ? 'border-(--ui-accent,#7c83ff)' : 'border-(--ui-stroke-tertiary)',
                    (isHover || linkSource === node.user_id) && 'ring-2 ring-(--ui-accent,#7c83ff)',
                    linkMode && 'cursor-pointer active:cursor-pointer'
                  )}
                  onPointerDown={event => startDrag(event, node.user_id)}
                  ref={el => {
                    cardRefs.current[node.user_id] = el
                  }}
                >
                  <div className="flex items-center justify-between gap-1.5">
                    <span className="truncate text-[0.8125rem] font-semibold">{label}</span>
                    <span
                      className={cn(
                        'shrink-0 rounded-full px-1.5 py-0.5 text-[0.625rem] font-medium',
                        isRoot ? 'bg-(--ui-accent,#7c83ff) text-[#0b0b14]' : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'
                      )}
                    >
                      {isRoot ? '主账号' : '子账号'}
                    </span>
                  </div>
                  {node.name ? <div className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">{node.email}</div> : null}
                  {!isRoot ? (
                    <select
                      className="desktop-input-chrome mt-1.5 w-full rounded-[3px] border px-1.5 py-0.5 text-[0.6875rem] leading-4 text-foreground outline-none"
                      onChange={event => onSetRole(node.user_id, event.target.value)}
                      onPointerDown={event => event.stopPropagation()}
                      value={node.role || ''}
                    >
                      <option value="">（未分配角色）</option>
                      {roles.map(r => (
                        <option key={r} value={r}>
                          {r}
                        </option>
                      ))}
                      {node.role && !roles.includes(node.role) ? (
                        <option value={node.role}>{node.role}(已删)</option>
                      ) : null}
                    </select>
                  ) : null}

                  {/* 知识库:主拉取存档的下级资源(Phase 1d)——看「每个子有哪些知识库」 */}
                  {kbs.length ? (
                    <div className="mt-1.5 border-t border-(--ui-stroke-tertiary) pt-1.5">
                      <div className="flex items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
                        <Codicon name="library" size="0.7rem" />
                        知识库 {kbs.length}
                      </div>
                      <div className="mt-1 grid gap-0.5">
                        {kbs.slice(0, MAX_KB_SHOWN).map(kb => (
                          <div
                            className="truncate rounded bg-(--ui-bg-tertiary) px-1.5 py-0.5 text-[0.6875rem] text-(--ui-text-secondary)"
                            key={kb.resource_id}
                            title={kbHint(kb)}
                          >
                            {kb.name}
                          </div>
                        ))}
                        {kbs.length > MAX_KB_SHOWN ? (
                          <div className="text-[0.625rem] text-(--ui-text-tertiary)">+{kbs.length - MAX_KB_SHOWN} 更多</div>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </div>

                {/* 连线手柄:拖出一个下级 */}
                <button
                  aria-label="拖出连线"
                  className="absolute -bottom-2 left-1/2 size-4 -translate-x-1/2 cursor-crosshair rounded-full border-2 border-(--ui-bg-elevated) bg-(--ui-accent,#7c83ff) transition hover:scale-125"
                  onPointerDown={event => startConnect(event, node.user_id)}
                  title="拖我连一个下级"
                  type="button"
                />
              </div>
            )
          })}
        </div>

        {nodes.length === 0 ? (
          <div className="grid h-full place-items-center text-xs text-(--ui-text-tertiary)">暂无账号</div>
        ) : null}
      </div>
    </div>
  )
}
