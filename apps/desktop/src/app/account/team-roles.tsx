import { type FormEvent, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import type { DesktopAccountTreeNode } from '@/global'
import { cn } from '@/lib/utils'

import { KIND_LABEL, resourceKey, type TeamGrant, type TeamResource, type TeamUserGrant } from './team-types'

// 角色管理 + 权限管理:从团队画布的小弹窗里**拿出来**,做成团队账号页的常驻面板。
//   - 角色管理(RolesPanel):定义组织角色集(加/删);把角色**分配**给具体子账号在画布卡片上做。
//   - 权限管理(PermissionsPanel):点一个角色 → 勾选它可用的资源(知识库/工作流/智能体),写主本地 grant_policy。

const PANEL = 'grid content-start gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-3'

export function RolesPanel({
  roles,
  onAddRole,
  onRemoveRole
}: {
  roles: string[]
  onAddRole: (name: string) => void
  onRemoveRole: (name: string) => void
}) {
  const [newRole, setNewRole] = useState('')

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const next = newRole.trim()

    if (next) {
      onAddRole(next)
      setNewRole('')
    }
  }

  return (
    <div className={PANEL}>
      <div className="text-xs font-semibold">
        角色管理 <span className="font-normal text-(--ui-text-tertiary)">· 主账号定义,在子账号卡片上分配</span>
      </div>
      {roles.length ? (
        <div className="flex flex-wrap gap-1.5">
          {roles.map(r => (
            <span
              className="flex items-center gap-1 rounded-full bg-(--ui-bg-tertiary) px-2 py-0.5 text-[0.6875rem] text-(--ui-text-secondary)"
              key={r}
            >
              <span className="max-w-[9rem] truncate">{r}</span>
              <button
                aria-label={`删除角色 ${r}`}
                className="text-(--ui-text-tertiary) hover:text-destructive"
                onClick={() => onRemoveRole(r)}
                type="button"
              >
                <Codicon name="close" size="0.7rem" />
              </button>
            </span>
          ))}
        </div>
      ) : (
        <p className="text-[0.6875rem] text-(--ui-text-tertiary)">还没有角色,先加一个(如 财务 / 客服 / 研发)。</p>
      )}
      <form className="flex gap-1.5" onSubmit={submit}>
        <Input onChange={event => setNewRole(event.target.value)} placeholder="新角色名" value={newRole} />
        <Button disabled={!newRole.trim()} size="sm" type="submit">
          添加
        </Button>
      </form>
    </div>
  )
}

export function PermissionsPanel({
  roles,
  grants,
  userGrants,
  nodes,
  resourcesByNode,
  langflowCapable = true,
  onGrant,
  onRevoke,
  onGrantUser,
  onRevokeUser
}: {
  roles: string[]
  grants: TeamGrant[]
  userGrants: TeamUserGrant[]
  nodes: DesktopAccountTreeNode[]
  resourcesByNode: Record<string, TeamResource[]>
  // 本账号是否「能力承载节点」(有 langflow)。否则不显示配 MCP —— 它没工作流可发。
  langflowCapable?: boolean
  onGrant: (grant: TeamGrant) => void
  onRevoke: (grant: TeamGrant) => void
  onGrantUser: (grant: TeamUserGrant) => void
  onRevokeUser: (grant: TeamUserGrant) => void
}) {
  // 按角色(批量)/ 按账号(个别微调)两种主体;各记各的选中,切模式不串。
  const [mode, setMode] = useState<'role' | 'account'>('role')
  const [pickedRole, setPickedRole] = useState<string | null>(null)
  const [pickedAccount, setPickedAccount] = useState<string | null>(null)

  // 下级账号(有 parent_id)= 「按账号微调」的可选目标。
  const subAccounts = useMemo(() => nodes.filter(n => n.parent_id), [nodes])

  // 当前模式的主体列表 + 选中项(派生:主体没了就当未选,不留悬空)。
  const subjects = useMemo(
    () =>
      mode === 'role'
        ? roles.map(r => ({ id: r, label: r }))
        : subAccounts.map(n => ({ id: n.user_id, label: n.name || n.email })),
    [mode, roles, subAccounts]
  )

  const active = useMemo(() => {
    const ids = new Set(subjects.map(s => s.id))
    const picked = mode === 'role' ? pickedRole : pickedAccount

    return picked && ids.has(picked) ? picked : null
  }, [subjects, mode, pickedRole, pickedAccount])

  // 现存可授权工作流(MCP)key —— 只 kind=workflow(智能体默认开、知识库走 query flow,均不在此授权);
  // 把"失效授权"从徽标计数里排除,对齐后端 list_authorized_resources。
  const existingResourceKeys = useMemo(() => {
    const set = new Set<string>()

    for (const list of Object.values(resourcesByNode)) {
      for (const r of list) {
        if (r.kind === 'workflow') {
          set.add(resourceKey(r))
        }
      }
    }

    return set
  }, [resourcesByNode])

  // 选中主体已授权的资源 key(回显勾选)——角色看 grants、账号看 userGrants。
  const grantedKeys = useMemo(() => {
    const set = new Set<string>()

    if (mode === 'role') {
      for (const g of grants) {
        if (g.role === active) {
          set.add(resourceKey(g))
        }
      }
    } else {
      for (const g of userGrants) {
        if (g.user_id === active) {
          set.add(resourceKey(g))
        }
      }
    }

    return set
  }, [mode, grants, userGrants, active])

  // 每个主体「实际可用」资源数(grant ∩ 现存)→ chip 徽标。
  const countBySubject = useMemo(() => {
    const counts = new Map<string, number>()

    const src =
      mode === 'role'
        ? grants.map(g => ({ id: g.role, key: resourceKey(g) }))
        : userGrants.map(g => ({ id: g.user_id, key: resourceKey(g) }))

    for (const { id, key } of src) {
      if (existingResourceKeys.has(key)) {
        counts.set(id, (counts.get(id) ?? 0) + 1)
      }
    }

    return counts
  }, [mode, grants, userGrants, existingResourceKeys])

  // 可授权的工作流(MCP):按节点(自己 + 下级)分组,只取 kind=workflow,带节点展示名。
  const grantableNodes = useMemo(
    () =>
      nodes
        .map(n => ({ node: n, resources: (resourcesByNode[n.user_id] ?? []).filter(r => r.kind === 'workflow') }))
        .filter(entry => entry.resources.length > 0),
    [nodes, resourcesByNode]
  )

  const pick = (id: string) =>
    mode === 'role' ? setPickedRole(c => (c === id ? null : id)) : setPickedAccount(c => (c === id ? null : id))

  // 勾/取消:角色模式写 grant_policy,账号模式写 grant_user。
  const toggle = (resource: TeamResource, granted: boolean) => {
    if (!active) {
      return
    }

    const base = { node_uid: resource.node_uid, kind: resource.kind, resource_id: resource.resource_id }

    if (mode === 'role') {
      ;(granted ? onRevoke : onGrant)({ role: active, ...base })
    } else {
      ;(granted ? onRevokeUser : onGrantUser)({ user_id: active, ...base })
    }
  }

  const activeLabel = subjects.find(s => s.id === active)?.label ?? active

  return (
    <div className={PANEL}>
      <div className="text-xs font-semibold">
        权限管理 <span className="font-normal text-(--ui-text-tertiary)">· 给角色或具体账号配可用的工作流(MCP)</span>
      </div>
      {!langflowCapable ? (
        <p className="text-[0.6875rem] leading-5 text-(--ui-text-tertiary)">
          此账号没有工作流(langflow)能力,不能配 MCP 给下级。请在带工作流的「能力承载节点」上配置。
        </p>
      ) : (
        <>
          {/* 模式:按角色(批量)/ 按账号(个别微调) */}
          <div className="flex gap-1 text-[0.6875rem]">
            {(['role', 'account'] as const).map(m => (
              <button
                className={cn(
                  'rounded px-2 py-0.5',
                  mode === m ? 'bg-(--ui-accent,#7c83ff) text-[#0b0b14]' : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'
                )}
                key={m}
                onClick={() => setMode(m)}
                type="button"
              >
                {m === 'role' ? '按角色' : '按账号微调'}
              </button>
            ))}
          </div>

          {subjects.length ? (
            <div className="flex flex-wrap gap-1.5">
              {subjects.map(s => (
                <button
                  className={cn(
                    'flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.6875rem]',
                    active === s.id ? 'bg-(--ui-accent,#7c83ff) text-[#0b0b14]' : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'
                  )}
                  key={s.id}
                  onClick={() => pick(s.id)}
                  title={`配置「${s.label}」可用的工作流`}
                  type="button"
                >
                  <span className="max-w-[9rem] truncate">{s.label}</span>
                  {countBySubject.get(s.id) ? (
                    <span
                      className={cn(
                        'shrink-0 rounded-full px-1 text-[0.5625rem] tabular-nums',
                        active === s.id ? 'bg-[#0b0b14]/15' : 'bg-(--ui-bg-elevated) text-(--ui-text-tertiary)'
                      )}
                    >
                      {countBySubject.get(s.id)}
                    </span>
                  ) : null}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-[0.6875rem] text-(--ui-text-tertiary)">
              {mode === 'role' ? '先在「角色管理」里加角色。' : '还没有下级账号 —— 去「团队账号」创建子账号。'}
            </p>
          )}

          {active ? (
            <div className="grid gap-1.5 border-t border-(--ui-stroke-tertiary) pt-2">
              <div className="text-[0.6875rem] text-(--ui-text-secondary)">
                「<span className="font-medium">{activeLabel}</span>」可用的工作流(MCP)
              </div>
              {grantableNodes.length ? (
                <div className="grid max-h-56 gap-2 overflow-y-auto pr-1">
                  {grantableNodes.map(({ node, resources }) => (
                    <div className="grid gap-1" key={node.user_id}>
                      <div className="truncate text-[0.625rem] font-medium text-(--ui-text-tertiary)">
                        {node.name || node.email}
                      </div>
                      {resources.map(resource => {
                        const granted = grantedKeys.has(resourceKey(resource))

                        return (
                          <label
                            className="flex cursor-pointer items-center gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)"
                            key={resourceKey(resource)}
                          >
                            <input
                              checked={granted}
                              className="accent-(--ui-accent,#7c83ff)"
                              onChange={() => toggle(resource, granted)}
                              type="checkbox"
                            />
                            <span className="truncate" title={resource.name}>
                              {resource.name}
                            </span>
                            <span className="ml-auto shrink-0 rounded bg-(--ui-bg-tertiary) px-1 text-[0.5625rem] text-(--ui-text-tertiary)">
                              {KIND_LABEL[resource.kind] ?? resource.kind}
                            </span>
                          </label>
                        )
                      })}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-[0.625rem] leading-5 text-(--ui-text-tertiary)">
                  还没有可授权的工作流(MCP) —— 去「工作流」建一个 ChatInput + ChatOutput 对话流,它会出现在这里供你授权。
                </p>
              )}
            </div>
          ) : subjects.length ? (
            <p className="text-[0.625rem] text-(--ui-text-tertiary)">
              点上面的{mode === 'role' ? '角色' : '账号'} → 勾选它可用的工作流(MCP)。
            </p>
          ) : null}
        </>
      )}
    </div>
  )
}
