import { type FormEvent, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import type { DesktopAccountTreeNode } from '@/global'
import { cn } from '@/lib/utils'

import { KIND_LABEL, resourceKey, type TeamGrant, type TeamResource } from './team-types'

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
  nodes,
  resourcesByNode,
  langflowCapable = true,
  onGrant,
  onRevoke
}: {
  roles: string[]
  grants: TeamGrant[]
  nodes: DesktopAccountTreeNode[]
  resourcesByNode: Record<string, TeamResource[]>
  // 本账号是否「能力承载节点」(有 langflow)。否则不显示配 MCP —— 它没工作流可发。
  langflowCapable?: boolean
  onGrant: (grant: TeamGrant) => void
  onRevoke: (grant: TeamGrant) => void
}) {
  const [picked, setPicked] = useState<string | null>(null)
  // 选中的角色被删了 → 当作未选(派生,不留悬空选中)。
  const activeRole = picked && roles.includes(picked) ? picked : null

  // 选中角色已授权的资源 key(O(1) 回显勾选)。
  const grantedKeys = useMemo(() => {
    const set = new Set<string>()

    for (const g of grants) {
      if (g.role === activeRole) {
        set.add(resourceKey(g))
      }
    }

    return set
  }, [grants, activeRole])

  // 现存的**可授权工作流(MCP)** key —— 只 kind=workflow(智能体默认开、知识库走 query flow,均不在此授权)。
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

  // 每个角色「实际可用」资源数(grant ∩ 现存)→ 角色 chip 徽标。
  const authorizedCountByRole = useMemo(() => {
    const counts = new Map<string, number>()

    for (const g of grants) {
      if (existingResourceKeys.has(resourceKey(g))) {
        counts.set(g.role, (counts.get(g.role) ?? 0) + 1)
      }
    }

    return counts
  }, [grants, existingResourceKeys])

  // 可授权的工作流(MCP):按节点(自己 + 下级)分组,只取 kind=workflow,带节点展示名。
  const grantableNodes = useMemo(
    () =>
      nodes
        .map(n => ({ node: n, resources: (resourcesByNode[n.user_id] ?? []).filter(r => r.kind === 'workflow') }))
        .filter(entry => entry.resources.length > 0),
    [nodes, resourcesByNode]
  )

  return (
    <div className={PANEL}>
      <div className="text-xs font-semibold">
        权限管理 <span className="font-normal text-(--ui-text-tertiary)">· 点角色 → 勾选它可用的工作流(MCP)</span>
      </div>
      {!langflowCapable ? (
        <p className="text-[0.6875rem] leading-5 text-(--ui-text-tertiary)">
          此账号没有工作流(langflow)能力,不能配 MCP 给下级。请在带工作流的「能力承载节点」上配置。
        </p>
      ) : (
        <>
          {roles.length ? (
        <div className="flex flex-wrap gap-1.5">
          {roles.map(r => (
            <button
              className={cn(
                'flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.6875rem]',
                activeRole === r
                  ? 'bg-(--ui-accent,#7c83ff) text-[#0b0b14]'
                  : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'
              )}
              key={r}
              onClick={() => setPicked(cur => (cur === r ? null : r))}
              title={`配置「${r}」可用的工作流`}
              type="button"
            >
              <span className="max-w-[9rem] truncate">{r}</span>
              {authorizedCountByRole.get(r) ? (
                <span
                  className={cn(
                    'shrink-0 rounded-full px-1 text-[0.5625rem] tabular-nums',
                    activeRole === r ? 'bg-[#0b0b14]/15' : 'bg-(--ui-bg-elevated) text-(--ui-text-tertiary)'
                  )}
                >
                  {authorizedCountByRole.get(r)}
                </span>
              ) : null}
            </button>
          ))}
        </div>
      ) : (
        <p className="text-[0.6875rem] text-(--ui-text-tertiary)">先在「角色管理」里加角色。</p>
      )}

      {activeRole ? (
        <div className="grid gap-1.5 border-t border-(--ui-stroke-tertiary) pt-2">
          <div className="text-[0.6875rem] text-(--ui-text-secondary)">
            「<span className="font-medium">{activeRole}</span>」可用的工作流(MCP)
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

                    const target: TeamGrant = {
                      role: activeRole,
                      node_uid: resource.node_uid,
                      kind: resource.kind,
                      resource_id: resource.resource_id
                    }

                    return (
                      <label
                        className="flex cursor-pointer items-center gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)"
                        key={resourceKey(resource)}
                      >
                        <input
                          checked={granted}
                          className="accent-(--ui-accent,#7c83ff)"
                          onChange={() => (granted ? onRevoke : onGrant)(target)}
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
      ) : roles.length ? (
        <p className="text-[0.625rem] text-(--ui-text-tertiary)">点上面的角色 → 勾选它可用的工作流(MCP)。</p>
      ) : null}
        </>
      )}
    </div>
  )
}
