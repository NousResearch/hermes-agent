// 团队资源 / 授权的共享类型 + 纯函数。
// 从 team-canvas.tsx 抽出来单独成模块,让组件文件只导出组件(否则导出 KIND_LABEL/resourceKey 这类
// 运行时值会破坏 React Fast Refresh:`hmr invalidate … export is incompatible`)。team-canvas / team-roles /
// index 都从这里取,单一来源。

// 主本地资源注册表里的一条(子上报、主拉取存档;Phase 1d)。kind = knowledge | workflow | agent。
export interface TeamResource {
  node_uid: string
  kind: string
  resource_id: string
  name: string
  meta?: Record<string, unknown>
  updated_ts?: number
}

// 一条授权策略(角色 → 某节点上的某资源;Phase 2a)。
export interface TeamGrant {
  role: string
  node_uid: string
  kind: string
  resource_id: string
}

// (node_uid, kind, resource_id) → 资源在某角色授权集合里的稳定 key。
export function resourceKey(r: { node_uid: string; kind: string; resource_id: string }): string {
  return `${r.node_uid}|${r.kind}|${r.resource_id}`
}

// 资源类型中文标签(授权列表里区分知识库 / 工作流 / 智能体)。
export const KIND_LABEL: Record<string, string> = { knowledge: '知识库', workflow: '工作流', agent: '智能体' }
