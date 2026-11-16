/**
 * 管理端菜单：模块 → 页面。每个页面挂一个功能权限，菜单、路由守卫与工作台都按它过滤；
 * 登录者没有权限的页面不出现入口（直接访问显示无权限，后端接口返回 403）。
 */
import type { Perm } from '../api'
import type { IconName } from './icons'

export type Tint = 'purple' | 'blue' | 'green' | 'amber' | 'pink'

export interface NavPage { name: string; path: string; perm: Perm | null }
export interface NavModule { key: string; name: string; icon: IconName; tint: Tint; pages: NavPage[] }

export const MODULES: NavModule[] = [
  { key: 'home', name: '工作台', icon: 'home', tint: 'purple', pages: [{ name: '工作台', path: '/', perm: null }] },
  { key: 'content', name: '内容任务', icon: 'tasks', tint: 'blue', pages: [
    { name: '任务列表', path: '/content/tasks', perm: 'content:asset' },
    { name: '内容看板', path: '/content/board', perm: 'content:dashboard' }
  ] },
  { key: 'review', name: '人工核对', icon: 'review', tint: 'green', pages: [
    { name: '文档核对', path: '/review/doc', perm: 'content:review' },
    { name: '音视频核对', path: '/review/media', perm: 'content:review' },
    { name: '标注工作台', path: '/review/label', perm: 'content:label' },
    { name: '分类确认', path: '/review/classify', perm: 'content:classify' }
  ] },
  { key: 'knowledge', name: '知识库', icon: 'book', tint: 'amber', pages: [{ name: '知识库', path: '/knowledge', perm: 'knowledge' }] },
  { key: 'agent', name: '智能体', icon: 'bot', tint: 'purple', pages: [
    { name: '智能体配置', path: '/agent/config', perm: 'agent:config' },
    { name: '技能 · 工具集', path: '/agent/skills', perm: 'agent:skill' },
    { name: '记忆 · 人格', path: '/agent/memory', perm: 'agent:memory' },
    { name: '定时任务 · 消息网关', path: '/agent/cron', perm: 'agent:cron' },
    { name: '问答会话审计', path: '/agent/sessions', perm: 'agent:audit' }
  ] },
  { key: 'sys', name: '权限管理', icon: 'shield', tint: 'pink', pages: [
    { name: '角色与权限', path: '/sys/roles', perm: 'sys:role' },
    { name: '用户 · 区域', path: '/sys/users', perm: 'sys:user' },
    { name: '租户管理', path: '/sys/tenants', perm: 'sys:tenant' }
  ] },
  { key: 'settings', name: '系统设置', icon: 'gear', tint: 'blue', pages: [
    { name: '系统配置', path: '/sys/config', perm: 'sys:config' },
    { name: '审计日志', path: '/sys/audit', perm: 'sys:audit' }
  ] }
]

export const TINTS: Record<Tint, [string, string]> = {
  purple: ['#EFECFE', '#5443D6'],
  blue: ['#E6F0FF', '#2557C7'],
  green: ['#E3F6EA', '#1E7A43'],
  amber: ['#FFF3D6', '#8A5A00'],
  pink: ['#FDE7EB', '#B42345']
}

/** 按权限裁剪后的菜单：没有任何可见页面的模块整个隐藏 */
export function visibleModules(can: (perm: Perm) => boolean): NavModule[] {
  return MODULES.map((m) => ({ ...m, pages: m.pages.filter((p) => !p.perm || can(p.perm)) })).filter((m) => m.pages.length)
}

export function moduleOf(path: string): NavModule | undefined {
  return MODULES.find((m) => m.pages.some((p) => (p.path === '/' ? path === '/' : path.startsWith(p.path))))
}
