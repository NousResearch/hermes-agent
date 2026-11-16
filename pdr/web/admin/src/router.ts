import { createRouter, createWebHistory } from 'vue-router'
import type { Perm } from './api'
import { MODULES } from './lib/nav'
import { useAuth } from './stores/auth'

/** 管理端路由；这里没有、也不应出现任何用户端路由或入口 */
const VIEWS: Record<string, () => Promise<unknown>> = {
  '/': () => import('./views/DashboardView.vue'),
  '/content/tasks': () => import('./views/TaskListView.vue'),
  '/content/board': () => import('./views/ContentBoardView.vue'),
  '/review/doc': () => import('./views/DocReviewView.vue'),
  '/review/media': () => import('./views/MediaReviewView.vue'),
  '/review/label': () => import('./views/LabelView.vue'),
  '/review/classify': () => import('./views/ClassifyView.vue'),
  '/knowledge': () => import('./views/KnowledgeView.vue'),
  '/agent/config': () => import('./views/AgentConfigView.vue'),
  '/agent/skills': () => import('./views/SkillsView.vue'),
  '/agent/memory': () => import('./views/MemoryView.vue'),
  '/agent/cron': () => import('./views/CronView.vue'),
  '/agent/sessions': () => import('./views/SessionsView.vue'),
  '/sys/roles': () => import('./views/RolesView.vue'),
  '/sys/users': () => import('./views/UsersView.vue'),
  '/sys/tenants': () => import('./views/TenantsView.vue'),
  '/sys/config': () => import('./views/SystemConfigView.vue'),
  '/sys/audit': () => import('./views/AuditView.vue')
}

const pages = MODULES.flatMap((m) => m.pages.map((p) => ({
  path: p.path,
  component: VIEWS[p.path],
  meta: { auth: true, perm: p.perm, title: p.name }
})))

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: () => import('./views/LoginView.vue'), meta: { title: '登录' } },
    ...pages,
    { path: '/403', component: () => import('./views/ForbiddenView.vue'), meta: { auth: true, title: '没有权限' } },
    { path: '/:pathMatch(.*)*', redirect: '/' }
  ]
})

router.beforeEach((to) => {
  const auth = useAuth()
  if (to.meta.auth && !auth.loggedIn) return { path: '/login', query: to.fullPath === '/' ? {} : { redirect: to.fullPath } }
  if (to.path === '/login' && auth.loggedIn) return { path: '/' }
  // 没有权限的页面不显示入口；直接输入地址也只会看到无权限提示
  if (to.meta.perm && !auth.can(to.meta.perm as Perm)) return { path: '/403', query: { from: to.path } }
  return true
})

router.afterEach((to) => {
  document.title = [to.meta.title as string, 'PDR 管理端'].filter(Boolean).join(' · ')
})
