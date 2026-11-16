import { createRouter, createWebHistory } from 'vue-router'
import { useAuth } from './stores/auth'

/** 用户端只有问答相关页面；这里没有、也不应出现任何管理端路由或入口 */
export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: () => import('./views/LoginView.vue'), meta: { title: '登录' } },
    { path: '/', component: () => import('./views/ChatView.vue'), meta: { auth: true, title: '智能问答' } },
    { path: '/c/:id', component: () => import('./views/ChatView.vue'), meta: { auth: true, title: '智能问答' } },
    { path: '/explore', component: () => import('./views/ExploreView.vue'), meta: { auth: true, title: '发现' } },
    { path: '/saved', component: () => import('./views/SavedView.vue'), meta: { auth: true, title: '收藏' } },
    { path: '/history', component: () => import('./views/HistoryView.vue'), meta: { auth: true, title: '历史' } },
    { path: '/:pathMatch(.*)*', redirect: '/' }
  ]
})

router.beforeEach((to) => {
  const auth = useAuth()
  if (to.meta.auth && !auth.loggedIn) return { path: '/login', query: { redirect: to.fullPath } }
  if (to.path === '/login' && auth.loggedIn) return { path: '/' }
  return true
})

router.afterEach((to) => {
  const tenant = useAuth().user?.tenantName
  document.title = [to.meta.title as string, tenant].filter(Boolean).join(' · ')
})
