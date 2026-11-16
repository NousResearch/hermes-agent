import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, setToken, type AdminUser, type Perm } from '../api'

const KEY = 'pdr-admin-auth'

function read(): { token: string; user: AdminUser | null } {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return JSON.parse(raw)
  } catch {
    // 存储不可用时视为未登录
  }
  return { token: '', user: null }
}

function write(value: { token: string; user: AdminUser | null }): void {
  try {
    if (value.token) localStorage.setItem(KEY, JSON.stringify(value))
    else localStorage.removeItem(KEY)
  } catch {
    // 只影响刷新后是否保持登录
  }
}

export const useAuth = defineStore('auth', () => {
  const saved = read()
  const token = ref(saved.token)
  const user = ref<AdminUser | null>(saved.user)
  setToken(token.value)

  const loggedIn = computed(() => !!token.value && !!user.value)
  const isSuper = computed(() => user.value?.perms === '*')

  /** 页面与按钮的权限判断：平台超级管理员拥有全部权限 */
  function can(perm: Perm | null | undefined): boolean {
    if (!perm) return true
    const perms = user.value?.perms
    return perms === '*' || !!perms?.includes(perm)
  }

  async function login(account: string, password: string): Promise<void> {
    const res = await api.login(account, password)
    token.value = res.token
    user.value = res.user
    setToken(res.token)
    write({ token: res.token, user: res.user })
  }

  async function logout(): Promise<void> {
    try {
      await api.logout()
    } finally {
      token.value = ''
      user.value = null
      setToken('')
      write({ token: '', user: null })
    }
  }

  return { token, user, loggedIn, isSuper, can, login, logout }
})
