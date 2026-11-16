import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, setHttpToken, type PortalUser } from '../api'

const KEY = 'pdr-portal-auth'

function read(): { token: string; user: PortalUser | null } {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) return JSON.parse(raw)
  } catch {
    // 存储不可用时视为未登录
  }
  return { token: '', user: null }
}

function write(value: { token: string; user: PortalUser | null }): void {
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
  const user = ref<PortalUser | null>(saved.user)
  setHttpToken(token.value)

  const loggedIn = computed(() => !!token.value && !!user.value)

  async function login(account: string, password: string): Promise<void> {
    const res = await api.login(account, password)
    token.value = res.token
    user.value = res.user
    setHttpToken(res.token)
    write({ token: res.token, user: res.user })
  }

  async function logout(): Promise<void> {
    try {
      await api.logout()
    } finally {
      token.value = ''
      user.value = null
      setHttpToken('')
      write({ token: '', user: null })
    }
  }

  return { token, user, loggedIn, login, logout }
})
