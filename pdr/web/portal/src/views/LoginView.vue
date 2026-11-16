<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api'
import { useAuth } from '../stores/auth'
import GlowOrb from '../components/GlowOrb.vue'

const auth = useAuth()
const router = useRouter()
const route = useRoute()
const mock = api.mode === 'mock'
const account = ref(mock ? 'chenxiao' : '')
const password = ref(mock ? 'demo' : '')
const error = ref('')
const loading = ref(false)

async function submit(): Promise<void> {
  error.value = ''
  loading.value = true
  try {
    await auth.login(account.value, password.value)
    const to = typeof route.query.redirect === 'string' && route.query.redirect.startsWith('/') ? route.query.redirect : '/'
    router.replace(to)
  } catch (e) {
    error.value = (e as Error).message || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="wrap">
    <div class="bg" aria-hidden="true" />
    <form class="card" @submit.prevent="submit">
      <GlowOrb :size="56" />
      <h1>登录智能问答</h1>
      <p class="sub">使用公司分配的账号登录</p>
      <label>
        <span>账号</span>
        <input v-model="account" autocomplete="username" required />
      </label>
      <label>
        <span>密码</span>
        <input v-model="password" type="password" autocomplete="current-password" required />
      </label>
      <p v-if="error" class="err" role="alert">{{ error }}</p>
      <button class="submit" type="submit" :disabled="loading">{{ loading ? '登录中…' : '登录' }}</button>
      <p v-if="mock" class="hint">演示模式：任意账号和密码都可以登录，数据为示例</p>
    </form>
  </div>
</template>

<style scoped>
.wrap { position: relative; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; background: var(--frame); overflow: hidden; }
.bg {
  position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(600px 300px at 50% 70%, rgba(150, 140, 255, 0.16), transparent 70%), radial-gradient(400px 240px at 70% 30%, rgba(140, 230, 200, 0.12), transparent 70%);
}
.card {
  position: relative; width: 100%; max-width: 380px; display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 32px 28px;
  background: rgba(255, 255, 255, 0.92); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow-card);
}
h1 { margin: 8px 0 0; font-size: 22px; }
.sub { margin: 0 0 8px; font-size: 13px; color: var(--muted); }
label { width: 100%; display: flex; flex-direction: column; gap: 6px; font-size: 13px; color: var(--text-2); }
input { height: 42px; padding: 0 12px; border: 1px solid var(--line); border-radius: 11px; background: #fff; outline: none; }
input:focus { border-color: #c9c8f7; box-shadow: 0 0 0 3px #efeffd; }
.err { margin: 0; width: 100%; font-size: 13px; color: var(--danger); }
.submit {
  width: 100%; height: 44px; margin-top: 6px; border: 0; border-radius: 12px; color: #fff; font-size: 15px; font-weight: 500;
  background: linear-gradient(160deg, #8a86ff, #5b5bea); box-shadow: 0 8px 18px rgba(91, 91, 234, 0.3);
}
.submit:disabled { opacity: 0.6; }
.hint { margin: 4px 0 0; font-size: 12px; color: #8a5a00; text-align: center; }
</style>
