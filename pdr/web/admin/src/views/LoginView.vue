<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, type DemoAccount } from '../api'
import AppIcon from '../components/AppIcon.vue'
import { errorText } from '../lib/toast'
import { useAuth } from '../stores/auth'

const auth = useAuth()
const router = useRouter()
const route = useRoute()

const account = ref('zhanglei')
const password = ref('demo')
const error = ref('')
const busy = ref(false)
const demos = ref<DemoAccount[]>([])

onMounted(async () => {
  demos.value = await api.demoAccounts()
})

async function submit(): Promise<void> {
  error.value = ''
  busy.value = true
  try {
    await auth.login(account.value, password.value)
    const target = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
    router.replace(target)
  } catch (e) {
    error.value = errorText(e)
  } finally {
    busy.value = false
  }
}

function pick(d: DemoAccount): void {
  account.value = d.account
  password.value = 'demo'
}
</script>

<template>
  <div class="login">
    <section class="panel card">
      <div class="brand">
        <span class="logo"><AppIcon name="check" :size="22" :stroke="2.4" /></span>
        <div>
          <h1>PDR 管理端</h1>
          <p>内容处理、知识库、智能体与权限管理</p>
        </div>
      </div>

      <form class="form" @submit.prevent="submit">
        <label>
          <span class="lbl">账号</span>
          <input v-model="account" class="fld" autocomplete="username" required />
        </label>
        <label>
          <span class="lbl">密码</span>
          <input v-model="password" class="fld" type="password" autocomplete="current-password" required />
        </label>
        <p v-if="error" class="notice bad" role="alert">{{ error }}</p>
        <button class="btn-p" type="submit" :disabled="busy">{{ busy ? '登录中…' : '登录' }}</button>
        <p class="hint">管理端只接受管理账号；用户端（问答）账号不能在这里登录。</p>
      </form>
    </section>

    <section v-if="demos.length" class="demos">
      <h2>演示账号</h2>
      <p>示例数据模式下任意密码均可登录。不同角色看到的菜单、数据与按钮不同。</p>
      <button
        v-for="d in demos"
        :key="d.account"
        type="button"
        class="demo card"
        :aria-pressed="account === d.account"
        @click="pick(d)"
      >
        <span class="avatar">{{ d.name.slice(0, 1) }}</span>
        <span class="grow">
          <b>{{ d.name }}</b><span class="pill purple">{{ d.roleName }}</span>
          <small>{{ d.account }} · {{ d.note }}</small>
        </span>
      </button>
    </section>
  </div>
</template>

<style scoped>
.login { min-height: 100vh; display: flex; align-items: center; justify-content: center; gap: 28px; padding: 32px 16px; flex-wrap: wrap; }
.panel { width: min(420px, 100%); padding: 32px 30px; border-radius: 24px; box-shadow: var(--shadow-lg); }
.brand { display: flex; align-items: center; gap: 14px; margin-bottom: 26px; }
.logo {
  width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; color: #fff; flex: none;
  background: linear-gradient(135deg, #8e80fa, #5443d6); box-shadow: 0 8px 18px rgba(84, 67, 214, 0.35);
}
h1 { font-size: 22px; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
.form { display: flex; flex-direction: column; gap: 14px; }
.form label { display: flex; flex-direction: column; gap: 6px; }
.form .fld { height: 42px; font-size: 14px; }
.form .btn-p { justify-content: center; height: 44px; margin-top: 4px; }
.hint { margin: 0; color: var(--faint); font-size: 12.5px; text-align: center; }
.demos { width: min(380px, 100%); display: flex; flex-direction: column; gap: 10px; }
.demos h2 { font-size: 16px; }
.demos > p { margin: 0 0 4px; color: var(--muted); font-size: 13px; }
.demo { display: flex; align-items: center; gap: 12px; padding: 12px 14px; text-align: left; border-radius: 14px; }
.demo:hover { border-color: #cfcbe6; }
.demo[aria-pressed='true'] { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(84, 67, 214, 0.12), var(--shadow); }
.demo .avatar { width: 36px; height: 36px; font-size: 14px; }
.demo b { margin-right: 8px; }
.demo small { display: block; margin-top: 2px; color: var(--muted); font-size: 12px; }
</style>
