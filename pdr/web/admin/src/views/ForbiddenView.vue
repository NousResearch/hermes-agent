<script setup lang="ts">
import AppIcon from '../components/AppIcon.vue'
import { useAuth } from '../stores/auth'

const auth = useAuth()
</script>

<template>
  <section class="card forbidden">
    <span class="ico"><AppIcon name="lock" :size="26" /></span>
    <h1>没有权限访问此页面</h1>
    <p>当前角色「{{ auth.user?.roleName }}」没有这个页面的功能权限，菜单中也不会出现它的入口。如需使用，请联系租户管理员分配权限。</p>
    <RouterLink to="/" class="btn-p sm">回到工作台</RouterLink>
  </section>
</template>

<style scoped>
.forbidden { max-width: 560px; margin: 60px auto; padding: 40px 32px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 12px; }
.ico { width: 60px; height: 60px; border-radius: 18px; display: flex; align-items: center; justify-content: center; background: var(--bad-soft); color: var(--bad); }
h1 { font-size: 20px; }
p { margin: 0 0 8px; color: var(--muted); }
.btn-p:hover { color: #fff; }
</style>
