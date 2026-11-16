<script setup lang="ts">
import { computed } from 'vue'
import type { ConfigScope } from '../api'
import ConfigEditor from '../components/ConfigEditor.vue'
import { useAuth } from '../stores/auth'

// 平台超级管理员可改 config.yaml（平台级）也可看租户覆盖；租户管理员只能做租户覆盖
const auth = useAuth()
const scopes = computed<ConfigScope[]>(() => (auth.can('sys:config') ? ['platform', 'tenant'] : ['tenant']))
</script>

<template>
  <ConfigEditor part="agent" :scopes="scopes" />
</template>
