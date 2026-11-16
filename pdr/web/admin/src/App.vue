<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import ShellLayout from './components/ShellLayout.vue'

const route = useRoute()
const bare = computed(() => route.path === '/login')
</script>

<template>
  <RouterView v-if="bare" />
  <ShellLayout v-else>
    <RouterView v-slot="{ Component }">
      <component :is="Component" :key="route.fullPath" />
    </RouterView>
  </ShellLayout>
</template>
