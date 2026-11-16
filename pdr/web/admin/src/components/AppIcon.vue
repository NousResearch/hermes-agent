<script setup lang="ts">
import { computed } from 'vue'
import { PATHS, type IconName } from '../lib/icons'

const props = defineProps<{ name: IconName; size?: number; stroke?: number }>()
const paths = computed(() => PATHS[props.name])
</script>

<template>
  <svg
    :width="size ?? 16"
    :height="size ?? 16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    :stroke-width="stroke ?? 1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path v-for="(d, i) in paths" :key="i" :d="d" />
  </svg>
</template>
