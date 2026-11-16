<script setup lang="ts">
import ConfigEditor from '../components/ConfigEditor.vue'
</script>

<template>
  <!-- PDR 自身的配置只有平台级（写入 config.yaml），租户不能覆盖 -->
  <ConfigEditor part="pdr" :scopes="['platform']" />
</template>
