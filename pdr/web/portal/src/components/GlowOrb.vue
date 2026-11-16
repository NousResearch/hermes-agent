<script setup lang="ts">
defineProps<{ size?: number }>()
</script>

<template>
  <span class="orb" :style="{ width: `${size ?? 64}px`, height: `${size ?? 64}px` }" aria-hidden="true">
    <span class="glow" />
    <span class="core" />
  </span>
</template>

<style scoped>
.orb { position: relative; display: inline-block; flex: none; }
.glow {
  position: absolute; inset: -35%; border-radius: 50%;
  background: radial-gradient(circle, rgba(150, 140, 255, 0.35), rgba(150, 200, 255, 0.12) 45%, transparent 70%);
  filter: blur(6px);
}
.core {
  position: absolute; inset: 0; border-radius: 50%;
  background:
    radial-gradient(circle at 32% 28%, #ffffff 0%, rgba(255, 255, 255, 0) 32%),
    radial-gradient(circle at 70% 75%, #9fd4ff 0%, rgba(159, 212, 255, 0) 45%),
    radial-gradient(circle at 25% 75%, #f3c6ff 0%, rgba(243, 198, 255, 0) 45%),
    radial-gradient(circle at 50% 50%, #8f86ff 0%, #b8b2ff 55%, #e7e5ff 100%);
  box-shadow: inset 0 -6px 14px rgba(110, 100, 240, 0.35), inset 0 4px 10px rgba(255, 255, 255, 0.8), 0 10px 26px rgba(120, 110, 250, 0.28);
  animation: float 6s ease-in-out infinite;
}
@keyframes float { 50% { transform: translateY(-4px) scale(1.02); } }
@media (prefers-reduced-motion: reduce) { .core { animation: none; } }
</style>
