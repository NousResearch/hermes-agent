import { fileURLToPath } from 'node:url'

import { defineConfig } from 'vitest/config'

// Targeted native seam proof: no unrelated renderer Vite plugins or full app build.
export default defineConfig({
  root: fileURLToPath(new URL('../../', import.meta.url)),
  test: { environment: 'node', include: ['electron/pool-retirement-live.test.ts'] },
})
