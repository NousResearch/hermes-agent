import { mergeConfig } from 'vitest/config'

import viteConfig from './vite.config'

// Vitest runs the project's Vite config by default, but adds two overrides:
// `resolve.extensions` is hoisted to put TypeScript ahead of the compiled
// `.js` mirrors that ship under `src/`; `test.environment` fixes the runtime
// to `jsdom` (matches what `npm run test:ui` previously hard-coded).
//
// Without these, vitest prefers `utils.js` over `utils.ts` and silently runs
// tests against stale build output instead of the live TypeScript source.
export default mergeConfig(viteConfig, {
  resolve: {
    extensions: ['.ts', '.tsx', '.mts', '.mjs', '.js', '.jsx', '.json']
  },
  test: {
    environment: 'jsdom'
  }
})
