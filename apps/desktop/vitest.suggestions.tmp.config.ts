import { resolve } from 'node:path'

import { defineConfig } from 'vitest/config'

// Minimal harness for one pure module: the app's own vite.config.ts pulls in
// plugins that are not installed in this checkout, and this test needs none of
// them (no DOM, no JSX, one import that resolves to a type).
const root = '/home/hamishburke/Developer/hermes-agent-composer-prompts/apps/desktop'

export default defineConfig({
  resolve: { alias: { '@': resolve(root, 'src') } },
  root,
  test: {
    environment: 'node',
    include: ['src/app/chat/composer/prompt-suggestions.test.ts']
  }
})
