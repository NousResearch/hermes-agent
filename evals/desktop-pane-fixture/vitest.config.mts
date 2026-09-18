// From apps/desktop: npx vitest run --config ../../evals/desktop-pane-fixture/vitest.config.mts
// Replay a slow module evaluation without changing any pane behavior or deadlines.
import { defineConfig, mergeConfig } from 'vitest/config'

import desktopConfig from '../../apps/desktop/vite.config.ts'

export default defineConfig(env =>
  mergeConfig(desktopConfig(env), {
    plugins: [
      {
        name: 'slow-pane-store-evaluation',
        enforce: 'post',
        transform(code: string, id: string) {
          if (id.endsWith('/components/pane-shell/tree/store.ts')) {
            return `${code}\nconsole.info('[pane-fixture] delaying real tree store evaluation by 11000ms');\nawait new Promise(resolve => setTimeout(resolve, 11000));\n`
          }
        }
      }
    ],
    test: {
      environment: 'jsdom',
      setupFiles: ['./vitest.setup.ts'],
      include: ['src/store/session-pane-focus.test.ts'],
      globals: true,
      testTimeout: 15_000
    }
  })
)
