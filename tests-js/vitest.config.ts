import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    include: ['**/*.test.ts'],
    // TEMPORARY: this job moved from a 32-core Larger Runner to the
    // standard 4-core `ubuntu-latest` (GitHub Free has no Larger Runners),
    // and vitest's 5000ms default is too tight for that much less CPU per
    // test. 15s gives headroom without masking genuinely hung tests.
    testTimeout: 15_000,
  },
})
