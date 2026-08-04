import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    pool: "forks",
    include: ["src/**/*.test.ts", "__tests__/**/*.test.ts"],
    exclude: ["dist/**", "node_modules/**", "**/*.e2e.test.ts"],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    clearMocks: true,
    restoreMocks: true,
    unstubEnvs: true,
    unstubGlobals: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/**/*.ts", "index.ts"],
      exclude: [
        "src/**/*.test.ts",
        "dist/**",
        "node_modules/**",
      ],
    },
  },
});
