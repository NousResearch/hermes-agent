import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/dashboard",
  testMatch: "v8-production-cutover.spec.ts",
  outputDir: "./test-results/v8-production-cutover",
  timeout: 45_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL: process.env.HERMES_AGENT_PRODUCTION_URL || "https://agent.tlccapitalgroup.com",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "mobile",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 390, height: 844 },
        isMobile: true,
      },
    },
  ],
});
