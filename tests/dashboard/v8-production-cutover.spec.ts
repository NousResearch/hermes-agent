import { expect, test, type Page, type TestInfo } from "@playwright/test";

const credentials = {
  username: process.env.HERMES_AGENT_DASHBOARD_USERNAME || "",
  password: process.env.HERMES_AGENT_DASHBOARD_PASSWORD || "",
};

const routes = [
  {
    path: "/package-native/media-engine",
    heading: /Media Engine Package-Native Ops/i,
    screenshot: "media-engine-production-package-native",
  },
  {
    path: "/package-native/khashi-vc",
    heading: /Khashi VC Package-Native ROC/i,
    screenshot: "khashi-vc-production-package-native",
  },
];

async function login(page: Page, nextPath: string) {
  await page.goto(`/login?next=${encodeURIComponent(nextPath)}`);
  await page.getByLabel("Username").fill(credentials.username);
  await page.getByLabel("Password").fill(credentials.password);
  await page.getByRole("button", { name: /^Sign in$/i }).click();
  await page.waitForURL((url) => url.pathname === nextPath, { timeout: 15_000 });
}

async function captureProductionScreenshot(page: Page, testInfo: TestInfo, name: string) {
  const screenshot = await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath(`${name}-${testInfo.project.name}.png`),
  });
  expect(screenshot.length).toBeGreaterThan(1_000);
}

test.describe("V8 production cutover evidence", () => {
  test("production agent status requires auth and exposes basic provider", async ({ request }) => {
    const response = await request.get("/api/status");
    expect(response.ok()).toBeTruthy();
    const status = await response.json();
    expect(status.auth_required).toBe(true);
    expect(status.auth_providers).toContain("basic");
  });

  for (const route of routes) {
    test(`${route.path} is protected before login`, async ({ page }) => {
      const response = await page.goto(route.path);
      expect(response?.status()).toBeGreaterThanOrEqual(200);
      await expect(page.getByRole("heading", { name: /Sign in/i })).toBeVisible();
      await expect(page.locator("input[name='username']")).toBeVisible();
      await expect(page.locator("input[name='password']")).toBeVisible();
    });

    test(`${route.path} renders after authenticated login`, async ({ page }, testInfo) => {
      test.skip(!credentials.username || !credentials.password, "Set HERMES_AGENT_DASHBOARD_USERNAME and HERMES_AGENT_DASHBOARD_PASSWORD to capture authenticated production evidence.");
      await login(page, route.path);
      await expect(page.getByRole("heading", { name: route.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: "Dashboard Snapshot Signals" })).toBeVisible();
      await expect(page.getByText("Adapter Retirement Gate")).toBeVisible();
      await captureProductionScreenshot(page, testInfo, route.screenshot);
    });
  }
});
