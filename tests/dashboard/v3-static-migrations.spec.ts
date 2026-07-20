import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const workspaceRoot = path.resolve(__dirname, "../../..");
const khashiRocHtml = path.join(workspaceRoot, "khashi-vc/public/roc/index.html");
const mediaEngineRoot = path.join(workspaceRoot, "media-engine");
const mediaDashboardBuilder = path.join(mediaEngineRoot, "tasks/build-unified-ops-dashboard.js");
const externalDashboardFixturesAvailable = fs.existsSync(khashiRocHtml) && fs.existsSync(mediaDashboardBuilder);

let mediaDashboardHtml = "";

test.beforeAll(() => {
  if (!externalDashboardFixturesAvailable) return;
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "media-engine-v3-playwright-"));
  execFileSync(
    process.execPath,
    [
      "tasks/build-unified-ops-dashboard.js",
      "--skip-live-stores",
      "--output-dir",
      outputDir,
      "--now",
      "2026-07-16T00:00:00.000Z",
    ],
    {
      cwd: mediaEngineRoot,
      stdio: "pipe",
    },
  );
  mediaDashboardHtml = path.join(outputDir, "index.html");
});

async function expectNoHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(2);
}

async function expectNoAxeViolations(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();

  expect(
    results.violations.map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      nodes: violation.nodes.map((node) => node.target),
    })),
  ).toEqual([]);
}

test.describe("V3 static dashboard migrations", () => {
  test.skip(!externalDashboardFixturesAvailable, "Requires sibling khashi-vc and media-engine workspaces.");

  test("Khashi ROC static adapter shell renders", async ({ page }) => {
    await page.goto(pathToFileURL(khashiRocHtml).href);
    await expect(page.locator(".hdk-body")).toBeVisible();
    await expect(page.locator(".hdk-shell")).toBeVisible();
    await expect(page.locator(".hdk-sidebar")).toBeVisible();
    await expect(page.locator(".hdk-main")).toBeVisible();
    await expect(page.locator(".hdk-button").first()).toBeVisible();
    expect(await page.locator(".hdk-metric-grid").count()).toBeGreaterThan(0);
    await expect(page.locator(".hdk-empty").first()).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectNoAxeViolations(page);
  });

  test("Media Engine generated ops dashboard renders V3 adapter surfaces", async ({ page }) => {
    await page.goto(pathToFileURL(mediaDashboardHtml).href);
    await expect(page.getByRole("heading", { name: "Media Engine Unified Ops" })).toBeVisible();
    await expect(page.locator(".hdk-shell")).toBeVisible();
    await expect(page.locator("[data-autopilot-control]")).toBeVisible();
    await expect(page.locator("[data-discord-preview]")).toBeVisible();
    await expect(page.locator(".hdk-metric-grid")).toBeVisible();
    await expect(page.locator(".hdk-table-wrap").first()).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectNoAxeViolations(page);
  });
});
