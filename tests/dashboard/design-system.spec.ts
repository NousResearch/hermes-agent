import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { mockDashboardBackend } from "./backend-mocks";

async function expectNoHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(() => {
    const documentElement = document.documentElement;
    return documentElement.scrollWidth - documentElement.clientWidth;
  });
  expect(overflow).toBeLessThanOrEqual(2);
}

async function expectKeyboardFocus(page: Page) {
  await page.keyboard.press("Tab");
  const active = await page.evaluate(() => {
    const element = document.activeElement;
    if (!element) return { tag: "", label: "" };
    return {
      tag: element.tagName.toLowerCase(),
      label: element.getAttribute("aria-label") || element.textContent || element.getAttribute("title") || "",
    };
  });
  expect(["a", "button", "input", "select", "textarea"]).toContain(active.tag);
  expect(active.label.trim().length).toBeGreaterThan(0);
}

async function captureDashboardScreenshot(page: Page, testInfo: TestInfo, name: string) {
  const screenshot = await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath(`${name}-${testInfo.project.name}.png`),
  });
  expect(screenshot.length).toBeGreaterThan(1_000);
}

async function expectReadableCoreContrast(page: Page) {
  const results = await page.evaluate(() => {
    const selectors = ["h1", "h2", "button", "a"];

    function normalizeCssColor(value: string) {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) return value;
      context.fillStyle = "#000000";
      context.fillStyle = value;
      return context.fillStyle;
    }

    function rgbFromCss(value: string) {
      const normalized = normalizeCssColor(value);
      const hex = normalized.match(/^#([0-9a-f]{6})$/i);
      if (hex) {
        return {
          r: Number.parseInt(hex[1].slice(0, 2), 16) / 255,
          g: Number.parseInt(hex[1].slice(2, 4), 16) / 255,
          b: Number.parseInt(hex[1].slice(4, 6), 16) / 255,
          a: 1,
        };
      }
      const srgb = normalized.match(/^color\(srgb\s+([.\d]+)\s+([.\d]+)\s+([.\d]+)(?:\s*\/\s*([.\d]+))?\)$/);
      if (srgb) {
        return {
          r: Number(srgb[1]),
          g: Number(srgb[2]),
          b: Number(srgb[3]),
          a: srgb[4] === undefined ? 1 : Number(srgb[4]),
        };
      }
      const match = normalized.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([.\d]+))?\)/);
      if (!match) return null;
      return {
        r: Number(match[1]) / 255,
        g: Number(match[2]) / 255,
        b: Number(match[3]) / 255,
        a: match[4] === undefined ? 1 : Number(match[4]),
      };
    }

    function linear(channel: number) {
      return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    }

    function luminance(color: { r: number; g: number; b: number }) {
      return 0.2126 * linear(color.r) + 0.7152 * linear(color.g) + 0.0722 * linear(color.b);
    }

    function ratio(foreground: { r: number; g: number; b: number }, background: { r: number; g: number; b: number }) {
      const lighter = Math.max(luminance(foreground), luminance(background));
      const darker = Math.min(luminance(foreground), luminance(background));
      return (lighter + 0.05) / (darker + 0.05);
    }

    function backgroundFor(element: Element) {
      let current: Element | null = element;
      while (current) {
        const color = rgbFromCss(getComputedStyle(current).backgroundColor);
        if (color && color.a > 0.05) return color;
        current = current.parentElement;
      }
      return { r: 1, g: 1, b: 1, a: 1 };
    }

    return selectors.flatMap((selector) =>
      Array.from(document.querySelectorAll(selector))
        .filter((element) => {
          const rect = element.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        })
        .slice(0, 4)
        .map((element) => {
          const foreground = rgbFromCss(getComputedStyle(element).color);
          const background = backgroundFor(element);
          return {
            selector,
            text: (element.textContent || element.getAttribute("aria-label") || "").trim().slice(0, 80),
            contrast: foreground ? ratio(foreground, background) : 0,
          };
        }),
    );
  });

  expect(results.length).toBeGreaterThan(0);
  for (const result of results) {
    expect(result.contrast, `${result.selector} ${result.text}`).toBeGreaterThanOrEqual(3);
  }
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

test.describe("Hermes dashboard design system", () => {
  test.beforeEach(async ({ page }) => {
    await mockDashboardBackend(page);
  });

  test("component gallery renders stable dashboard primitives", async ({ page }, testInfo) => {
    await page.goto("/design-system");
    await expect(page.getByRole("heading", { name: /Hermes Dashboard Design System/i })).toBeVisible();
    await expect(page.getByText("Status And Capacity")).toBeVisible();
    await expect(page.getByText("Tables And Filters")).toBeVisible();
    await expect(page.getByText("V7 Full-Page Dashboard Recipes")).toBeVisible();
    await expect(page.getByText("State Patterns")).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await captureDashboardScreenshot(page, testInfo, "design-system-gallery");
  });

  test("component gallery keeps core theme contrast readable", async ({ page }) => {
    await page.goto("/design-system");
    await expectReadableCoreContrast(page);
    await expectNoAxeViolations(page);
  });

  test("Hermes OS dashboard route keeps shell responsive", async ({ page }, testInfo) => {
    await page.goto("/hermes-os");
    await expect(page.getByRole("heading", { name: /Hermes OS/i }).first()).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await expectNoAxeViolations(page);
    await captureDashboardScreenshot(page, testInfo, "hermes-os-dashboard");
  });

  test("executive summary dashboard route keeps cross-project rollups responsive", async ({ page }, testInfo) => {
    await page.goto("/executive-summary");
    await expect(page.getByRole("heading", { name: /TLC Executive Summary/i }).first()).toBeVisible();
    await expect(page.getByRole("heading", { name: "Project Scorecards" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Cost, Capacity, And Throughput" })).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await expectNoAxeViolations(page);
    await captureDashboardScreenshot(page, testInfo, "executive-summary-dashboard");
  });

  test("package-native migration dashboard route keeps V8 work trackable", async ({ page }, testInfo) => {
    await page.goto("/dashboard-migrations");
    await expect(page.getByRole("heading", { name: /Package-Native Dashboard Migrations/i }).first()).toBeVisible();
    await expect(page.getByRole("cell", { name: "Media Engine Ops" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "Khashi VC ROC" })).toBeVisible();
    await expect(page.getByText("Retire Adapter")).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await expectNoAxeViolations(page);
    await captureDashboardScreenshot(page, testInfo, "package-native-migrations-dashboard");
  });

  test("central command route keeps V12 executive layer visible", async ({ page }, testInfo) => {
    await page.goto("/central-command");
    await expect(page.getByRole("heading", { name: /Hermes Central Command/i }).first()).toBeVisible();
    await expect(page.getByRole("heading", { name: "Daily Cross-Project Brief" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Business Impact Read" })).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await expectNoAxeViolations(page);
    await captureDashboardScreenshot(page, testInfo, "central-command-dashboard");
  });

  test("theme system route keeps V13 brand profiles visible", async ({ page }, testInfo) => {
    await page.goto("/theme-system");
    await expect(page.getByRole("heading", { name: /Multi-Brand Dashboard Themes/i }).first()).toBeVisible();
    await expect(page.getByRole("cell", { name: "TLC Base" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "Khashi Research" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Token Swatches" })).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await expectNoAxeViolations(page);
    await captureDashboardScreenshot(page, testInfo, "theme-system-dashboard");
  });

  test("dashboard marketplace route keeps V14 plugin registry visible", async ({ page }, testInfo) => {
    await page.goto("/dashboard-marketplace");
    await expect(page.getByRole("heading", { name: /Dashboard Marketplace/i }).first()).toBeVisible();
    await expect(page.getByRole("cell", { name: "Khashi VC ROC" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "Media Engine Ops" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Permission-Aware Commands" })).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await expectNoAxeViolations(page);
    await captureDashboardScreenshot(page, testInfo, "dashboard-marketplace");
  });

  const operatingSystemRoutes = [
    {
      route: "/live-signals",
      heading: /Live Project Signal Integration/i,
      secondary: "Signal Integrations",
      screenshot: "live-signals-dashboard",
    },
    {
      route: "/task-routing",
      heading: /Agent Task Routing/i,
      secondary: "Work Intake Queue",
      screenshot: "task-routing-dashboard",
    },
    {
      route: "/decision-ledger",
      heading: /Memory And Decision Ledger/i,
      secondary: "Decision Records",
      screenshot: "decision-ledger-dashboard",
    },
    {
      route: "/model-routing",
      heading: /Model And Cost Routing/i,
      secondary: "Routing Policies",
      screenshot: "model-routing-dashboard",
    },
    {
      route: "/operating-loops",
      heading: /Autonomous Operating Loops/i,
      secondary: "Operating Loop Registry",
      screenshot: "operating-loops-dashboard",
    },
    {
      route: "/permission-security",
      heading: /Secure Tool And Permission Layer/i,
      secondary: "Permission Policies",
      screenshot: "permission-security-dashboard",
    },
    {
      route: "/business-os",
      heading: /TLC Business Operating System/i,
      secondary: "Business Unit Scorecards",
      screenshot: "business-os-dashboard",
    },
  ];

  for (const routeConfig of operatingSystemRoutes) {
    test(`${routeConfig.route} keeps V15-V21 operating-system layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  const autonomyReadinessRoutes = [
    {
      route: "/project-snapshots",
      heading: /Live Project Snapshot Contracts/i,
      secondary: "Snapshot Contract Registry",
      screenshot: "project-snapshots-dashboard",
    },
    {
      route: "/durable-memory",
      heading: /Durable Memory And Decision Store/i,
      secondary: "Memory Store Readiness",
      screenshot: "durable-memory-dashboard",
    },
    {
      route: "/permission-runtime",
      heading: /Permission Enforcement Runtime/i,
      secondary: "Runtime Enforcement Gates",
      screenshot: "permission-runtime-dashboard",
    },
    {
      route: "/cost-governor",
      heading: /Model Router And Cost Governor/i,
      secondary: "Provider Routing Policy",
      screenshot: "cost-governor-dashboard",
    },
    {
      route: "/loop-runner",
      heading: /Operating Loop Runner/i,
      secondary: "Loop Execution Readiness",
      screenshot: "loop-runner-dashboard",
    },
    {
      route: "/business-command",
      heading: /Cross-Business Command Center/i,
      secondary: "Business Command Rollups",
      screenshot: "business-command-dashboard",
    },
    {
      route: "/agent-workbench",
      heading: /Agent Workbench/i,
      secondary: "Supervised Workbench Flow",
      screenshot: "agent-workbench-dashboard",
    },
    {
      route: "/evaluation-gates",
      heading: /Evaluation And Quality Gates/i,
      secondary: "Quality Gate Registry",
      screenshot: "evaluation-gates-dashboard",
    },
    {
      route: "/autonomy-readiness",
      heading: /Production Autonomy Readiness/i,
      secondary: "Autonomy Readiness Checklist",
      screenshot: "autonomy-readiness-dashboard",
    },
  ];

  for (const routeConfig of autonomyReadinessRoutes) {
    test(`${routeConfig.route} keeps V22-V30 autonomy-readiness layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expect(page.getByRole("heading", { name: "Runtime Evidence" })).toBeVisible();
      await expect(page.getByRole("button", { name: /Run readiness check/i })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  const executiveOperatingRoutes = [
    {
      route: "/project-registry",
      heading: /Production Project Registry/i,
      secondary: "Project Registry Coverage",
      screenshot: "project-registry-dashboard",
    },
    {
      route: "/telemetry-fabric",
      heading: /Telemetry Fabric/i,
      secondary: "Telemetry Signal Fabric",
      screenshot: "telemetry-fabric-dashboard",
    },
    {
      route: "/incident-command",
      heading: /Incident Command/i,
      secondary: "Incident Response Queue",
      screenshot: "incident-command-dashboard",
    },
    {
      route: "/deployment-promotion",
      heading: /Deployment Promotion Rail/i,
      secondary: "Promotion Gate Checklist",
      screenshot: "deployment-promotion-dashboard",
    },
    {
      route: "/secrets-posture",
      heading: /Secrets And Access Posture/i,
      secondary: "Secrets Posture Matrix",
      screenshot: "secrets-posture-dashboard",
    },
    {
      route: "/data-source-catalog",
      heading: /Data Source Catalog/i,
      secondary: "Source Coverage Catalog",
      screenshot: "data-source-catalog-dashboard",
    },
    {
      route: "/finance-attribution",
      heading: /Finance And Cost Attribution/i,
      secondary: "Cost Attribution Model",
      screenshot: "finance-attribution-dashboard",
    },
    {
      route: "/learning-engine",
      heading: /Learning Engine/i,
      secondary: "Learning Evidence Loop",
      screenshot: "learning-engine-dashboard",
    },
    {
      route: "/agent-eval-lab",
      heading: /Agent Evaluation Lab/i,
      secondary: "Provider Evaluation Matrix",
      screenshot: "agent-eval-lab-dashboard",
    },
    {
      route: "/executive-cockpit",
      heading: /Executive Cockpit/i,
      secondary: "Executive Operating Cockpit",
      screenshot: "executive-cockpit-dashboard",
    },
  ];

  for (const routeConfig of executiveOperatingRoutes) {
    test(`${routeConfig.route} keeps V31-V40 executive operating-system layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expect(page.getByRole("heading", { name: "Runtime Evidence" })).toBeVisible();
      await expect(page.getByRole("button", { name: /Run readiness check/i })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  const liveOperationsRoutes = [
    {
      route: "/production-verification",
      heading: /Live Production Verification Runner/i,
      secondary: "Production Verification Checks",
      screenshot: "production-verification-dashboard",
    },
    {
      route: "/command-gates",
      heading: /Command Gate Runtime/i,
      secondary: "Command Gate Matrix",
      screenshot: "command-gates-dashboard",
    },
    {
      route: "/telemetry-adapters",
      heading: /Project Telemetry Adapter Kit/i,
      secondary: "Telemetry Adapter Contract",
      screenshot: "telemetry-adapters-dashboard",
    },
    {
      route: "/incident-ingestion",
      heading: /Incident Ingestion And Escalation/i,
      secondary: "Incident Ingestion Rules",
      screenshot: "incident-ingestion-dashboard",
    },
    {
      route: "/promotion-runner",
      heading: /Shared Deployment Promotion Runner/i,
      secondary: "Promotion Runner Gates",
      screenshot: "promotion-runner-dashboard",
    },
    {
      route: "/secret-scanner",
      heading: /Secrets Posture Scanner/i,
      secondary: "Secret Presence Matrix",
      screenshot: "secret-scanner-dashboard",
    },
    {
      route: "/cost-attribution-engine",
      heading: /Cost Attribution Engine/i,
      secondary: "Cost Attribution Inputs",
      screenshot: "cost-attribution-engine-dashboard",
    },
    {
      route: "/learning-ingestion",
      heading: /Learning Ingestion Pipeline/i,
      secondary: "Learning Ingestion Sources",
      screenshot: "learning-ingestion-dashboard",
    },
    {
      route: "/model-eval-harness",
      heading: /Agent And Model Eval Harness/i,
      secondary: "Eval Harness Coverage",
      screenshot: "model-eval-harness-dashboard",
    },
    {
      route: "/circuit-breakers",
      heading: /Runtime Circuit Breakers/i,
      secondary: "Circuit Breaker Controls",
      screenshot: "circuit-breakers-dashboard",
    },
  ];

  for (const routeConfig of liveOperationsRoutes) {
    test(`${routeConfig.route} keeps V41-V50 live operations layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expect(page.getByRole("heading", { name: "Runtime Evidence" })).toBeVisible();
      await expect(page.getByRole("button", { name: /Run readiness check/i })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  const boundaryClosureRoutes = [
    {
      route: "/production-sweep",
      heading: /Production DNS And Health Sweep/i,
      secondary: "Live Production Sweep",
      screenshot: "production-sweep-dashboard",
    },
    {
      route: "/hetzner-promotion-execution",
      heading: /Hetzner Promotion Execution/i,
      secondary: "Promotion Execution Rail",
      screenshot: "hetzner-promotion-execution-dashboard",
    },
    {
      route: "/command-gate-coverage",
      heading: /Command Gate Coverage Auditor/i,
      secondary: "Command Gate Coverage",
      screenshot: "command-gate-coverage-dashboard",
    },
    {
      route: "/project-adapter-rollout",
      heading: /Project Adapter Rollout/i,
      secondary: "Adapter Adoption Matrix",
      screenshot: "project-adapter-rollout-dashboard",
    },
    {
      route: "/incident-automation",
      heading: /Incident Automation Engine/i,
      secondary: "Automated Incident Sources",
      screenshot: "incident-automation-dashboard",
    },
    {
      route: "/live-secret-scan",
      heading: /Live Secret Presence Scan/i,
      secondary: "Secret Presence Sweep",
      screenshot: "live-secret-scan-dashboard",
    },
    {
      route: "/cost-reconciliation",
      heading: /Cost Reconciliation Import/i,
      secondary: "Cost Reconciliation Inputs",
      screenshot: "cost-reconciliation-dashboard",
    },
    {
      route: "/outcome-learning-feeds",
      heading: /Outcome Learning Feeds/i,
      secondary: "Outcome Feed Sources",
      screenshot: "outcome-learning-feeds-dashboard",
    },
    {
      route: "/golden-eval-execution",
      heading: /Golden Eval Execution/i,
      secondary: "Golden Task Runs",
      screenshot: "golden-eval-execution-dashboard",
    },
    {
      route: "/hard-breaker-enforcement",
      heading: /Hard Circuit Breaker Enforcement/i,
      secondary: "Hard Enforcement Checks",
      screenshot: "hard-breaker-enforcement-dashboard",
    },
  ];

  for (const routeConfig of boundaryClosureRoutes) {
    test(`${routeConfig.route} keeps V51-V60 boundary closure layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expect(page.getByRole("heading", { name: "Runtime Evidence" })).toBeVisible();
      await expect(page.getByRole("button", { name: /Run readiness check/i })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  const liveAdapterRoutes = [
    { route: "/network-runner-adapter", heading: /Network Runner Adapter/i, secondary: "Network Adapter Contract", screenshot: "network-runner-adapter-dashboard" },
    { route: "/hetzner-ssh-adapter", heading: /Hetzner SSH Adapter/i, secondary: "Hetzner SSH Execution Contract", screenshot: "hetzner-ssh-adapter-dashboard" },
    { route: "/secret-provider-adapter", heading: /Secret Provider Adapter/i, secondary: "Secret Provider Contracts", screenshot: "secret-provider-adapter-dashboard" },
    { route: "/billing-provider-adapter", heading: /Billing Provider Adapter/i, secondary: "Billing Adapter Contract", screenshot: "billing-provider-adapter-dashboard" },
    { route: "/project-outcome-emitter", heading: /Project Outcome Emitter/i, secondary: "Outcome Emitter Contract", screenshot: "project-outcome-emitter-dashboard" },
    { route: "/provider-eval-runner", heading: /Provider Eval Runner/i, secondary: "Provider Eval Runner Contract", screenshot: "provider-eval-runner-dashboard" },
    { route: "/breaker-middleware", heading: /Breaker Middleware SDK/i, secondary: "Breaker Middleware Contract", screenshot: "breaker-middleware-dashboard" },
    { route: "/incident-subscriptions", heading: /Incident Subscription Bus/i, secondary: "Incident Subscription Rules", screenshot: "incident-subscriptions-dashboard" },
    { route: "/evidence-artifact-store", heading: /Evidence Artifact Store/i, secondary: "Evidence Artifact Index", screenshot: "evidence-artifact-store-dashboard" },
    { route: "/release-train-orchestrator", heading: /Release Train Orchestrator/i, secondary: "Release Train Gates", screenshot: "release-train-orchestrator-dashboard" },
  ];

  for (const routeConfig of liveAdapterRoutes) {
    test(`${routeConfig.route} keeps V61-V70 live adapter layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expect(page.getByRole("heading", { name: "Runtime Evidence" })).toBeVisible();
      await expect(page.getByRole("button", { name: /Run readiness check/i })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  const operationalReadinessRoutes = [
    { route: "/production-screenshot-runner", heading: /Production Screenshot Runner/i, secondary: "Screenshot Capture Gates", screenshot: "production-screenshot-runner-dashboard" },
    { route: "/hetzner-promotion-transport", heading: /Hetzner Promotion Transport/i, secondary: "Remote Promotion Execution", screenshot: "hetzner-promotion-transport-dashboard" },
    { route: "/server-secret-posture-scanner", heading: /Server Secret Posture Scanner/i, secondary: "Server Secret Presence", screenshot: "server-secret-posture-scanner-dashboard" },
    { route: "/incident-notification-fanout", heading: /Incident Notification Fanout/i, secondary: "Incident Fanout Rules", screenshot: "incident-notification-fanout-dashboard" },
    { route: "/durable-artifact-backend", heading: /Durable Artifact Backend/i, secondary: "Artifact Backend Policy", screenshot: "durable-artifact-backend-dashboard" },
    { route: "/remaining-project-outcome-adapters", heading: /Remaining Project Outcome Adapters/i, secondary: "Outcome Adapter Adoption", screenshot: "remaining-project-outcome-adapters-dashboard" },
    { route: "/breaker-middleware-rollout", heading: /Breaker Middleware Rollout/i, secondary: "Live Path Enforcement", screenshot: "breaker-middleware-rollout-dashboard" },
    { route: "/provider-eval-execution", heading: /Provider Eval Execution/i, secondary: "Provider Eval Execution", screenshot: "provider-eval-execution-dashboard" },
    { route: "/billing-provider-integrations", heading: /Billing Provider Integrations/i, secondary: "Billing Integration Sources", screenshot: "billing-provider-integrations-dashboard" },
    { route: "/release-train-execution", heading: /Release Train Execution/i, secondary: "Release Train Execution Gates", screenshot: "release-train-execution-dashboard" },
  ];

  for (const routeConfig of operationalReadinessRoutes) {
    test(`${routeConfig.route} keeps V71-V80 operational readiness layer visible`, async ({ page }, testInfo) => {
      await page.goto(routeConfig.route);
      await expect(page.getByRole("heading", { name: routeConfig.heading }).first()).toBeVisible();
      await expect(page.getByRole("heading", { name: routeConfig.secondary, exact: true })).toBeVisible();
      await expect(page.getByRole("heading", { name: "Runtime Evidence" })).toBeVisible();
      await expect(page.getByRole("button", { name: /Run readiness check/i })).toBeVisible();
      await expectNoHorizontalOverflow(page);
      await expectKeyboardFocus(page);
      await expectNoAxeViolations(page);
      await captureDashboardScreenshot(page, testInfo, routeConfig.screenshot);
    });
  }

  test("state surfaces expose empty, loading, and error states", async ({ page }) => {
    await page.goto("/design-system");
    await expect(page.getByText("Loading example")).toBeVisible();
    await expect(page.getByText("Empty example")).toBeVisible();
    await expect(page.getByText("Error example")).toBeVisible();
    await expect(page.getByText("Disabled example")).toBeVisible();
  });

  test("gallery exposes V2 launcher, command safety, and copyable examples", async ({ page }) => {
    await page.goto("/design-system");
    await expect(page.getByRole("heading", { name: "Launcher Pattern", exact: true })).toBeVisible();
    await expect(page.getByText("Health check passed.").first()).toBeVisible();
    await expect(page.getByText("missing production URL")).toBeVisible();
    await expect(page.getByText("No dashboard manifests are registered.")).toBeVisible();
    await expect(page.getByText("Admin approval required.")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Copyable Usage Examples" })).toBeVisible();
    await expect(page.getByText("Dashboard shell")).toBeVisible();
    await expect(page.getByText("Launcher health")).toBeVisible();
  });
});
