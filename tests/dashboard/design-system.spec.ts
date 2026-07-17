import { expect, test, type Page, type TestInfo } from "@playwright/test";

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

test.describe("Hermes dashboard design system", () => {
  test("component gallery renders stable dashboard primitives", async ({ page }, testInfo) => {
    await page.goto("/design-system");
    await expect(page.getByRole("heading", { name: /Hermes Dashboard Design System/i })).toBeVisible();
    await expect(page.getByText("Status And Capacity")).toBeVisible();
    await expect(page.getByText("Tables And Filters")).toBeVisible();
    await expect(page.getByText("State Patterns")).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await captureDashboardScreenshot(page, testInfo, "design-system-gallery");
  });

  test("component gallery keeps core theme contrast readable", async ({ page }) => {
    await page.goto("/design-system");
    await expectReadableCoreContrast(page);
  });

  test("Hermes OS dashboard route keeps shell responsive", async ({ page }, testInfo) => {
    await page.goto("/hermes-os");
    await expect(page.getByRole("heading", { name: /Hermes OS/i }).first()).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await expectKeyboardFocus(page);
    await captureDashboardScreenshot(page, testInfo, "hermes-os-dashboard");
  });

  test("state surfaces expose empty, loading, and error states", async ({ page }) => {
    await page.goto("/design-system");
    await expect(page.getByText("Loading example")).toBeVisible();
    await expect(page.getByText("Empty example")).toBeVisible();
    await expect(page.getByText("Error example")).toBeVisible();
  });
});
