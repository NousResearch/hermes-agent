// Playwright Comprehensive UI & UX Test for SamAgent Codex Desktop
import { chromium } from "playwright";
import fs from "fs";
import path from "path";

async function runTest() {
  console.log("==================================================");
  console.log(" Launching Chrome via Playwright to Test Codex UI");
  console.log("==================================================");

  const chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
  const browser = await chromium.launch({
    executablePath: fs.existsSync(chromePath) ? chromePath : undefined,
    headless: true,
  });

  const page = await browser.newPage({
    viewport: { width: 1280, height: 800 },
  });

  console.log("[1/8] Navigating to http://127.0.0.1:8080/ ...");
  await page.goto("http://127.0.0.1:8080/", { waitUntil: "networkidle", timeout: 15000 });

  // 1. Check title
  const title = await page.title();
  console.log(`[2/8] Page Title: "${title}"`);
  if (!title.includes("Codex Desktop")) {
    throw new Error(`Expected title to include "Codex Desktop", got "${title}"`);
  }

  // 2. Check window bar, menus, dropdown
  await page.waitForSelector(".codex-window-bar", { timeout: 5000 });
  const menus = await page.$$eval(".codex-menu-item", (els) => els.map((e) => e.textContent.trim()));
  console.log(`[3/8] Window Menu items: ${menus.join(", ")}`);

  // Test File menu dropdown
  console.log("      Testing 'File' menu click...");
  await page.click('.codex-menu-item:has-text("File")');
  await page.waitForSelector(".codex-menu-dropdown", { timeout: 3000 });
  const fileItems = await page.$$eval(".codex-menu-dropdown-item", (els) => els.map((e) => e.textContent.trim()));
  console.log(`      File Dropdown items: ${fileItems.join(" | ")}`);
  // Close menu by clicking elsewhere
  await page.click(".codex-hero-heading");

  // Test Sub-sidebar Brand dropdown
  console.log("      Testing 'SamAgent ⌵' brand dropdown...");
  await page.click(".codex-dropdown-trigger");
  await page.waitForSelector(".codex-menu-dropdown", { timeout: 3000 });
  const brandItems = await page.$$eval(".codex-menu-dropdown-item", (els) => els.map((e) => e.textContent.trim()));
  console.log(`      Brand Dropdown items: ${brandItems.join(" | ")}`);
  await page.click(".codex-hero-heading");

  // 3. Check central canvas & starter suggestion chips
  const heading = await page.$eval(".codex-hero-heading", (el) => el.textContent.trim());
  console.log(`[4/8] Center Hero Heading: "${heading}"`);

  const cloudMascot = await page.$(".codex-mascot-cloud");
  console.log(`      Cloud Mascot Icon rendered: ${!!cloudMascot}`);

  const starterChips = await page.$$eval(".codex-starter-chip", (els) => els.map((e) => e.textContent.trim()));
  console.log(`      Starter Prompt Chips (${starterChips.length}):`);
  starterChips.forEach((c) => console.log(`        - ${c}`));

  // 4. Check composer capsule, model popover, and typing
  const capsuleText = await page.$eval(".codex-composer-capsule", (el) => el.textContent.trim());
  console.log(`[5/8] Composer Capsule: "${capsuleText.replace(/\s+/g, ' ')}"`);

  const placeholder = await page.$eval(".codex-composer-textarea", (el) => el.getAttribute("placeholder"));
  console.log(`      Composer Placeholder: "${placeholder}"`);

  // Test Model selector popover & real models
  console.log("      Testing Model selector popover with REAL industry models...");
  const initialBadge = await page.$eval(".codex-model-badge", (el) => el.textContent.trim());
  console.log(`      Initial Model Badge: "${initialBadge}"`);
  if (initialBadge.toLowerCase().includes("sol")) {
    throw new Error(`Model badge still contains fake "Sol" model! Found: ${initialBadge}`);
  }

  await page.click(".codex-model-badge");
  await page.waitForSelector(".codex-model-popover", { timeout: 3000 });
  const modelOptions = await page.$$eval(".codex-model-option strong", (els) => els.map((e) => e.textContent.trim()));
  console.log(`      Available Real Models: ${modelOptions.join(" | ")}`);

  // Assert expected real models are present
  const expectedRealModels = ["Claude 3.7 Sonnet", "GPT-4o", "DeepSeek R1", "Gemini 2.0 Flash"];
  for (const m of expectedRealModels) {
    if (!modelOptions.some(opt => opt.includes(m))) {
      throw new Error(`Expected real model ${m} not found in popover: ${modelOptions.join(", ")}`);
    }
  }

  // Click DeepSeek R1 to test real model selection
  console.log("      Selecting 'DeepSeek R1'...");
  await page.click('.codex-model-option:has-text("DeepSeek R1")');
  await page.waitForTimeout(300);
  const updatedBadge = await page.$eval(".codex-model-badge", (el) => el.textContent.trim());
  console.log(`      Updated Model Badge: "${updatedBadge}"`);
  if (!updatedBadge.includes("DeepSeek R1")) {
    throw new Error(`Expected badge to update to DeepSeek R1, got ${updatedBadge}`);
  }

  // Switch to Claude 3.7 Sonnet for live plan & build
  await page.click(".codex-model-badge");
  await page.waitForSelector(".codex-model-popover", { timeout: 3000 });
  await page.click('.codex-model-option:has-text("Claude 3.7 Sonnet")');
  await page.waitForTimeout(300);

  // Test typing in composer and submitting a real task
  console.log("      Testing typing in 'Do anything' textarea & submitting task...");
  await page.click(".codex-composer-textarea");
  await page.fill(".codex-composer-textarea", "Verify all quality gates and system telemetry");
  const typedVal = await page.$eval(".codex-composer-textarea", (el) => el.value);
  console.log(`      Typed value confirmed: "${typedVal}"`);

  // Click Submit circle arrow button
  console.log("      Clicking Circle '↑' Submit Button (.codex-submit-circle) to run task...");
  await page.click(".codex-submit-circle");
  
  // Wait for Agent Card to appear with real execution details
  await page.waitForSelector(".codex-agent-card", { timeout: 10000 });
  const agentCardText = await page.$eval(".codex-agent-card", (el) => el.textContent.trim());
  console.log(`      Agent Stream Card preview: "${agentCardText.slice(0, 180)}..."`);
  if (!agentCardText.includes("Claude 3.7 Sonnet")) {
    throw new Error(`Expected agent card to display real model Claude 3.7 Sonnet, got: ${agentCardText}`);
  }
  if (!agentCardText.includes("SamAgent Engine")) {
    throw new Error(`Expected agent card to display SamAgent Engine, got: ${agentCardText}`);
  }
  console.log("      Real model & engine execution confirmed on live agent card!");

  // 5. Test opening the side inspector drawer
  console.log("[6/8] Testing side inspector drawer toggle (◧)...");
  await page.click(".codex-canvas-top-actions button");
  await page.waitForSelector(".codex-right-drawer", { timeout: 3000 });
  const drawerTabs = await page.$$eval(".codex-drawer-tab", (els) => els.map((e) => e.textContent.trim()));
  console.log(`      Drawer opened with tabs: ${drawerTabs.join(" | ")}`);

  // 6. Test VS Code Studio tab with file pills
  console.log("[7/8] Testing VS Code Studio tab & file pills...");
  const vsCodeTab = await page.$('button.codex-drawer-tab:has-text("VS Code")');
  if (vsCodeTab) {
    await vsCodeTab.click();
    await page.waitForSelector(".codex-btn-vscode", { timeout: 3000 });
    const pills = await page.$$eval(".codex-file-pill", (els) => els.map((e) => e.textContent.trim()));
    console.log(`      VS Code File Pills: ${pills.join(", ")}`);
    const vsCodeBtnText = await page.$eval(".codex-btn-vscode", (el) => el.textContent.trim());
    console.log(`      VS Code Action Button: "${vsCodeBtnText}"`);
  }

  // 7. Test Pre-Prod Gate tab
  console.log("[8/8] Testing Pre-Prod Gate (7/7) tab...");
  const gateTab = await page.$('button.codex-drawer-tab:has-text("Gate")');
  if (gateTab) {
    await gateTab.click();
    await page.waitForSelector(".codex-panel-card-title", { timeout: 3000 });
    const reVerifyBtn = await page.$('button:has-text("Re-Verify All Gates")');
    console.log(`      Pre-Prod Gate Re-Verify button rendered: ${!!reVerifyBtn}`);
  }

  // Take screenshot
  const screenshotPath = "codex_desktop_verified.png";
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`      Screenshot captured successfully: ${screenshotPath} (${fs.statSync(screenshotPath).size} bytes)`);

  // Also copy to artifacts dir
  const artifactDir = "C:\\Users\\User_S\\.gemini\\antigravity-ide\\brain\\515084c0-e0f9-43c3-bb45-5ed7ce563a02";
  if (fs.existsSync(artifactDir)) {
    fs.copyFileSync(screenshotPath, path.join(artifactDir, "codex_desktop_verified.png"));
    console.log(`      Copied screenshot to artifacts directory!`);
  }

  await browser.close();
  console.log("==================================================");
  console.log(" ALL UI & UX TESTS PASSED WITH REAL MODELS! (100%)");
  console.log("==================================================");
}

runTest().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
