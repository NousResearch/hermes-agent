#!/usr/bin/env node
// Probe search on the LIVE docs site the way a visitor uses it.
//
// Search broke in production twice with a green build: once when the Algolia
// search key was deleted, once when Vercel access controls got in the way.
// Neither shows up in CI because CI never talks to the deployed site, so this
// script does: it reads the Algolia credentials out of the search-config.json
// the site is actually serving, queries Algolia with the site's Origin, opens
// the top hits, and then drives the real search modal in a headless browser.
//
// Usage: node apps/docs/scripts/check-live-search.mjs [--no-browser]
//   DOCS_URL       site to probe (default https://hermes-agent.nousresearch.com/docs)
//   PLAYWRIGHT_DIR directory holding node_modules/playwright (browser check)
// Exit 0 = search works; exit 1 = broken. Reasons go to stderr and, in
// Actions, to the step summary and `::error::` annotations (alert receivers
// read the annotations off the failed run).

import { appendFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

import { QUERIES, UA, checkAlgolia, checkHitPagesOpen, readDeployedConfig } from './live-search-probe.mjs';

const DOCS_URL = (process.env.DOCS_URL || 'https://hermes-agent.nousresearch.com/docs').replace(/\/$/, '');
// The whole probe gets one deadline, so a slow-drip server fails the check
// with time left to retry and alert instead of running into the job's timeout.
const PROBE_DEADLINE_MS = 240_000;

const failures = [];
const fail = (msg) => {
  failures.push(msg);
  console.error(`FAIL ${msg}`);
  if (process.env.GITHUB_ACTIONS) console.log(`::error title=Docs search::${msg.replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A')}`);
};
const ok = (msg) => console.log(`ok   ${msg}`);

async function checkBrowser() {
  const require = createRequire(path.join(process.env.PLAYWRIGHT_DIR || process.cwd(), 'noop.js'));
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ userAgent: UA });
    const algoliaErrors = [];
    page.on('response', (r) => {
      if (r.url().includes('algolia') && r.status() >= 400) algoliaErrors.push(`${r.status()} ${r.url().split('?')[0]}`);
    });
    page.on('requestfailed', (r) => {
      if (r.url().includes('algolia')) algoliaErrors.push(`${r.failure()?.errorText} ${r.url().split('?')[0]}`);
    });
    await page.goto(`${DOCS_URL}/`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
    // Fumadocs renders the search trigger itself (the Docusaurus
    // .DocSearch-Button is gone from the export). The full navbar trigger is
    // the visible one; the compact [data-search] icons are the fallback for
    // layouts that only render those.
    try {
      await page.locator('[data-search-full]').first().click({ timeout: 20_000 });
    } catch {
      await page.locator('[data-search]').first().click({ timeout: 20_000 });
    }
    // The open dialog is still @docsearch/react's DocSearchModal.
    await page.locator('.DocSearch-Input').fill(QUERIES[0], { timeout: 10_000 });
    try {
      await page.locator('.DocSearch-Hit').first().waitFor({ timeout: 15_000 });
    } catch {
      const shown = (await page.locator('.DocSearch-Modal').innerText().catch(() => '')).replace(/\s+/g, ' ').slice(0, 200);
      fail(`search modal showed no results for "${QUERIES[0]}" (${algoliaErrors.join('; ') || 'no Algolia error seen'}; modal: "${shown}")`);
      return;
    }
    ok(`browser: search modal lists ${await page.locator('.DocSearch-Hit').count()} results for "${QUERIES[0]}"`);
    for (const [locale, prefix, query] of [['en', '', 'skills hub'], ['zh-Hans', '/zh-Hans', '安装']]) {
      const response = await page.goto(`${DOCS_URL}${prefix}/search/?q=${encodeURIComponent(query)}`, {
        waitUntil: 'domcontentloaded', timeout: 30_000,
      });
      if (!response?.ok()) {
        fail(`full search page ${locale} returned HTTP ${response?.status() ?? 'no response'}`);
        continue;
      }
      try {
        const input = page.locator('#docs-search-input');
        await input.waitFor({ timeout: 15_000 });
        const results = page.locator('form[role="search"]').locator('..').locator('ul > li > a');
        await results.first().waitFor({ timeout: 15_000 });
        if (await input.inputValue() !== query) throw new Error('the URL query did not reach the input');
        if (locale === 'en') {
          const skills = new URL(`${DOCS_URL}/skills/`).pathname;
          await results.locator(`xpath=self::a[@href="${skills}" or @href="${skills.slice(0, -1)}"]`).first().waitFor({ timeout: 10_000 });
        }
        ok(`browser: ${locale} full search lists ${await results.count()} results for "${query}"`);
      } catch (error) {
        fail(`full search page ${locale} failed: ${error.message.split('\n')[0]}`);
      }
    }
  } finally {
    await browser.close();
  }
}

async function probe() {
  const found = await readDeployedConfig(DOCS_URL);
  if (found.error) {
    fail(found.error);
  } else {
    ok(found.note);
    const { failures: algoliaFailures, notes, hitUrls } = await checkAlgolia(found.config, { docsUrl: DOCS_URL });
    for (const note of notes) ok(note);
    for (const failure of algoliaFailures) fail(failure);
    const opened = await checkHitPagesOpen(hitUrls);
    for (const note of opened.notes) ok(note);
    for (const failure of opened.failures) fail(failure);
  }
  if (!process.argv.includes('--no-browser')) {
    await checkBrowser().catch((e) => fail(`browser check crashed: ${e.message.split('\n')[0]}`));
  }
}

// One exit path for clean failures, thrown errors and the deadline, so the
// alert always carries the real reason.
function finish() {
  if (process.env.GITHUB_STEP_SUMMARY) {
    const lines = failures.length ? failures.map((f) => `- ${f}`) : ['- all checks passed'];
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `### Docs search (${DOCS_URL})\n${lines.join('\n')}\n`);
  }
  process.exit(failures.length ? 1 : 0);
}

setTimeout(() => {
  fail(`probe did not finish within ${PROBE_DEADLINE_MS / 1000}s`);
  finish();
}, PROBE_DEADLINE_MS).unref();
probe()
  .catch((e) => fail(`probe crashed: ${e.cause?.message || e.message}`))
  .finally(finish);
