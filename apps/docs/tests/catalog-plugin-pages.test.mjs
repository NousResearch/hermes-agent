// Tests for the pure data builders of the build-time plugin/author page layer
// (apps/docs/src/lib/plugin-pages.ts) — the port of the Docusaurus
// plugin-catalog-pages plugin's loadContent/contentLoaded step.
//
// Only the pure builders and the degrade path are covered: the fs/network layer
// needs the generated public/api/plugins.json snapshot and the README renderer,
// which the build exercises. Node runs the .ts module directly via type stripping,
// so no test runner install is needed.
//
// Run: node --test apps/docs/tests/   (from apps/docs)

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import {
  buildAuthorPageData,
  buildPluginPageData,
  decodeCatalogParam,
  groupAuthors,
  listAuthorPageParams,
  listPluginPageParams,
} from "../src/lib/plugin-pages.ts";

const entry = (overrides = {}) => ({
  name: "zeta",
  description: "Does zeta things.",
  repo: "https://github.com/acme/zeta",
  sha: "0123456789abcdef0123456789abcdef01234567",
  shaShort: "0123456",
  tier: "official",
  category: "memory",
  maintainer: "Ana",
  maintainerSlug: "ana",
  installCommand: "hermes plugins install zeta",
  ...overrides,
});

const ENTRIES = [
  entry(), // zeta · ana · stars-first position 1
  entry({ name: "alpha", maintainer: "Ana", maintainerSlug: "ana", tier: "community", stars: 5 }),
  entry({ name: "beta", maintainer: "Bo", maintainerSlug: "bo", category: "web", stars: 2 }),
  entry({ name: "orphan", maintainer: "", maintainerSlug: undefined, tier: "community" }),
];

const SNAPSHOT = {
  entries: ENTRIES,
  meta: { generatedAt: "2026-10-20T00:00:00Z", starsFetchedAt: "2026-10-21T00:00:00Z" },
};

// ── groupAuthors ─────────────────────────────────────────────────────────────

test("groupAuthors preserves the catalog's stars-first order inside each group", () => {
  const groups = groupAuthors(ENTRIES);
  const ana = groups.find((g) => g.slug === "ana");
  assert.deepEqual(ana.plugins.map((p) => p.name), ["zeta", "alpha"]);
});

test("groupAuthors keys groups by first appearance of the maintainer slug", () => {
  const groups = groupAuthors(ENTRIES);
  assert.deepEqual(groups.map((g) => g.slug), ["ana", "bo", "unknown"]);
});

test("groupAuthors falls back to 'unknown' and to the slug as the display name", () => {
  const groups = groupAuthors(ENTRIES);
  const unknown = groups.find((g) => g.slug === "unknown");
  assert.equal(unknown.name, "unknown"); // entry.maintainer was empty
  assert.deepEqual(unknown.plugins.map((p) => p.name), ["orphan"]);
});

// ── buildPluginPageData ───────────────────────────────────────────────────────

test("buildPluginPageData returns null for a name that is not in the catalog", () => {
  assert.equal(buildPluginPageData(SNAPSHOT, "missing"), null);
});

test("buildPluginPageData carries the rendered README through", () => {
  const data = buildPluginPageData(SNAPSHOT, "zeta", {
    html: "<h1>zeta</h1>",
    url: "https://raw.githubusercontent.com/acme/zeta/0123/README.md",
  });
  assert.equal(data.readmeHtml, "<h1>zeta</h1>");
  assert.equal(data.readmeSourceUrl, "https://raw.githubusercontent.com/acme/zeta/0123/README.md");
});

test("buildPluginPageData leaves the README section off when it did not resolve", () => {
  const data = buildPluginPageData(SNAPSHOT, "zeta", null);
  assert.equal(data.readmeHtml, null);
  assert.equal(data.readmeSourceUrl, null);
});

test("buildPluginPageData attaches the author and the sibling shelf, excluding the entry itself", () => {
  const data = buildPluginPageData(SNAPSHOT, "alpha");
  assert.deepEqual(data.author, { slug: "ana", name: "Ana", count: 2 });
  assert.deepEqual(data.moreByAuthor.map((p) => p.name), ["zeta"]);
});

test("buildPluginPageData survives an entry whose maintainer is absent from the map", () => {
  // An entry with a maintainerSlug no other entry shares still gets an author group.
  const data = buildPluginPageData(SNAPSHOT, "beta");
  assert.deepEqual(data.author, { slug: "bo", name: "Bo", count: 1 });
  assert.deepEqual(data.moreByAuthor, []);
});

test("buildPluginPageData nulls empty-string meta values (old `|| null` semantics)", () => {
  const snap = { entries: ENTRIES, meta: { generatedAt: "", starsFetchedAt: undefined } };
  const data = buildPluginPageData(snap, "zeta");
  assert.equal(data.generatedAt, null);
  assert.equal(data.starsFetchedAt, null);
});

test("buildPluginPageData stamps meta timestamps onto the page", () => {
  const data = buildPluginPageData(SNAPSHOT, "zeta");
  assert.equal(data.generatedAt, "2026-10-20T00:00:00Z");
  assert.equal(data.starsFetchedAt, "2026-10-21T00:00:00Z");
});

// ── buildAuthorPageData ──────────────────────────────────────────────────────

test("buildAuthorPageData returns null for an unknown slug", () => {
  assert.equal(buildAuthorPageData(SNAPSHOT, "nobody"), null);
});

test("buildAuthorPageData keeps the catalog's order and the author record in sync", () => {
  const data = buildAuthorPageData(SNAPSHOT, "ana");
  assert.deepEqual(data.author, { slug: "ana", name: "Ana", count: 2 });
  assert.deepEqual(data.plugins.map((p) => p.name), ["zeta", "alpha"]);
  assert.equal(data.generatedAt, "2026-10-20T00:00:00Z");
});

// ── decodeCatalogParam ────────────────────────────────────────────────────────

test("decodeCatalogParam decodes URL-encoded params", () => {
  assert.equal(decodeCatalogParam("a%20b"), "a b");
  assert.equal(decodeCatalogParam("plain-slug"), "plain-slug");
});

test("decodeCatalogParam passes malformed sequences through untouched", () => {
  assert.equal(decodeCatalogParam("100%"), "100%");
  assert.equal(decodeCatalogParam("%zz"), "%zz");
});

// ── degrade path: a missing snapshot generates zero pages, never a failure ────

test("a missing plugins.json yields zero static params (degrade, not fail)", async () => {
  if (existsSync(resolve("public", "api", "plugins.json"))) {
    // A generated snapshot exists in this checkout — the degrade path can't be
    // observed here; the build itself covers the populated path.
    return;
  }
  assert.deepEqual(await listPluginPageParams(), []);
  assert.deepEqual(await listAuthorPageParams(), []);
});
