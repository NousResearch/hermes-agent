// Behaviour tests for the read-only static-export validator
// (apps/docs/scripts/validate-output.mjs).
//
// The validator is the CI gate invoked after `npm run build`
// (.github/workflows/docs-site-checks.yml, "Validate static export output"),
// so every test drives the real CLI / callable against filesystem fixtures
// shaped like the export: a temp repo root with apps/docs/{content,public,out,.next}.
// The fixture's sitemap and legacy-alias pages are produced by the REAL
// postbuild script, so the validator is checked against the producer recipe
// rather than a reimplementation.
//
// Run: TMPDIR=… node --test tests/output-validation.test.mjs   (from apps/docs)

import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, cpSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { validateOutput } from "../scripts/validate-output.mjs";

const validatorCli = fileURLToPath(new URL("../scripts/validate-output.mjs", import.meta.url));
const postbuildCli = fileURLToPath(new URL("../scripts/postbuild.mjs", import.meta.url));
const docsAppDir = fileURLToPath(new URL("..", import.meta.url));
const repoRoot = path.resolve(docsAppDir, "../..");

// Source content: every English doc must have a zh-Hans page (translated or
// en-fallback); installation + automation-blueprints stay untranslated on
// purpose, and zh-only.mdx exists only in zh-Hans.
const EN_DOCS = [
  "index.mdx",
  "getting-started/quickstart.mdx",
  "getting-started/installation.mdx",
  "guides/automation-blueprints.mdx",
  "developer-guide/plugins.mdx",
];
const ZH_DOCS = ["index.mdx", "getting-started/quickstart.mdx", "developer-guide/plugins.mdx"];
const ZH_ONLY_DOCS = ["zh-only.mdx"];

const PLUGINS = [
  {
    name: "alpha", description: "Alpha", repo: "https://github.com/acme/alpha",
    sha: "0000000000000000000000000000000000000000", shaShort: "0000000", tier: "official",
    category: "memory", maintainer: "Ana", maintainerSlug: "ana", installCommand: "hermes plugins install alpha",
  },
  {
    name: "beta", description: "Beta", repo: "https://github.com/acme/beta",
    sha: "1111111111111111111111111111111111111111", shaShort: "1111111", tier: "community",
    category: "tools", maintainer: "Bo", maintainerSlug: "bo", installCommand: "hermes plugins install beta",
  },
];

function zhHref(route) {
  return "/zh-Hans" + (route === "/" ? "" : route);
}

function expectedRoutes() {
  const en = EN_DOCS.map((doc) => "/" + doc.replace(/\.mdx?$/, "").replace(/(?:^|\/)index$/, ""));
  const zh = [...en.map(zhHref), ...ZH_ONLY_DOCS.map((doc) => zhHref("/" + doc.replace(/\.mdx?$/, "")))];
  const hubs = ["/skills", "/plugins", zhHref("/skills"), zhHref("/plugins")];
  const catalog = PLUGINS.flatMap((plugin) => [
    `/plugins/${plugin.name}`, `/plugins/by/${plugin.maintainerSlug}`,
    zhHref(`/plugins/${plugin.name}`), zhHref(`/plugins/by/${plugin.maintainerSlug}`),
  ]);
  return [...en, ...zh, ...hubs, ...catalog];
}

const PAGE_HTML = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Page</title>
<link rel="stylesheet" href="/docs/_next/static/app.css">
<script src="/docs/_next/static/app.js"></script></head>
<body><img src="/docs/img/logo.png">
<img srcset="/docs/img/logo.png 1x, /docs/img/logo@2x.png 2x">
<img src="https://example.com/remote.png">
<script src="data:text/javascript,void(0)"></script></body></html>`;

/** A complete, valid bilingual export rooted at <tmp>/apps/docs. */
function buildExport(t) {
  const root = mkdtempSync(path.join(tmpdir(), "hermes-validate-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const appDir = path.join(root, "apps", "docs");
  const outDir = path.join(appDir, "out");
  mkdirSync(path.join(appDir, ".next"), { recursive: true });

  for (const doc of EN_DOCS) {
    const file = path.join(appDir, "content", "en", doc);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, `# ${doc}\n`);
  }
  for (const doc of [...ZH_DOCS, ...ZH_ONLY_DOCS]) {
    const file = path.join(appDir, "content", "zh-Hans", doc);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, `# ${doc}\n`);
  }

  const routes = expectedRoutes();
  for (const route of routes) {
    const file = path.join(outDir, route, "index.html");
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, PAGE_HTML);
    writeFileSync(path.join(path.dirname(file), "index.txt"), "RSC payload");
  }
  mkdirSync(path.join(outDir, "_next", "static"), { recursive: true });
  writeFileSync(path.join(outDir, "_next", "static", "app.css"), "body{}");
  writeFileSync(path.join(outDir, "_next", "static", "app.js"), "void 0");
  cpSync(path.join(outDir, "_next/static"), path.join(appDir, ".next/static"), { recursive: true });
  writeFileSync(path.join(outDir, "404.html"), "<!doctype html><title>404</title>");
  writeFileSync(path.join(outDir, ".nojekyll"), "");

  const publicDir = path.join(appDir, "public");
  const writePublic = (...segments) => {
    const file = path.join(publicDir, ...segments);
    mkdirSync(path.dirname(file), { recursive: true });
    return file;
  };
  writeFileSync(writePublic("img", "logo.png"), "PNG");
  writeFileSync(writePublic("img", "logo@2x.png"), "PNG2");
  writeFileSync(writePublic("search-config.json"), JSON.stringify({
    appId: "A1B2C3D4E5", apiKey: "0123456789abcdef0123456789abcdef", indexName: "hermes docs",
  }));
  writeFileSync(writePublic("oauth", "client-metadata.json"), JSON.stringify({
    client_id: "https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json",
    client_name: "Hermes Agent",
    application_type: "native",
    grant_types: ["authorization_code", "refresh_token"],
    response_types: ["code"],
    token_endpoint_auth_method: "none",
    redirect_uris: ["http://127.0.0.1:27890/callback", "http://localhost:27890/callback"],
  }));
  writeFileSync(writePublic("llms.txt"), "# Hermes Agent docs\n");
  writeFileSync(writePublic("llms-full.txt"), "# Hermes Agent docs (full)\n");
  writeFileSync(writePublic("api", "skills.json"), JSON.stringify([
    { name: "skill-one", category: "productivity", source: "built-in", overview: "One." },
    { name: "skill-two", category: "devops", source: "optional", overview: "Two." },
  ]));
  writeFileSync(writePublic("api", "skills-meta.json"), JSON.stringify({
    extractedAt: "2026-11-02T00:00:00Z", totalSkills: 2, localSkills: 2, externalSkills: 0,
    externalSource: "none", bySource: { "built-in": 1, optional: 1 },
  }));
  writeFileSync(writePublic("api", "plugins.json"), JSON.stringify(PLUGINS));
  writeFileSync(writePublic("api", "plugins-meta.json"), JSON.stringify({
    generatedAt: "2026-11-02T00:00:00Z", total: PLUGINS.length,
    byTier: { official: 1, verified: 0, community: 1 },
    byCategory: { memory: 1, tools: 1 },
    removedCount: 1, starsFetchedAt: null,
  }));
  writeFileSync(writePublic("api", "plugin-catalog.json"), JSON.stringify({
    generated_at: "2026-11-02T00:00:00Z",
    entries: PLUGINS.map(({ name, repo, sha }) => ({ name, repo, sha })),
    removed: [{ name: "retired-one" }],
  }));
  writeFileSync(writePublic("api", "automation-blueprints-index.json"), JSON.stringify([
    { key: "daily-digest", title: "Daily digest", description: "", category: "automation",
      tags: [], fields: [], source: "builtin", plugin: null, schedule: "0 9 * * *",
      scheduleHuman: "daily at 09:00", command: "/jobs add daily-digest", appUrl: "hermes://x" },
  ]));
  writeFileSync(writePublic("api", "model-catalog.json"), JSON.stringify({
    version: 1, updated_at: "2026-10-26T19:00:43Z", metadata: { source: "hermes-agent repo" },
    providers: { openrouter: { metadata: { display_name: "OpenRouter" }, models: [{ id: "org/model-one", description: "" }] } },
  }));

  cpSync(publicDir, outDir, { recursive: true });

  writeFileSync(path.join(appDir, ".next", "prerender-manifest.json"), JSON.stringify({
    version: 4,
    routes: Object.fromEntries([...routes, "/_not-found"].map((route) => [route, {}])),
    dynamicRoutes: {},
    notFoundRoutes: [],
  }));

  const post = spawnSync(process.execPath, [postbuildCli, "--app-dir", appDir], { encoding: "utf8" });
  assert.equal(post.status, 0, post.stderr);
  return { root, appDir, outDir, routes };
}

function runValidator(appDir, options = {}) {
  return spawnSync(process.execPath, [validatorCli, "--app-dir", appDir], {
    encoding: "utf8",
    cwd: options.cwd ?? repoRoot,
  });
}

/** Remove a route from the manifest AND its page files from out/, as if the
 *  build dropped it — without touching sibling or child routes. */
function dropRoute(appDir, route) {
  const manifestFile = path.join(appDir, ".next", "prerender-manifest.json");
  const manifest = JSON.parse(readFileSync(manifestFile, "utf8"));
  delete manifest.routes[route];
  writeFileSync(manifestFile, JSON.stringify(manifest));
  const pageDir = path.join(appDir, "out", ...route.split("/").filter(Boolean));
  rmSync(path.join(pageDir, "index.html"), { force: true });
  rmSync(path.join(pageDir, "index.txt"), { force: true });
}

/** Overwrite a generated file in BOTH public/ and out/ so the copy check stays quiet. */
function rewriteArtifact(appDir, relative, content) {
  for (const base of ["public", "out"]) {
    writeFileSync(path.join(appDir, base, relative), content);
  }
}

test("accepts a complete bilingual export via the CLI", (t) => {
  const { appDir } = buildExport(t);
  const result = runValidator(appDir);
  assert.equal(result.status, 0, result.stderr);
});

test("validateOutput(appDir) returns ok with no errors for a complete export", async (t) => {
  const { appDir } = buildExport(t);
  const result = validateOutput(appDir);
  assert.equal(result.ok, true);
  assert.deepEqual(result.errors, []);
});

test("the CLI default anchors to its own apps/docs, not the working directory", (t) => {
  const { root, appDir, routes } = buildExport(t);
  const scripts = path.join(appDir, "scripts");
  mkdirSync(scripts);
  cpSync(validatorCli, path.join(scripts, "validate-output.mjs"));
  cpSync(postbuildCli, path.join(scripts, "postbuild.mjs"));
  symlinkSync(path.join(docsAppDir, "src"), path.join(appDir, "src"), "dir");
  symlinkSync(path.join(docsAppDir, "node_modules"), path.join(appDir, "node_modules"), "dir");
  const result = spawnSync(process.execPath, [path.join(scripts, "validate-output.mjs")], {
    encoding: "utf8", cwd: root,
  });
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stdout.includes(`${routes.length} exported pages`), result.stdout);
});

test("rejects an exported page whose HTML is missing", (t) => {
  const { appDir } = buildExport(t);
  rmSync(path.join(appDir, "out", "getting-started", "quickstart", "index.html"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("Missing exported HTML for /getting-started/quickstart"), result.stderr);
});

test("rejects a page with an empty RSC payload", (t) => {
  const { appDir } = buildExport(t);
  writeFileSync(path.join(appDir, "out", "index.txt"), "");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/index.txt"), result.stderr);
});

test("rejects an export without 404.html", (t) => {
  const { appDir } = buildExport(t);
  rmSync(path.join(appDir, "out", "404.html"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("404.html"), result.stderr);
});

test("rejects an export without .nojekyll", (t) => {
  const { appDir } = buildExport(t);
  rmSync(path.join(appDir, "out", ".nojekyll"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes(".nojekyll"), result.stderr);
});

test("rejects an English doc without a zh-Hans fallback page", (t) => {
  const { appDir } = buildExport(t);
  dropRoute(appDir, "/zh-Hans/getting-started/installation");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("zh-Hans fallback"), result.stderr);
  assert.ok(result.stderr.includes("/zh-Hans/getting-started/installation"), result.stderr);
});

test("rejects a source document missing from the export", (t) => {
  const { appDir } = buildExport(t);
  writeFileSync(path.join(appDir, "content", "en", "getting-started", "new-page.mdx"), "# New page\n");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/getting-started/new-page"), result.stderr);
});

test("rejects a missing catalog hub page in either locale", (t) => {
  const { appDir } = buildExport(t);
  dropRoute(appDir, "/zh-Hans/plugins");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/zh-Hans/plugins"), result.stderr);
});

test("rejects a plugin entry without its per-plugin page", (t) => {
  const { appDir } = buildExport(t);
  dropRoute(appDir, "/plugins/alpha");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/plugins/alpha"), result.stderr);
});

test("rejects a catalog maintainer without its author page", (t) => {
  const { appDir } = buildExport(t);
  dropRoute(appDir, "/plugins/by/ana");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/plugins/by/ana"), result.stderr);
});

test("rejects a page referencing a missing local asset", (t) => {
  const { appDir } = buildExport(t);
  const page = path.join(appDir, "out", "index.html");
  writeFileSync(page, readFileSync(page, "utf8").replace('app.js', 'missing.js'));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("missing.js"), result.stderr);
});

test("accepts URL-encoded Next chunk paths", (t) => {
  const { appDir, outDir } = buildExport(t);
  const chunk = path.join(outDir, "_next/static/chunks/[[...slug]]/page.js");
  mkdirSync(path.dirname(chunk), { recursive: true });
  writeFileSync(chunk, "void 0");
  const page = path.join(outDir, "index.html");
  writeFileSync(page, readFileSync(page, "utf8").replace(
    "/docs/_next/static/app.js", "/docs/_next/static/chunks/%5B%5B...slug%5D%5D/page.js"));
  const result = runValidator(appDir);
  assert.equal(result.status, 0, result.stderr);
});

test("rejects a local asset outside the /docs base path", (t) => {
  const { appDir } = buildExport(t);
  const page = path.join(appDir, "out", "index.html");
  writeFileSync(page, readFileSync(page, "utf8").replace('src="/docs/img/logo.png"', 'src="/img/logo.png"'));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/docs base path"), result.stderr);
});

test("accepts data URLs and commas in srcset asset URLs", (t) => {
  const { appDir, outDir } = buildExport(t);
  writeFileSync(path.join(outDir, "img/logo,2.png"), "PNG2");
  const page = path.join(outDir, "index.html");
  writeFileSync(page, readFileSync(page, "utf8").replace(
    "/docs/img/logo.png 1x, /docs/img/logo@2x.png 2x",
    "data:image/png;base64,Zm9v 1x, /docs/img/logo,2.png 2x"));
  const result = runValidator(appDir);
  assert.equal(result.status, 0, result.stderr);
});

for (const [url, file] of [
  ["/docsite/app.js", "ite/app.js"],
  ["/docs/../outside.js", "../outside.js"],
  ["/docs/%2e%2e%2foutside.js", "../outside.js"],
]) {
  test(`rejects asset URLs outside the export boundary: ${url}`, (t) => {
    const { appDir, outDir } = buildExport(t);
    const target = path.join(outDir, file);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, "void 0");
    const page = path.join(outDir, "index.html");
    writeFileSync(page, readFileSync(page, "utf8").replace("/docs/_next/static/app.js", url));
    const result = runValidator(appDir);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /base path|export boundary/);
  });
}

test("rejects a missing absolute same-origin asset URL", (t) => {
  const { appDir, outDir } = buildExport(t);
  const page = path.join(outDir, "index.html");
  writeFileSync(page, readFileSync(page, "utf8").replace(
    "/docs/_next/static/app.js", "https://hermes-agent.nousresearch.com/docs/missing.js"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("missing.js"), result.stderr);
});

test("rejects directories masquerading as asset files", (t) => {
  const { appDir, outDir } = buildExport(t);
  const file = path.join(outDir, "_next/static/app.js");
  rmSync(file);
  mkdirSync(file);
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("app.js"), result.stderr);
});

test("rejects a public file that was not copied into the export", (t) => {
  const { appDir } = buildExport(t);
  rmSync(path.join(appDir, "out", "api", "model-catalog.json"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("api/model-catalog.json"), result.stderr);
});

test("rejects an export with no Next static assets at all", (t) => {
  const { appDir } = buildExport(t);
  rmSync(path.join(appDir, "out", "_next"), { recursive: true, force: true });
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("_next"), result.stderr);
});

test("rejects corrupt JSON in the export", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "skills.json"), "{oops");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("api/skills.json"), result.stderr);
  assert.ok(result.stderr.includes("valid JSON"), result.stderr);
});

test("rejects a missing generated catalog artifact", (t) => {
  const { appDir } = buildExport(t);
  rmSync(path.join(appDir, "out", "api", "plugins.json"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("api/plugins.json"), result.stderr);
  assert.ok(result.stderr.includes("extract-plugins.py"), result.stderr);
});

for (const [files, badValues] of [
  [["api/skills-meta.json", "api/plugins-meta.json", "api/plugin-catalog.json", "api/model-catalog.json", "oauth/client-metadata.json", "search-config.json"], [null, []]],
  [["api/skills.json", "api/plugins.json", "api/automation-blueprints-index.json"], [null, {}]],
]) {
  for (const file of files) {
    for (const value of badValues) {
      test(`rejects invalid JSON shape ${JSON.stringify(value)} in ${file}`, (t) => {
        const { appDir } = buildExport(t);
        rewriteArtifact(appDir, file, JSON.stringify(value));
        const result = runValidator(appDir);
        assert.notEqual(result.status, 0, file);
        assert.ok(result.stderr.includes(file), result.stderr);
      });
    }
  }
}

test("rejects malformed skills.json entries", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "skills.json"), JSON.stringify([{ category: "devops", source: "built-in" }]));
  rewriteArtifact(appDir, path.join("api", "skills-meta.json"), JSON.stringify({
    extractedAt: "2026-11-02T00:00:00Z", totalSkills: 1, localSkills: 1, externalSkills: 0,
    bySource: { "built-in": 1 },
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("skills.json"), result.stderr);
});

test("rejects skills-meta totals inconsistent with skills.json", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "skills-meta.json"), JSON.stringify({
    extractedAt: "2026-11-02T00:00:00Z", totalSkills: 5, localSkills: 2, externalSkills: 3,
    bySource: { "built-in": 1, optional: 1 },
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("totalSkills"), result.stderr);
});

test("rejects skills-meta source counts inconsistent with skills.json", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "skills-meta.json"), JSON.stringify({
    extractedAt: "2026-11-02T00:00:00Z", totalSkills: 2, localSkills: 2, externalSkills: 0,
    bySource: { "built-in": 2, optional: 0 },
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("bySource"), result.stderr);
});

test("validates the optional skills-index.json count when the file is present", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "skills-index.json"), JSON.stringify({
    version: 1, generated_at: "2026-11-02T00:00:00Z", skill_count: 7,
    skills: [{ id: "a" }, { id: "b" }, { id: "c" }],
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("skills-index.json"), result.stderr);
  assert.ok(result.stderr.includes("skill_count"), result.stderr);
});

test("rejects plugins-meta totals inconsistent with plugins.json", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "plugins-meta.json"), JSON.stringify({
    generatedAt: "2026-11-02T00:00:00Z", total: 9,
    byTier: { official: 1, verified: 0, community: 1 },
    byCategory: { memory: 1, tools: 1 }, removedCount: 1, starsFetchedAt: null,
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("plugins-meta.json"), result.stderr);
  assert.ok(result.stderr.includes("total"), result.stderr);
});

test("rejects plugins-meta tier counts inconsistent with plugins.json", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "plugins-meta.json"), JSON.stringify({
    generatedAt: "2026-11-02T00:00:00Z", total: 2,
    byTier: { official: 1, verified: 0, community: 0 },
    byCategory: { memory: 1, tools: 1 }, removedCount: 1, starsFetchedAt: null,
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("byTier"), result.stderr);
});

test("rejects plugins-meta removedCount inconsistent with plugin-catalog.json", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "plugins-meta.json"), JSON.stringify({
    generatedAt: "2026-11-02T00:00:00Z", total: 2,
    byTier: { official: 1, verified: 0, community: 1 },
    byCategory: { memory: 1, tools: 1 }, removedCount: 0, starsFetchedAt: null,
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("removedCount"), result.stderr);
});

test("rejects plugin-catalog.json entries missing a plugins.json entry", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "plugin-catalog.json"), JSON.stringify({
    generated_at: "2026-11-02T00:00:00Z",
    entries: [{ name: "alpha", repo: "https://github.com/acme/alpha", sha: "0000000000000000000000000000000000000000" }],
    removed: [{ name: "retired-one" }],
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("plugin-catalog.json"), result.stderr);
  assert.ok(result.stderr.includes("beta"), result.stderr);
});

test("rejects an empty plugins.json when the source catalog has valid entries", (t) => {
  const { appDir, root } = buildExport(t);
  mkdirSync(path.join(root, "plugin-catalog"), { recursive: true });
  writeFileSync(path.join(root, "plugin-catalog", "one.yaml"),
    `name: one\nrepo: https://github.com/acme/one\nsha: ${"2".repeat(40)}\nmaintainer: Cy\n`);
  rewriteArtifact(appDir, path.join("api", "plugins.json"), "[]");
  rewriteArtifact(appDir, path.join("api", "plugins-meta.json"), JSON.stringify({
    generatedAt: "2026-11-02T00:00:00Z", total: 0,
    byTier: { official: 0, verified: 0, community: 0 },
    removedCount: 0, starsFetchedAt: null,
  }));
  rewriteArtifact(appDir, path.join("api", "plugin-catalog.json"), JSON.stringify({
    generated_at: "2026-11-02T00:00:00Z", entries: [], removed: [],
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("extract-plugins.py"), result.stderr);
  assert.ok(result.stderr.includes("plugin-catalog"), result.stderr);
});

test("permits an empty UI catalog when source pins are malformed", (t) => {
  const { appDir, root } = buildExport(t);
  mkdirSync(path.join(root, "plugin-catalog"), { recursive: true });
  writeFileSync(path.join(root, "plugin-catalog/bad.yaml"),
    "name: bad\nrepo: https://github.com/acme/bad\nsha: not-a-commit\n");
  writeFileSync(path.join(root, "plugin-catalog/broken.yaml"), "name: [broken");
  rewriteArtifact(appDir, "api/plugins.json", "[]");
  rewriteArtifact(appDir, "api/plugins-meta.json", JSON.stringify({
    total: 0, byTier: { official: 0, verified: 0, community: 0 }, byCategory: {}, removedCount: 0,
  }));
  rewriteArtifact(appDir, "api/plugin-catalog.json", JSON.stringify({
    entries: [{ name: "bad", repo: "https://github.com/acme/bad", sha: "not-a-commit" }], removed: [],
  }));
  const result = runValidator(appDir);
  assert.equal(result.status, 0, result.stderr);
});

test("rejects an empty skills.json when local skill sources exist", (t) => {
  const { appDir, root } = buildExport(t);
  mkdirSync(path.join(root, "skills", "my-skill"), { recursive: true });
  writeFileSync(path.join(root, "skills", "my-skill", "SKILL.md"), "---\nname: my-skill\n---\n# My skill\n");
  rewriteArtifact(appDir, path.join("api", "skills.json"), "[]");
  rewriteArtifact(appDir, path.join("api", "skills-meta.json"), JSON.stringify({
    extractedAt: "2026-11-02T00:00:00Z", totalSkills: 0, localSkills: 0, externalSkills: 0,
    bySource: {},
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("extract-skills.py"), result.stderr);
});

test("rejects an oauth client_id that is not the published metadata URL", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("oauth", "client-metadata.json"), JSON.stringify({
    client_id: "https://example.com/client-metadata.json",
    redirect_uris: ["http://127.0.0.1:27890/callback"],
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("client_id"), result.stderr);
});

test("rejects a placeholder redirect URI in the OAuth metadata", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("oauth", "client-metadata.json"), JSON.stringify({
    client_id: "https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json",
    redirect_uris: ["https://example.com/callback"],
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("redirect_uris"), result.stderr);
});

test("rejects a malformed search-config.json without printing its key", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, "search-config.json", JSON.stringify({
    appId: "A1B2C3D4E5", apiKey: "", indexName: "hermes docs",
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("search-config.json"), result.stderr);
  assert.ok(result.stderr.includes("apiKey"), result.stderr);
  assert.ok(!result.stderr.includes("0123456789abcdef0123456789abcdef"), result.stderr);
});

test("rejects malformed blueprint catalog entries", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "automation-blueprints-index.json"),
    JSON.stringify([{ title: "No key" }]));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("automation-blueprints-index.json"), result.stderr);
});

test("rejects a gutted model catalog", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, path.join("api", "model-catalog.json"), JSON.stringify({
    version: 1, providers: {},
  }));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("model-catalog.json"), result.stderr);
});

test("rejects an empty llms.txt in the export", (t) => {
  const { appDir } = buildExport(t);
  rewriteArtifact(appDir, "llms.txt", "");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("llms.txt"), result.stderr);
});

test("rejects a legacy alias that no longer matches the redirect recipe", (t) => {
  const { appDir } = buildExport(t);
  const alias = path.join(appDir, "out", "quickstart", "index.html");
  writeFileSync(alias, readFileSync(alias, "utf8").replaceAll(
    "/docs/getting-started/quickstart", "/docs/getting-started/wrong"));
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("/quickstart"), result.stderr);
});

test("rejects a redirect script that no longer matches its producer", (t) => {
  const { appDir, outDir } = buildExport(t);
  const alias = readFileSync(path.join(outDir, "quickstart/index.html"), "utf8");
  const src = alias.match(/<script[^>]+src="([^"]+)"/)[1];
  writeFileSync(path.join(outDir, src.slice("/docs/".length)), "void 0");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /redirect.*(recipe|producer)/);
});

test("rejects a dropped lazy Next asset that no initial HTML references", (t) => {
  const { appDir } = buildExport(t);
  writeFileSync(path.join(appDir, ".next/static/lazy.js"), "void 0");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("lazy.js"), result.stderr);
});

test("rejects a sitemap that does not match the exported routes", (t) => {
  const { appDir } = buildExport(t);
  const sitemap = path.join(appDir, "out", "sitemap.xml");
  writeFileSync(sitemap, readFileSync(sitemap, "utf8") + "<!-- stale -->\n");
  const result = runValidator(appDir);
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes("sitemap.xml"), result.stderr);
});
