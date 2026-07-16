#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { shopifyGraphql, loadShopifyConfig, connectionStatus } from "./lib/shopify-dev-dashboard-auth.mjs";

const DEFAULT_ENV = "skill-hub.env";
const REQUIRED_READ_SCOPES = ["read_content", "read_online_store_pages"];
const REQUIRED_WRITE_SCOPES = ["write_content", "write_online_store_pages"];
const ARTICLE_FIELDS = `id title handle body summary isPublished publishedAt updatedAt author { name } blog { id title handle } image { altText url }`;

function usage() {
  console.log(`Usage:
  node shopify-blog-seo-admin.mjs onboarding
  node shopify-blog-seo-admin.mjs init-env --method shopify_cli_oauth --env skill-hub.env
  node shopify-blog-seo-admin.mjs connection-check --env skill-hub.env
  node shopify-blog-seo-admin.mjs find --env skill-hub.env --url <article-url>
  node shopify-blog-seo-admin.mjs find --env skill-hub.env --title "<article title>"
  node shopify-blog-seo-admin.mjs audit --env skill-hub.env --article-id <article-id> --output audit.json
  node shopify-blog-seo-admin.mjs report --audit audit.json --candidate candidate.json --output report.html
  node shopify-blog-seo-admin.mjs report --input audit-plan.json --output report.html
  node shopify-blog-seo-admin.mjs apply --env skill-hub.env --input approved-plan.json [--execute]
  node shopify-blog-seo-admin.mjs verify --env skill-hub.env --article-id <article-id>`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value.startsWith("--")) { args._.push(value); continue; }
    const key = value.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) args[key] = true;
    else { args[key] = next; index += 1; }
  }
  return args;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

function safeReportUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "#";
  } catch {
    return "#";
  }
}

function stripHtml(value) {
  return String(value || "").replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/<style[\s\S]*?<\/style>/gi, " ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

function safeArticleHtml(value) {
  return String(value || "")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<(iframe|object|embed|form|input|button|textarea|select)\b[\s\S]*?>[\s\S]*?<\/\1>/gi, "")
    .replace(/\son[a-z]+\s*=\s*(["'])[^"']*\1/gi, "")
    .replace(/javascript:/gi, "")
    .replace(/<(html|head|body|main)\b[^>]*>|<\/(html|head|body|main)>/gi, "")
    .replace(/\s(href|src)\s*=\s*(["'])\s*(?!https?:|\/|#|mailto:)[^"']*\2/gi, " $1=\"#\"");
}

function previewFaqHtml(value) {
  const html = safeArticleHtml(value);
  if (!/<section\b[^>]*\bid=["']faq["']/i.test(html) || /<details\b/i.test(html)) return html;
  return html.replace(/<section([^>]*\bid=["']faq["'][^>]*)>([\s\S]*?)<\/section>/i, (match, attrs, inner) => {
    const heading = inner.match(/^\s*(<h2\b[\s\S]*?<\/h2>)/i)?.[1] || "<h2>Frequently asked questions</h2>";
    const content = inner.replace(/^\s*<h2\b[\s\S]*?<\/h2>/i, "");
    const items = [];
    const pattern = /<h3\b[^>]*>([\s\S]*?)<\/h3>([\s\S]*?)(?=<h3\b|$)/gi;
    let matchItem;
    while ((matchItem = pattern.exec(content))) {
      const question = matchItem[1].trim();
      const answer = matchItem[2].trim();
      if (question && answer) items.push(`<details><summary>${question}</summary>${answer}</details>`);
    }
    return items.length ? `<section${attrs}>${heading}${items.join("")}</section>` : match;
  });
}

function normalizeTitle(value) { return stripHtml(value).replace(/[“”‘’]/g, "'").toLowerCase().replace(/\s+/g, " ").trim(); }

function parseArticleUrl(rawUrl) {
  const url = new URL(rawUrl);
  const parts = url.pathname.split("/").filter(Boolean);
  const blogIndex = parts.indexOf("blogs");
  if (blogIndex >= 0 && parts[blogIndex + 2]) return { url: url.href, blogHandle: parts[blogIndex + 1], articleHandle: parts[blogIndex + 2] };
  return { url: url.href };
}

function articleUrl(article, fallbackDomain) {
  if (!article?.blog?.handle || !article?.handle) return null;
  return `https://${fallbackDomain}/blogs/${article.blog.handle}/${article.handle}`;
}

const ARTICLE_QUERY = `query FindArticles($query: String, $first: Int!) { articles(first: $first, query: $query, sortKey: UPDATED_AT, reverse: true) { nodes { ${ARTICLE_FIELDS} } } }`;
const ARTICLE_QUERY_BY_ID = `query ArticleById($id: ID!) { article(id: $id) { ${ARTICLE_FIELDS} } }`;
const SHOP_QUERY = `query ShopConnection { shop { name myshopifyDomain } }`;
const CONNECTION_CHECK_QUERY = `query ConnectionCheck { shop { name myshopifyDomain } articles(first: 1) { nodes { id } } }`;
const ARTICLE_UPDATE = `mutation UpdateArticle($id: ID!, $article: ArticleUpdateInput!) { articleUpdate(id: $id, article: $article) { article { id title handle body summary updatedAt } userErrors { field message code } } }`;

async function readJson(input) {
  const text = input === "-" ? await new Promise((resolve, reject) => { let body = ""; process.stdin.setEncoding("utf8"); process.stdin.on("data", (chunk) => { body += chunk; }); process.stdin.on("end", () => resolve(body)); process.stdin.on("error", reject); }) : await fs.readFile(input, "utf8");
  return JSON.parse(text);
}

async function writeJson(output, value) { await fs.writeFile(output, JSON.stringify(value, null, 2), "utf8"); }

function extractMeta(html, pattern) { return String(html).match(pattern)?.[1]?.trim() || null; }

async function publicUrlSignals(rawUrl) {
  try {
    const response = await fetch(rawUrl, { redirect: "follow" });
    const html = await response.text();
    const passwordPage = /\/password(?:["'/?#]|$)/i.test(response.url);
    const accessible = response.ok && !passwordPage && !/form[^>]+action=["'][^"']*password/i.test(html);
    return {
      status: response.status,
      finalUrl: response.url,
      canonical: extractMeta(html, /<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']+)/i),
      title: extractMeta(html, /<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)/i) || extractMeta(html, /<title[^>]*>([\s\S]*?)<\/title>/i),
      passwordPage,
      accessible,
      accessState: accessible ? "real-storefront" : (passwordPage ? "password-protected" : (response.ok ? "blocked-or-non-article" : "unavailable")),
    };
  } catch (error) {
    return { status: null, accessible: false, accessState: "unavailable", error: error.message };
  }
}

async function storefrontPreviewState(article) {
  const storefrontUrl = article.storefrontUrl || null;
  if (!storefrontUrl) return { mode: "theme-like-fallback", accessState: "unavailable", storefrontUrl: null, note: "No storefront URL could be derived." };
  const signals = await publicUrlSignals(storefrontUrl);
  if (signals.accessible) {
    return {
      mode: "real-storefront-reference",
      accessState: signals.accessState,
      storefrontUrl,
      finalUrl: signals.finalUrl,
      status: signals.status,
      note: "The storefront page was reachable. The report renders the candidate in a local responsive shell and records the live page as the reference; it does not claim a pixel-perfect theme clone.",
      signals,
    };
  }
  return {
    mode: "theme-like-fallback",
    accessState: signals.accessState,
    storefrontUrl,
    finalUrl: signals.finalUrl || storefrontUrl,
    status: signals.status,
    note: signals.passwordPage ? "The storefront is password protected; the real frontend was not verified." : "The storefront could not be safely accessed; the preview is an explicitly labelled fallback.",
    signals,
  };
}

async function findArticles(env, searchQuery) {
  const data = await shopifyGraphql(env, ARTICLE_QUERY, { query: searchQuery || null, first: 50 }, { requiredAnyScopes: REQUIRED_READ_SCOPES });
  return data.articles.nodes.map((article) => ({ ...article, storefrontUrl: articleUrl(article, env.SHOPIFY_STORE_DOMAIN) }));
}

async function resolveArticle(env, args) {
  if (args["article-id"]) {
    const data = await shopifyGraphql(env, ARTICLE_QUERY_BY_ID, { id: args["article-id"] }, { requiredAnyScopes: REQUIRED_READ_SCOPES });
    if (!data.article) throw new Error("ARTICLE_NOT_FOUND: the supplied Article ID returned no Article.");
    return { article: { ...data.article, storefrontUrl: articleUrl(data.article, env.SHOPIFY_STORE_DOMAIN) }, matchedBy: "article-id" };
  }
  if (args.url) {
    const parsed = parseArticleUrl(args.url);
    let candidates = parsed.articleHandle ? await findArticles(env, `handle:${parsed.articleHandle}`) : [];
    if (parsed.blogHandle) candidates = candidates.filter((article) => article.blog?.handle === parsed.blogHandle);
    if (!candidates.length) {
      const signals = await publicUrlSignals(args.url);
      if (signals.canonical && signals.canonical !== args.url) return resolveArticle(env, { url: signals.canonical });
      if (signals.title) candidates = await findArticles(env, `title:${signals.title.replace(/[":]/g, " ")}`);
    }
    if (candidates.length === 1) return { article: candidates[0], matchedBy: "url", publicUrlSignals: await publicUrlSignals(args.url) };
    if (!candidates.length) throw new Error("ARTICLE_NOT_FOUND: no Article matched the supplied URL.");
    return { candidates, matchedBy: "url-ambiguous" };
  }
  if (args.title) {
    const candidates = await findArticles(env, `title:${String(args.title).replace(/[":]/g, " ")}`);
    const exact = candidates.filter((article) => normalizeTitle(article.title) === normalizeTitle(args.title));
    if (exact.length === 1) return { article: exact[0], matchedBy: "exact-title" };
    if (exact.length > 1 || candidates.length > 1) return { candidates: exact.length ? exact : candidates, matchedBy: "title-ambiguous" };
    if (candidates.length === 1) return { article: candidates[0], matchedBy: "close-title" };
  }
  throw new Error("TARGET_REQUIRED: provide --url, --title, or --article-id.");
}

function auditHtml(article) {
  const body = String(article.body || "");
  const headings = [...body.matchAll(/<h([1-6])\b([^>]*)>([\s\S]*?)<\/h\1>/gi)].map((match) => ({ level: Number(match[1]), text: stripHtml(match[3]), attrs: match[2] }));
  const ids = [...body.matchAll(/\bid=["']([^"']+)["']/gi)].map((match) => match[1]);
  const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
  const images = [...body.matchAll(/<img\b([^>]*)>/gi)].map((match) => ({ alt: match[1].match(/\balt=["']([^"']*)["']/i)?.[1] ?? null, src: match[1].match(/\bsrc=["']([^"']*)["']/i)?.[1] ?? null }));
  const links = [...body.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)].map((match) => ({ href: match[1].match(/\bhref=["']([^"']*)["']/i)?.[1] ?? null, text: stripHtml(match[2]) }));
  const tocRefs = [...body.matchAll(/href=["']#([^"']+)["']/gi)].map((match) => match[1]);
  const linkableIds = new Set(ids);
  let storeHost = null;
  try { storeHost = new URL(article.storefrontUrl || "").host; } catch { /* storefront URL is optional */ }
  const headingIssues = [];
  for (let index = 1; index < headings.length; index += 1) if (headings[index].level - headings[index - 1].level > 1) headingIssues.push(`H${headings[index - 1].level} → H${headings[index].level} skips a level`);
  return {
    content: {
      wordCount: stripHtml(body).split(/\s+/).filter(Boolean).length,
      summaryMissing: !stripHtml(article.summary),
      headingCount: headings.length,
      headings: headings.map(({ level, text }) => ({ level, text })),
      spellingGrammar: "requires AI review",
      duplicateContent: "requires AI comparison",
      informationGaps: "requires intent and research review",
    },
    structure: { headingIssues, duplicateIds, emptyParagraphs: (body.match(/<p>\s*(?:&nbsp;|\u00a0)?\s*<\/p>/gi) || []).length, tocPresent: /<nav\b[^>]*id=["']article-toc["']/i.test(body), unresolvedTocRefs: tocRefs.filter((id) => !linkableIds.has(id)) },
    accessibility: { images, missingAltCount: images.filter((image) => image.alt === null || image.alt.trim() === "").length, scriptTagCount: (body.match(/<script\b/gi) || []).length, unsafeTagCount: (body.match(/<(iframe|object|embed|form)\b/gi) || []).length },
    links: {
      count: links.length,
      links,
      emptyTextCount: links.filter((link) => !link.text).length,
      externalCount: links.filter((link) => {
        try { return /^https?:\/\//i.test(link.href || "") && new URL(link.href).host !== storeHost; } catch { return false; }
      }).length,
      internalCount: links.filter((link) => {
        if (/^(#|\/)/.test(link.href || "")) return true;
        try { return Boolean(storeHost) && new URL(link.href).host === storeHost; } catch { return false; }
      }).length,
      connectivityChecks: { status: "not-run", results: [] },
    },
    eeat: { status: "requires-ai-research", methodology: "references/eeat-methodology.md" },
  };
}

async function checkLinkConnectivity(links) {
  const candidates = [...new Set(links.map((link) => link.href).filter((href) => /^https?:\/\//i.test(href || "")))].slice(0, 30);
  const results = await Promise.all(candidates.map(async (url) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      let response = await fetch(url, { method: "HEAD", redirect: "follow", signal: controller.signal });
      if ([403, 405, 501].includes(response.status)) response = await fetch(url, { method: "GET", redirect: "follow", signal: controller.signal });
      return { url, ok: response.ok, status: response.status, finalUrl: response.url };
    } catch (error) {
      return { url, ok: false, status: null, error: error.name === "AbortError" ? "timeout" : error.message };
    } finally {
      clearTimeout(timeout);
    }
  }));
  return { status: "complete", checkedCount: candidates.length, skippedCount: Math.max(0, links.filter((link) => /^https?:\/\//i.test(link.href || "")).length - candidates.length), results };
}

function renderFind(result) {
  if (result.article) return { ...result.article, body: undefined };
  return { matchedBy: result.matchedBy, candidates: result.candidates.map((candidate) => ({ ...candidate, body: undefined })) };
}

function issueRows(audit) {
  const rows = [];
  if (audit.content.summaryMissing) rows.push(["P1", "Summary", "The Article summary is empty.", "Draft a concise summary after intent review."]);
  if (audit.structure.headingIssues.length) rows.push(["P1", "Heading structure", audit.structure.headingIssues.join("; "), "Repair the hierarchy without changing meaning."]);
  if (audit.accessibility.missingAltCount) rows.push(["P1", "Image accessibility", `${audit.accessibility.missingAltCount} image(s) have missing or empty alt text.`, "Create context-accurate alt text; ask for visual confirmation when context is insufficient."]);
  if (audit.links.links.some((link) => !link.href)) rows.push(["P1", "Links", "At least one link has no usable href.", "Repair or remove the link after review."]);
  if (audit.links.emptyTextCount) rows.push(["P1", "Link accessibility", `${audit.links.emptyTextCount} link(s) have no visible link text.`, "Use descriptive anchor text or remove the empty link."]);
  const brokenLinks = audit.links.connectivityChecks?.results?.filter((link) => !link.ok) || [];
  if (brokenLinks.length) rows.push(["P1", "Link connectivity", `${brokenLinks.length} link(s) did not return a successful response.`, "Review the destination and replace or remove it only after confirming the correct URL."]);
  if (audit.structure.duplicateIds.length) rows.push(["P0", "HTML anchors", `${audit.structure.duplicateIds.length} duplicate id(s) were found.`, "Regenerate unique IDs before creating a clickable TOC."]);
  if (audit.structure.emptyParagraphs) rows.push(["P2", "HTML cleanup", `${audit.structure.emptyParagraphs} empty paragraph(s) add unnecessary spacing.`, "Remove empty paragraphs while preserving intentional layout."]);
  if (audit.structure.unresolvedTocRefs.length) rows.push(["P0", "TOC", "Some TOC anchors do not resolve.", "Regenerate unique heading IDs and verify every anchor."]);
  if (audit.accessibility.scriptTagCount || audit.accessibility.unsafeTagCount) rows.push(["P0", "HTML safety", "The body contains script or unsafe embedded tags.", "Remove them from Article body and handle structured data in the correct theme/app surface."]);
  return rows;
}

function renderReport(plan) {
  const article = plan.article || {};
  const audit = plan.audit || auditHtml(article);
  const eeat = plan.eeat || { score: null, findings: [], blockingIssues: ["AI deep research has not been attached yet."] };
  const candidate = plan.candidate || {};
  const preview = plan.preview || {};
  const body = previewFaqHtml(candidate.body || article.body || "");
  const rows = issueRows(audit);
  const changes = (candidate.changes || []).map((change) => `<li><strong>${escapeHtml(change.area || "Change")}</strong>: ${escapeHtml(change.summary || change)}</li>`).join("") || "<li>No candidate changes were supplied.</li>";
  const research = (plan.research || []).map((item) => `<li><a href="${escapeHtml(safeReportUrl(item.url))}">${escapeHtml(item.title || item.publisher || item.url || "Source")}</a> — ${escapeHtml(item.supports || item.note || "")}</li>`).join("") || "<li>No research sources attached.</li>";
  const findings = rows.map(([priority, area, problem, recommendation]) => `<tr><td>${priority}</td><td>${escapeHtml(area)}</td><td>${escapeHtml(problem)}</td><td>${escapeHtml(recommendation)}</td></tr>`).join("") || `<tr><td colspan="4">No deterministic blocking findings.</td></tr>`;
  const eeatWarning = eeat.blockingIssues?.length ? `<p class="warn">${escapeHtml(eeat.blockingIssues.join(" "))}</p>` : "";
  const imageMarkup = article.image?.url ? `<img src="${escapeHtml(safeReportUrl(article.image.url))}" alt="${escapeHtml(article.image.altText || "")}">` : "";
  const summaryMarkup = candidate.summary || article.summary ? `<p><strong>${escapeHtml(candidate.summary || article.summary)}</strong></p>` : "";
  const previewNote = preview.note || "The report uses a responsive Shopify-style shell. The access state above determines whether the real storefront was verified.";
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shopify Blog SEO Audit — ${escapeHtml(article.title || "Article")}</title><style>
body{margin:0;background:#f4f5f7;color:#1f2933;font:15px/1.6 Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.wrap{max-width:1180px;margin:auto;padding:28px}.badge{display:inline-block;background:#fff3cd;color:#7a4b00;border:1px solid #e7c66b;border-radius:999px;padding:5px 12px;font-size:12px;font-weight:700}.hero,.panel,.storefront{background:white;border:1px solid #dfe3e8;border-radius:16px;box-shadow:0 8px 30px #1f29330d}.hero{padding:28px;margin-bottom:18px}.hero h1{font-size:32px;line-height:1.2;margin:10px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{padding:22px;margin:18px 0}.panel h2{margin-top:0}.metric{display:inline-block;border:1px solid #dfe3e8;border-radius:12px;padding:10px 14px;margin:4px 6px 4px 0;background:#fafbfc}.metric strong{display:block;font-size:22px}.warn{color:#8a4b00;background:#fff8e6;border-left:4px solid #e7a900;padding:10px 14px}.bad{color:#9b1c1c;background:#fff0f0;border-left:4px solid #d14343;padding:10px 14px}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e5e7eb;padding:10px}th{background:#f8fafc}.storefront{overflow:hidden}.store-head{padding:16px 24px;border-bottom:1px solid #e5e7eb;color:#667085}.article{max-width:760px;margin:auto;padding:34px 24px 60px}.article h1{font-size:40px;line-height:1.15}.article h2{margin-top:36px}.article img{max-width:100%;height:auto;border-radius:10px}.article nav{border:1px solid #dfe3e8;background:#f8fafc;padding:14px 18px;border-radius:10px}.article details{border-top:1px solid #e5e7eb;padding:12px 0}.article summary{cursor:pointer;font-weight:700}.article a{color:#1769aa}.article blockquote{border-left:4px solid #cbd5e1;padding-left:16px;color:#52606d}@media(max-width:760px){.wrap{padding:14px}.grid{grid-template-columns:1fr}.hero h1{font-size:26px}.article h1{font-size:30px}}
  </style></head><body><main class="wrap"><section class="hero"><span class="badge">Preview only — not published</span><h1>${escapeHtml(article.title || "Untitled Article")}</h1><p>${escapeHtml(article.storefrontUrl || preview.storefrontUrl || "Storefront URL not available")}</p><div class="metric"><small>Audit status</small><strong>${escapeHtml(plan.status || "Draft")}</strong></div><div class="metric"><small>E-E-A-T</small><strong>${escapeHtml(eeat.score ?? "Research")}</strong></div><div class="metric"><small>Preview mode</small><strong>${escapeHtml(preview.mode || "theme-like-fallback")}</strong></div><div class="metric"><small>Frontend access</small><strong>${escapeHtml(preview.accessState || "unknown")}</strong></div><p class="warn">${escapeHtml(previewNote)}</p></section><div class="grid"><section class="panel"><h2>What changed</h2><ul>${changes}</ul></section><section class="panel"><h2>Research and E-E-A-T</h2><ul>${research}</ul>${eeatWarning}</section></div><section class="panel"><h2>Audit findings</h2><table><thead><tr><th>Priority</th><th>Area</th><th>Problem</th><th>Recommendation</th></tr></thead><tbody>${findings}</tbody></table></section><section class="storefront"><div class="store-head">Shopify storefront preview · ${escapeHtml(preview.mode || "theme-like-fallback")}</div><article class="article"><h1>${escapeHtml(article.title || "Untitled Article")}</h1>${imageMarkup}${summaryMarkup}${body}</article></section><section class="panel"><h2>Approval bundle</h2><p>Proposed fields: ${escapeHtml(Object.keys(candidate.updates || { body: true }).join(", ") || "body")}. Fields not listed remain unchanged.</p><p class="warn">This report is a review artifact. The skill must receive explicit approval before applying changes.</p></section></main></body></html>`;
}

async function initEnv(args) {
  const output = args.env || DEFAULT_ENV;
  const method = args.method || "shopify_cli_oauth";
  if (!["dev_dashboard_client_credentials", "shopify_cli_oauth"].includes(method)) throw new Error("INVALID_ACCESS_METHOD: use dev_dashboard_client_credentials or shopify_cli_oauth.");
  const text = `SKILL_HUB_SHOPIFY_ACCESS_METHOD=${method}\nSKILL_HUB_SHOPIFY_STORE_DOMAIN=\n# Direct Dev Dashboard mode only:\n# SKILL_HUB_SHOPIFY_CLIENT_ID=\n# SKILL_HUB_SHOPIFY_CLIENT_SECRET=\n# Optional: only for approved Dev Dashboard permission releases; never a store API credential.\n# SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN=\n`;
  await fs.writeFile(output, text, { encoding: "utf8", flag: "wx" }).catch(() => {});
  console.log(JSON.stringify({
    ok: true,
    env: path.resolve(output),
    method,
    requiredScopes: "read_content|read_online_store_pages,write_content|write_online_store_pages",
    nextSteps: [
      "Fill SKILL_HUB_SHOPIFY_STORE_DOMAIN with the target .myshopify.com domain.",
      method === "dev_dashboard_client_credentials" ? "Fill the private Dev Dashboard Client ID and Client Secret." : "Run shopify store auth when the connection check asks for it.",
      `Run: node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs connection-check --env ${output}`,
    ],
  }, null, 2));
}

function printOnboarding() {
  console.log(JSON.stringify({
    install: "npx skills add lvsao/shopify-skill-hub --skill shopify-blog-seo-optimizer",
    setup: `node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs init-env --method shopify_cli_oauth --env ${DEFAULT_ENV}`,
    config: [
      "Set SKILL_HUB_SHOPIFY_STORE_DOMAIN in the private working-directory skill-hub.env.",
      "Quick mode: run Shopify browser authorization with read_content,write_content.",
      "Direct mode: set SKILL_HUB_SHOPIFY_CLIENT_ID and SKILL_HUB_SHOPIFY_CLIENT_SECRET for the merchant's installed Dev App.",
      "Optional direct-mode automation: set SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN privately for approved permission releases only.",
    ],
    workflow: [
      "connection-check",
      "find by URL, exact title, or Article ID",
      "audit and research",
      "AI creates candidate.json from the audit",
      "report --audit audit.json --candidate candidate.json",
      "ask the merchant for explicit approval",
      "apply --input approved-plan.json --execute",
      "verify",
    ],
    safety: "No access token or client secret is written to reports; write execution requires an explicit approved: true plan.",
  }, null, 2));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args._[0];
  if (!command || command === "help" || command === "--help") { usage(); return; }
  if (command === "onboarding") { printOnboarding(); return; }
  if (command === "init-env") return initEnv(args);
  if (command === "report") {
    let plan;
    if (args.audit || args.candidate) {
      const auditPlan = args.audit ? await readJson(args.audit) : {};
      const candidatePlan = args.candidate ? await readJson(args.candidate) : {};
      plan = {
        ...auditPlan,
        ...candidatePlan,
        article: candidatePlan.article || auditPlan.article,
        audit: auditPlan.audit || candidatePlan.audit,
        preview: auditPlan.preview || candidatePlan.preview,
      };
    } else plan = await readJson(args.input || "-");
    const output = args.output || `shopify-blog-seo-report-${Date.now()}.html`;
    await fs.writeFile(output, renderReport(plan), "utf8");
    console.log(JSON.stringify({ ok: true, output: path.resolve(output), previewMode: plan.preview?.mode || "theme-like-fallback", approvalRequired: true }, null, 2));
    return;
  }
  const env = await loadShopifyConfig(args.env || DEFAULT_ENV);
  if (command === "connection-check") {
    const connection = await connectionStatus(env);
    try {
      const data = await shopifyGraphql(env, CONNECTION_CHECK_QUERY, {}, { requiredAnyScopes: REQUIRED_READ_SCOPES });
      const writeGranted = connection.scopeStatus?.write?.granted;
      console.log(JSON.stringify({ ok: true, connection, shop: data.shop, canReadArticles: true, hasWriteScope: writeGranted ? writeGranted.length > 0 : "unknown", nextStep: "Provide an article URL, exact title, or Article ID to start the audit." }, null, 2));
    } catch (error) {
      console.log(JSON.stringify({ ok: false, connection, canReadArticles: false, error: error.message, nextStep: "Grant the missing read scope family, approve the app permission update, then rerun connection-check." }, null, 2));
      process.exitCode = 1;
    }
    return;
  }
  if (command === "find") { const result = await resolveArticle(env, args); console.log(JSON.stringify(renderFind(result), null, 2)); return; }
  if (command === "audit") {
    const result = await resolveArticle(env, args);
    if (!result.article) { console.log(JSON.stringify(renderFind(result), null, 2)); process.exitCode = 2; return; }
    const audit = auditHtml(result.article);
    audit.links.connectivityChecks = await checkLinkConnectivity(audit.links.links);
    const output = args.output || "-";
    const preview = await storefrontPreviewState(result.article);
    const value = { article: { ...result.article, body: result.article.body }, audit, preview, status: "audit-complete", next: "The AI agent should create candidate.json from this audit, then render the combined report before asking for approval." };
    if (output === "-") console.log(JSON.stringify(value, null, 2)); else { await writeJson(output, value); console.log(JSON.stringify({ ok: true, output: path.resolve(output), articleId: result.article.id }, null, 2)); }
    return;
  }
  if (command === "apply") {
    const plan = await readJson(args.input || "-");
    if (!plan.articleId && !plan.article?.id) throw new Error("APPROVAL_PLAN_INVALID: articleId is required.");
    const article = {};
    const updates = plan.updates || plan.candidate?.updates || {};
    for (const key of ["body", "summary"]) if (updates[key] !== undefined) article[key] = key === "body" ? safeArticleHtml(updates[key]) : String(updates[key]);
    if (!Object.keys(article).length) throw new Error("APPROVAL_PLAN_INVALID: updates.body or updates.summary is required.");
    const preview = { previewOnly: !args.execute, articleId: plan.articleId || plan.article.id, fields: Object.keys(article), bodyLength: article.body?.length || null };
    if (!args.execute) { console.log(JSON.stringify(preview, null, 2)); return; }
    if (plan.approved !== true && plan.approval?.confirmed !== true) throw new Error("APPROVAL_REQUIRED: set approved: true only after the merchant explicitly confirms the combined report.");
    const data = await shopifyGraphql(env, ARTICLE_UPDATE, { id: plan.articleId || plan.article.id, article }, { mutation: true, requiredAnyScopes: REQUIRED_WRITE_SCOPES });
    const result = data.articleUpdate;
    if (result.userErrors?.length) throw new Error(`ARTICLE_UPDATE_FAILED: ${JSON.stringify(result.userErrors)}`);
    console.log(JSON.stringify({ ...preview, previewOnly: false, updated: result.article }, null, 2));
    return;
  }
  if (command === "verify") {
    const id = args["article-id"];
    if (!id) throw new Error("ARTICLE_ID_REQUIRED: provide --article-id.");
    const data = await shopifyGraphql(env, ARTICLE_QUERY_BY_ID, { id }, { requiredAnyScopes: REQUIRED_READ_SCOPES });
    if (!data.article) throw new Error("ARTICLE_NOT_FOUND: no Article returned.");
    console.log(JSON.stringify({ article: { ...data.article, body: undefined }, audit: auditHtml(data.article), preview: await storefrontPreviewState({ ...data.article, storefrontUrl: articleUrl(data.article, env.SHOPIFY_STORE_DOMAIN) } ) }, null, 2));
    return;
  }
  usage();
  process.exitCode = 1;
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
