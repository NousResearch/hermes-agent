#!/usr/bin/env node
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { assertRequiredScopes, devDashboardGraphql, isDevDashboardMode, mergeRuntimeEnv } from "./lib/shopify-dev-dashboard-auth.mjs";
import { fetchPublic } from "./lib/public-fetch.mjs";

const DEFAULT_ENV = "skill-hub.env";
const REQUIRED_SCOPES = "read_products,write_products,read_content,write_content,read_files,write_files";
const SOFT_ALT_LIMIT = 125;
const HARD_ALT_LIMIT = 512;
const execFileAsync = promisify(execFile);

function fail(message) {
  console.error(`ERROR: ${message}`);
  process.exit(1);
}

function printUsage() {
  console.log(`Usage: node shopify-alt-text-admin.mjs <command> [options]

Commands:
  init-env          --env skill-hub.env
  connection-check  --env skill-hub.env
  target            --env skill-hub.env --product|--collection|--article|--media-id|--url <value> [--download] [--limit 3]
  scan              --env skill-hub.env [--page-size 50]
  vision-sample     --env skill-hub.env [--limit 3]
  apply             --env skill-hub.env --input <file|->
`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i];
    if (!value.startsWith("--")) {
      args._.push(value);
      continue;
    }
    const key = value.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function parseEnv(text) {
  const env = {};
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    env[key] = value;
  }
  return env;
}

async function loadEnv(file) {
  const text = await fs.readFile(file, "utf8").catch(() => null);
  const env = mergeRuntimeEnv(text ? parseEnv(text) : {});
  if (!text && !env.SKILL_HUB_SHOPIFY_STORE_DOMAIN) {
    fail(`Missing env file: ${file}. Current working directory: ${process.cwd()}. The env must be in the user's working directory, not the installed skill directory. Run from the folder that contains skill-hub.env, or pass an absolute path such as --env "<USER_WORKDIR>\\skill-hub.env".`);
  }
  if (!["shopify_cli_oauth", "dev_dashboard_client_credentials"].includes(env.SKILL_HUB_SHOPIFY_ACCESS_METHOD || "shopify_cli_oauth")) {
    fail("Unsupported SKILL_HUB_SHOPIFY_ACCESS_METHOD. Use shopify_cli_oauth or dev_dashboard_client_credentials.");
  }
  return env;
}

function normalizeDomain(value) {
  if (!value) fail("SKILL_HUB_SHOPIFY_STORE_DOMAIN is required.");
  return value.replace(/^https?:\/\//i, "").replace(/\/.*$/, "").trim().toLowerCase();
}

async function ensureGitignoreLine(line, directory = process.cwd()) {
  const gitignore = path.join(directory, ".gitignore");
  const existing = await fs.readFile(gitignore, "utf8").catch(() => null);
  if (existing === null) return { updated: false, reason: "missing .gitignore" };
  if (existing.split(/\r?\n/).includes(line)) return { updated: false, reason: "already ignored" };
  const next = existing.endsWith("\n") ? `${existing}${line}\n` : `${existing}\n${line}\n`;
  await fs.writeFile(gitignore, next, "utf8");
  return { updated: true };
}

async function initEnv(args) {
  const envFile = args.env || DEFAULT_ENV;
  const body = `# Skill Hub shared Shopify configuration
# Choose quick browser connection by default. For long-running agents, set the
# method to dev_dashboard_client_credentials and add the Client ID and Secret.

SKILL_HUB_SHOPIFY_ACCESS_METHOD=shopify_cli_oauth
SKILL_HUB_SHOPIFY_STORE_DOMAIN=
# SKILL_HUB_SHOPIFY_CLIENT_ID=
# SKILL_HUB_SHOPIFY_CLIENT_SECRET=
# SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN=
`;

  const exists = await fs.readFile(envFile, "utf8").then(() => true).catch(() => false);
  if (!exists) await fs.writeFile(envFile, body, "utf8");
  const gitignore = await ensureGitignoreLine(path.basename(envFile), path.dirname(path.resolve(envFile)));
  console.log(JSON.stringify({ envFile, created: !exists, gitignore }, null, 2));
}

async function pathExists(filePath) {
  return fs.access(filePath).then(() => true).catch(() => false);
}

async function resolveShopifyCliJs(env = {}) {
  const configured = env.SKILL_HUB_SHOPIFY_CLI_JS || process.env.SKILL_HUB_SHOPIFY_CLI_JS;
  const candidates = [];
  if (configured) candidates.push(configured);

  const npmRoot = await execFileAsync("npm", ["root", "-g"], { windowsHide: true })
    .then(({ stdout }) => stdout.trim())
    .catch(() => "");
  if (npmRoot) candidates.push(path.join(npmRoot, "@shopify", "cli", "bin", "run.js"));
  if (process.env.APPDATA) candidates.push(path.join(process.env.APPDATA, "npm", "node_modules", "@shopify", "cli", "bin", "run.js"));

  for (const candidate of candidates) {
    if (candidate && await pathExists(candidate)) return candidate;
  }
  const searched = candidates.filter(Boolean).join("; ") || "(none)";
  const error = new Error(`CLI_NOT_FOUND: Could not locate Shopify CLI JS entrypoint. Set SKILL_HUB_SHOPIFY_CLI_JS to @shopify/cli/bin/run.js. Searched: ${searched}`);
  error.code = "CLI_NOT_FOUND";
  throw error;
}

function normalizeCliJson(raw) {
  if (raw?.errors) return { errors: raw.errors };
  return raw?.data ? raw : { data: raw };
}

function classifyCliError(error, detail) {
  const text = `${detail || ""}\n${error?.message || ""}`;
  if (error?.code === "CLI_NOT_FOUND") return { code: "CLI_NOT_FOUND", message: error.message };
  if (error?.code === "ENOENT" || error?.code === "EINVAL" || error?.code === "EFTYPE") {
    return { code: "CLI_SPAWN_FAILED", message: `Could not start Shopify CLI: ${error.message}` };
  }
  if (/access denied|Access denied|denied access/i.test(text)) return { code: "CLI_ACCESS_DENIED", message: text.trim() };
  if (/store auth|stored store auth|auth.*required|not authenticated|login/i.test(text)) return { code: "CLI_AUTH_REQUIRED", message: text.trim() };
  return { code: "CLI_SPAWN_FAILED", message: text.trim() || "Shopify CLI request failed." };
}

async function shopifyCliFetch({ cliJs, shop, query, variables, allowMutations = false }) {
  shop = String(shop || "").trim().toLowerCase();
  if (!shop.endsWith(".myshopify.com") || shop.includes("/")) {
    fail(`Invalid shop domain: ${shop}. Request blocked for security.`);
  }
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "skill-hub-shopify-cli-"));
  const queryFile = path.join(tempDir, "query.graphql");
  const variableFile = path.join(tempDir, "variables.json");
  const outputFile = path.join(tempDir, "output.json");
  try {
    await fs.writeFile(queryFile, query, "utf8");
    await fs.writeFile(variableFile, JSON.stringify(variables || {}), "utf8");
    const args = [
      cliJs,
      "store",
      "execute",
      "--store",
      shop,
      "--query-file",
      queryFile,
      "--variable-file",
      variableFile,
      "--output-file",
      outputFile,
      "--json",
      "--no-color",
    ];
    if (allowMutations) args.push("--allow-mutations");

    const execResult = await execFileAsync(process.execPath, args, {
      timeout: 180000,
      maxBuffer: 1024 * 1024 * 20,
      windowsHide: true,
    }).catch((error) => {
      const detail = [error.stderr, error.stdout].filter(Boolean).map(String).join("\n");
      const classified = classifyCliError(error, detail);
      return { cliError: classified, stdout: error.stdout || "", stderr: error.stderr || "" };
    });

    if (execResult.cliError) return { ok: false, status: execResult.cliError.code, apiVersion: "shopify-cli", json: { errors: [execResult.cliError] } };
    if (!await pathExists(outputFile)) {
      return { ok: false, status: "CLI_OUTPUT_MISSING", apiVersion: "shopify-cli", json: { errors: [{ code: "CLI_OUTPUT_MISSING", message: `Shopify CLI did not create output JSON. ${execResult.stderr || execResult.stdout || ""}`.trim() }] } };
    }

    const output = await fs.readFile(outputFile, "utf8");
    let json;
    try {
      json = normalizeCliJson(JSON.parse(output));
    } catch (error) {
      return { ok: false, status: "CLI_JSON_PARSE_FAILED", apiVersion: "shopify-cli", json: { errors: [{ code: "CLI_JSON_PARSE_FAILED", message: error.message }] } };
    }
    return { ok: !json.errors, status: 200, apiVersion: "shopify-cli", json };
  } catch (error) {
    const classified = classifyCliError(error);
    return { ok: false, status: classified.code, apiVersion: "shopify-cli", json: { errors: [classified] } };
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}

async function resolveStoreDomain(rawInput) {
  const raw = String(rawInput || "").trim();
  if (!raw) fail("A Shopify store URL or domain is required.");
  const adminMatch = raw.match(/admin\.shopify\.com\/store\/([^/\s?#]+)/i);
  if (adminMatch) return `${adminMatch[1].toLowerCase()}.myshopify.com`;
  const host = normalizeDomain(raw);
  if (host.endsWith(".myshopify.com")) return host;

  const html = await fetchPublic(`https://${host}`, {}, { timeoutMs: 15000 })
    .then((response) => response.ok ? response.text() : "")
    .catch(() => "");
  const match = html.match(/(?:Shopify\.shop|myshopify(?:_domain|Domain)|permanent(?:_domain|Domain))\s*[:=]\s*["']([^"']+\.myshopify\.com)["']/i)
    || html.match(/https?:\\?\/\\?\/([a-z0-9-]+\.myshopify\.com)/i);
  if (match?.[1]) return match[1].toLowerCase();
  fail(`Could not identify the Shopify permanent domain from "${raw}". Please paste the Shopify admin URL (https://admin.shopify.com/store/your-store) or your .myshopify.com domain.`);
}

async function resolveAdmin(env) {
  const shop = await resolveStoreDomain(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  if (isDevDashboardMode(env)) {
    const result = await devDashboardGraphql(env, shop, `query SkillHubAltTextConnectionCheck { shop { name myshopifyDomain } }`);
    assertRequiredScopes(result.scopes, REQUIRED_SCOPES);
    return { shop, version: result.version, transport: "dev_dashboard_client_credentials", env, shopInfo: result.data.shop };
  }
  const cliJs = await resolveShopifyCliJs(env);
  const cliProbe = await shopifyCliFetch({
    cliJs,
    shop,
    query: `query SkillHubAltTextConnectionCheck { shop { name myshopifyDomain } }`,
    variables: {},
  });
  if (cliProbe.ok) return { shop, version: "shopify-cli", transport: "shopify_cli", cliJs, shopInfo: cliProbe.json.data.shop };
  const detail = JSON.stringify(cliProbe.json.errors || cliProbe.json, null, 2);
  fail(`Shopify CLI connection check failed. If authorization is missing or expired, run: shopify store auth --store ${shop} --scopes ${REQUIRED_SCOPES} --json\n${detail}`);
}

async function gql(client, query, variables = {}) {
  if (client.transport === "dev_dashboard_client_credentials") {
    return (await devDashboardGraphql(client.env, client.shop, query, variables)).data;
  }
  const allowMutations = /(^|\n)\s*mutation\b/i.test(query);
  const result = await shopifyCliFetch({ cliJs: client.cliJs, shop: client.shop, query, variables, allowMutations });
  if (!result.ok) {
    const detail = JSON.stringify(result.json.errors || result.json, null, 2);
    fail(`Shopify GraphQL request failed: ${detail}`);
  }
  return result.json.data;
}

function firstImageUrl(node) {
  return node?.image?.url || node?.preview?.image?.url || null;
}

function normalizeAlt(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeText(value) {
  return String(value || "").trim().toLowerCase();
}

function escapeSearchValue(value) {
  return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function gidType(value) {
  const match = String(value || "").match(/^gid:\/\/shopify\/([^/]+)\//);
  return match?.[1] || "";
}

function handleFromUrl(value, segment) {
  const raw = String(value || "").trim();
  if (!raw || !/^https?:\/\//i.test(raw)) return "";
  try {
    const url = new URL(raw);
    const parts = url.pathname.split("/").filter(Boolean);
    const index = parts.indexOf(segment);
    return index !== -1 && parts[index + 1] ? decodeURIComponent(parts[index + 1]) : "";
  } catch {
    return "";
  }
}

function articleHandleFromUrl(value) {
  const raw = String(value || "").trim();
  if (!raw || !/^https?:\/\//i.test(raw)) return "";
  try {
    const url = new URL(raw);
    const parts = url.pathname.split("/").filter(Boolean);
    const blogsIndex = parts.indexOf("blogs");
    return blogsIndex !== -1 && parts[blogsIndex + 2] ? decodeURIComponent(parts[blogsIndex + 2]) : "";
  } catch {
    return "";
  }
}

function productImageRecord(product, media, position) {
  const alt = normalizeAlt(media.alt);
  return {
    type: "product_media",
    id: media.id,
    productId: product.id,
    productTitle: product.title,
    productHandle: product.handle,
    vendor: product.vendor,
    productType: product.productType,
    tags: product.tags,
    status: product.status,
    position,
    currentAlt: alt,
    issue: issueForAlt(alt),
    url: firstImageUrl(media),
  };
}

function collectionImageRecord(collection) {
  const alt = normalizeAlt(collection.image?.altText);
  return {
    type: "collection_featured_image",
    id: collection.id,
    collectionTitle: collection.title,
    collectionHandle: collection.handle,
    descriptionHtml: collection.descriptionHtml,
    currentAlt: alt,
    issue: issueForAlt(alt),
    url: collection.image?.url || null,
  };
}

function articleFeaturedImageRecord(article) {
  const alt = normalizeAlt(article.image?.altText);
  return {
    type: "article_featured_image",
    id: article.id,
    articleTitle: article.title,
    articleHandle: article.handle,
    blogTitle: article.blog?.title,
    summary: article.summary,
    currentAlt: alt,
    issue: issueForAlt(alt),
    url: article.image?.url || null,
  };
}

function issueForAlt(alt) {
  if (!alt) return "missing";
  if (alt.length > SOFT_ALT_LIMIT) return "over-soft-limit";
  const words = alt.toLowerCase().split(/[\s,;|/]+/).filter(Boolean);
  const unique = new Set(words);
  if (words.length >= 8 && unique.size / words.length < 0.6) return "repetitive";
  return null;
}

const PRODUCT_TARGET_QUERY = `query SkillHubAltTextProductTarget($identifier: ProductIdentifierInput!) {
  product: productByIdentifier(identifier: $identifier) {
    id title handle vendor productType status tags description
    media(first: 50) {
      nodes {
        ... on MediaImage {
          id alt status fileStatus
          image { url width height }
          preview { image { url } }
        }
      }
    }
  }
}`;

const PRODUCTS_SEARCH_QUERY = `query SkillHubAltTextProductSearch($first: Int!, $query: String) {
  products(first: $first, query: $query, sortKey: UPDATED_AT, reverse: true) {
    nodes {
      id title handle vendor productType status tags description
      media(first: 50) {
        nodes {
          ... on MediaImage {
            id alt status fileStatus
            image { url width height }
            preview { image { url } }
          }
        }
      }
    }
  }
}`;

const COLLECTION_TARGET_QUERY = `query SkillHubAltTextCollectionTarget($identifier: CollectionIdentifierInput!) {
  collection: collectionByIdentifier(identifier: $identifier) {
    id title handle descriptionHtml image { url altText }
  }
}`;

const ARTICLE_TARGET_QUERY = `query SkillHubAltTextArticleTarget($id: ID!) {
  article(id: $id) {
    id title handle summary body
    blog { title handle }
    image { altText url }
  }
}`;

const MEDIA_NODE_QUERY = `query SkillHubAltTextMediaNode($id: ID!) {
  node(id: $id) {
    ... on MediaImage {
      id alt status fileStatus
      image { url width height }
      preview { image { url } }
    }
  }
}`;

async function readProducts(client, first) {
  const query = `query SkillHubAltTextProducts($first: Int!, $after: String) {
    products(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id title handle vendor productType status tags description
        media(first: 50) {
          nodes {
            ... on MediaImage {
              id alt status fileStatus
              image { url width height }
              preview { image { url } }
            }
          }
        }
      }
    }
  }`;
  const products = [];
  let after = null;
  do {
    const data = await gql(client, query, { first, after });
    products.push(...data.products.nodes);
    after = data.products.pageInfo.hasNextPage ? data.products.pageInfo.endCursor : null;
  } while (after);
  return products;
}

async function productByIdentifier(client, identifier) {
  const data = await gql(client, PRODUCT_TARGET_QUERY, { identifier });
  return data.product || null;
}

async function findProductTarget(client, target) {
  const value = String(target || "").trim();
  if (!value) return { matches: [], method: "none" };

  if (gidType(value) === "Product") {
    const product = await productByIdentifier(client, { id: value });
    return { matches: product ? [product] : [], method: "product_gid" };
  }

  const handle = handleFromUrl(value, "products") || (!value.includes(" ") && !value.includes("/") ? value : "");
  if (handle) {
    const product = await productByIdentifier(client, { handle });
    if (product) return { matches: [product], method: "product_handle" };
  }

  const search = `title:"${escapeSearchValue(value)}"`;
  const data = await gql(client, PRODUCTS_SEARCH_QUERY, { first: 20, query: search });
  let matches = data.products.nodes || [];
  const normalized = normalizeText(value);
  const exact = matches.filter((product) => normalizeText(product.title) === normalized || normalizeText(product.handle) === normalized);
  if (exact.length) matches = exact;
  if (!matches.length) {
    const products = await readProducts(client, 50);
    matches = products.filter((product) => normalizeText(product.title).includes(normalized) || normalizeText(product.handle).includes(normalized));
    return { matches, method: "product_inventory_filter" };
  }
  return { matches, method: "product_title_search" };
}

async function readCollections(client, first) {
  const query = `query SkillHubAltTextCollections($first: Int!, $after: String) {
    collections(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { id title handle descriptionHtml image { url altText } }
    }
  }`;
  const collections = [];
  let after = null;
  do {
    const data = await gql(client, query, { first, after });
    collections.push(...data.collections.nodes);
    after = data.collections.pageInfo.hasNextPage ? data.collections.pageInfo.endCursor : null;
  } while (after);
  return collections;
}

async function collectionByIdentifier(client, identifier) {
  const data = await gql(client, COLLECTION_TARGET_QUERY, { identifier });
  return data.collection || null;
}

async function findCollectionTarget(client, target, first) {
  const value = String(target || "").trim();
  if (!value) return { matches: [], method: "none" };

  if (gidType(value) === "Collection") {
    const collection = await collectionByIdentifier(client, { id: value });
    return { matches: collection ? [collection] : [], method: "collection_gid" };
  }

  const handle = handleFromUrl(value, "collections") || (!value.includes(" ") && !value.includes("/") ? value : "");
  if (handle) {
    const collection = await collectionByIdentifier(client, { handle });
    if (collection) return { matches: [collection], method: "collection_handle" };
  }

  const collections = await readCollections(client, first);
  const normalized = normalizeText(value);
  const matches = collections.filter((collection) => normalizeText(collection.title).includes(normalized) || normalizeText(collection.handle).includes(normalized));
  return { matches, method: "collection_title_filter" };
}

async function readArticles(client, first) {
  const query = `query SkillHubAltTextArticles($first: Int!, $after: String) {
    articles(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id title handle summary body
        blog { title handle }
        image { altText url }
      }
    }
  }`;
  const articles = [];
  let after = null;
  do {
    const data = await gql(client, query, { first, after });
    articles.push(...data.articles.nodes);
    after = data.articles.pageInfo.hasNextPage ? data.articles.pageInfo.endCursor : null;
  } while (after);
  return articles;
}

async function articleById(client, id) {
  const data = await gql(client, ARTICLE_TARGET_QUERY, { id });
  return data.article || null;
}

async function findArticleTarget(client, target, first) {
  const value = String(target || "").trim();
  if (!value) return { matches: [], method: "none" };

  if (gidType(value) === "Article") {
    const article = await articleById(client, value);
    return { matches: article ? [article] : [], method: "article_gid" };
  }

  const articleHandle = articleHandleFromUrl(value);
  const articles = await readArticles(client, first);
  const normalized = normalizeText(articleHandle || value);
  const matches = articles.filter((article) => {
    return normalizeText(article.title).includes(normalized) ||
      normalizeText(article.handle).includes(normalized) ||
      `${normalizeText(article.blog?.handle)}/${normalizeText(article.handle)}`.includes(normalized);
  });
  return { matches, method: articleHandle ? "article_url_handle_filter" : "article_title_filter" };
}

function extractInlineImages(html) {
  const images = [];
  if (!html) return images;
  const regex = /<img\b[^>]*>/gi;
  let match;
  let index = 0;
  while ((match = regex.exec(html))) {
    const tag = match[0];
    const src = attr(tag, "src") || attr(tag, "data-src");
    const alt = attr(tag, "alt") || "";
    images.push({ index, tag, src, alt, issue: issueForAlt(alt), offset: match.index });
    index += 1;
  }
  return images;
}

function attr(tag, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`\\b${escaped}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s>]+))`, "i");
  const match = tag.match(regex);
  return match ? (match[2] ?? match[3] ?? match[4] ?? "") : null;
}

async function scan(args) {
  const env = await loadEnv(args.env || DEFAULT_ENV);
  const client = await resolveAdmin(env);
  const pageSize = Number(args["page-size"] || 50);
  const includeProducts = args.surface ? String(args.surface).includes("products") : true;
  const includeCollections = args.surface ? String(args.surface).includes("collections") : true;
  const includeArticles = args.surface ? String(args.surface).includes("articles") : true;
  const inventory = {
    scannedAt: new Date().toISOString(),
    shop: { domain: client.shop, apiVersion: client.version, name: client.shopInfo.name },
    summary: {
      products: 0,
      productImages: 0,
      productImagesNeedingOptimization: 0,
      collections: 0,
      collectionImages: 0,
      collectionImagesNeedingOptimization: 0,
      articles: 0,
      articleFeaturedImages: 0,
      articleFeaturedImagesNeedingOptimization: 0,
      articleInlineImages: 0,
      articleInlineImagesNeedingOptimization: 0,
    },
    productImages: [],
    collectionImages: [],
    articleFeaturedImages: [],
    articleInlineImages: [],
    sharedFiles: [],
  };

  const fileRefs = new Map();
  if (includeProducts) {
    const products = await readProducts(client, pageSize);
    inventory.summary.products = products.length;
    for (const product of products) {
      let position = 0;
      for (const media of product.media.nodes.filter(Boolean)) {
        const alt = normalizeAlt(media.alt);
        const issue = issueForAlt(alt);
        const image = {
          type: "product_media",
          id: media.id,
          productId: product.id,
          productTitle: product.title,
          productHandle: product.handle,
          vendor: product.vendor,
          productType: product.productType,
          tags: product.tags,
          status: product.status,
          position,
          currentAlt: alt,
          issue,
          url: firstImageUrl(media),
        };
        inventory.productImages.push(image);
        const refs = fileRefs.get(media.id) || [];
        refs.push({ productId: product.id, productTitle: product.title, position });
        fileRefs.set(media.id, refs);
        position += 1;
      }
    }
    inventory.summary.productImages = inventory.productImages.length;
    inventory.summary.productImagesNeedingOptimization = inventory.productImages.filter((item) => item.issue).length;
    inventory.sharedFiles = [...fileRefs.entries()]
      .filter(([, refs]) => refs.length > 1)
      .map(([id, refs]) => ({ id, references: refs }));
  }

  if (includeCollections) {
    const collections = await readCollections(client, pageSize);
    inventory.summary.collections = collections.length;
    for (const collection of collections) {
      if (!collection.image?.url) continue;
      const alt = normalizeAlt(collection.image.altText);
      const issue = issueForAlt(alt);
      inventory.collectionImages.push({
        type: "collection_featured_image",
        id: collection.id,
        collectionTitle: collection.title,
        collectionHandle: collection.handle,
        descriptionHtml: collection.descriptionHtml,
        currentAlt: alt,
        issue,
        url: collection.image.url,
      });
    }
    inventory.summary.collectionImages = inventory.collectionImages.length;
    inventory.summary.collectionImagesNeedingOptimization = inventory.collectionImages.filter((item) => item.issue).length;
  }

  if (includeArticles) {
    const articles = await readArticles(client, pageSize);
    inventory.summary.articles = articles.length;
    for (const article of articles) {
      if (article.image?.url) {
        const alt = normalizeAlt(article.image.altText);
        const issue = issueForAlt(alt);
        inventory.articleFeaturedImages.push({
          type: "article_featured_image",
          id: article.id,
          articleTitle: article.title,
          articleHandle: article.handle,
          blogTitle: article.blog?.title,
          summary: article.summary,
          currentAlt: alt,
          issue,
          url: article.image.url,
        });
      }
      for (const image of extractInlineImages(article.body)) {
        inventory.articleInlineImages.push({
          type: "article_inline_image",
          id: `${article.id}#inline-${image.index}`,
          articleId: article.id,
          articleTitle: article.title,
          articleHandle: article.handle,
          blogTitle: article.blog?.title,
          inlineIndex: image.index,
          src: image.src,
          currentAlt: normalizeAlt(image.alt),
          issue: image.issue,
        });
      }
    }
    inventory.summary.articleFeaturedImages = inventory.articleFeaturedImages.length;
    inventory.summary.articleFeaturedImagesNeedingOptimization = inventory.articleFeaturedImages.filter((item) => item.issue).length;
    inventory.summary.articleInlineImages = inventory.articleInlineImages.length;
    inventory.summary.articleInlineImagesNeedingOptimization = inventory.articleInlineImages.filter((item) => item.issue).length;
  }

  console.log(JSON.stringify(inventory, null, 2));
}

async function downloadTargetImages(items, limit) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "skill-hub-alt-target-"));
  const downloaded = [];
  for (const [index, item] of items.filter((entry) => entry.url).slice(0, limit).entries()) {
    const localPath = path.join(tempDir, `target-${index + 1}${extensionFromUrl(item.url)}`);
    try {
      const bytes = await downloadImage(item.url, localPath);
      downloaded.push({ id: item.id, type: item.type, url: item.url, localPath, bytes, status: "downloaded" });
    } catch (error) {
      downloaded.push({ id: item.id, type: item.type, url: item.url, localPath, status: "download_failed", error: error.message });
    }
  }
  return {
    tempDir,
    cleanupRequired: true,
    instruction: "Open localPath values with the host-native image input or image-view tool before claiming visual evidence, then delete tempDir.",
    downloaded,
  };
}

async function target(args) {
  const env = await loadEnv(args.env || DEFAULT_ENV);
  const client = await resolveAdmin(env);
  const pageSize = Number(args["page-size"] || 50);
  const input = args.product || args.collection || args.article || args["media-id"] || args.url || args.target || args._?.[0];
  if (!input) fail("Use target with --product, --collection, --article, --media-id, --url, or --target.");

  const result = {
    ok: true,
    shop: { domain: client.shop, apiVersion: client.version, name: client.shopInfo.name },
    input,
    method: null,
    matches: [],
    items: [],
  };

  if (args.product || (!args.collection && !args.article && !args["media-id"] && !args.url)) {
    const found = await findProductTarget(client, input);
    result.method = found.method;
    result.matches = found.matches.map((product) => ({ type: "product", id: product.id, title: product.title, handle: product.handle, status: product.status }));
    for (const product of found.matches) {
      let position = 0;
      for (const media of product.media.nodes.filter(Boolean)) {
        result.items.push(productImageRecord(product, media, position));
        position += 1;
      }
    }
  } else if (args.collection) {
    const found = await findCollectionTarget(client, input, pageSize);
    result.method = found.method;
    result.matches = found.matches.map((collection) => ({ type: "collection", id: collection.id, title: collection.title, handle: collection.handle }));
    for (const collection of found.matches) {
      if (collection.image?.url) result.items.push(collectionImageRecord(collection));
    }
  } else if (args.article) {
    const found = await findArticleTarget(client, input, pageSize);
    result.method = found.method;
    result.matches = found.matches.map((article) => ({ type: "article", id: article.id, title: article.title, handle: article.handle, blogTitle: article.blog?.title }));
    for (const article of found.matches) {
      if (article.image?.url) result.items.push(articleFeaturedImageRecord(article));
      for (const image of extractInlineImages(article.body)) {
        result.items.push({
          type: "article_inline_image",
          id: `${article.id}#inline-${image.index}`,
          articleId: article.id,
          articleTitle: article.title,
          articleHandle: article.handle,
          blogTitle: article.blog?.title,
          inlineIndex: image.index,
          src: image.src,
          url: image.src,
          currentAlt: normalizeAlt(image.alt),
          issue: image.issue,
        });
      }
    }
  } else if (args["media-id"]) {
    const data = await gql(client, MEDIA_NODE_QUERY, { id: input });
    result.method = "media_gid";
    if (data.node) {
      const alt = normalizeAlt(data.node.alt);
      result.items.push({
        type: "product_media",
        id: data.node.id,
        currentAlt: alt,
        issue: issueForAlt(alt),
        url: firstImageUrl(data.node),
        note: "MediaImage was found by ID. Parent product context is not available from this lookup; use --product when product context is needed.",
      });
    }
  } else if (args.url) {
    const scanArgs = { ...args, surface: args.surface || "products,collections,articles" };
    const products = await readProducts(client, pageSize);
    result.method = "image_url_filter";
    for (const product of products) {
      let position = 0;
      for (const media of product.media.nodes.filter(Boolean)) {
        const item = productImageRecord(product, media, position);
        if (item.url === input) result.items.push(item);
        position += 1;
      }
    }
    if (!scanArgs.surface || String(scanArgs.surface).includes("collections")) {
      const collections = await readCollections(client, pageSize);
      for (const collection of collections) {
        if (collection.image?.url === input) result.items.push(collectionImageRecord(collection));
      }
    }
    if (!scanArgs.surface || String(scanArgs.surface).includes("articles")) {
      const articles = await readArticles(client, pageSize);
      for (const article of articles) {
        if (article.image?.url === input) result.items.push(articleFeaturedImageRecord(article));
        for (const image of extractInlineImages(article.body)) {
          if (image.src === input) {
            result.items.push({
              type: "article_inline_image",
              id: `${article.id}#inline-${image.index}`,
              articleId: article.id,
              articleTitle: article.title,
              articleHandle: article.handle,
              blogTitle: article.blog?.title,
              inlineIndex: image.index,
              src: image.src,
              url: image.src,
              currentAlt: normalizeAlt(image.alt),
              issue: image.issue,
            });
          }
        }
      }
    }
  }

  result.count = result.items.length;
  result.needingOptimization = result.items.filter((item) => item.issue).length;
  if (args.download) {
    result.download = await downloadTargetImages(result.items, Math.min(Math.max(Number(args.limit || result.items.length || 1), 1), 20));
  }
  console.log(JSON.stringify(result, null, 2));
}

function extensionFromUrl(url) {
  try {
    const pathname = new URL(url).pathname.toLowerCase();
    const ext = path.extname(pathname);
    return ext && ext.length <= 6 ? ext : ".jpg";
  } catch {
    return ".jpg";
  }
}

async function downloadImage(url, filePath) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();
    const isShopifyCdn = host === "cdn.shopify.com" || 
                         host.endsWith(".shopifycdn.com") || 
                         host.endsWith(".shopifycdn.net") || 
                         host.endsWith(".myshopify.com");
    if (!isShopifyCdn) {
      throw new Error(`Invalid CDN domain: "${host}". Only official Shopify CDN downloads are allowed.`);
    }
  } catch (urlErr) {
    throw new Error(`Invalid URL: ${urlErr.message}`);
  }
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const bytes = Buffer.from(await response.arrayBuffer());
      await fs.writeFile(filePath, bytes);
      return bytes.length;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 300 * attempt));
    }
  }
  throw new Error(lastError?.cause?.message || lastError?.message || "download failed");
}

async function visionSample(args) {
  const env = await loadEnv(args.env || DEFAULT_ENV);
  const client = await resolveAdmin(env);
  const limit = Math.min(Math.max(Number(args.limit || 3), 1), 10);
  const pageSize = Number(args["page-size"] || 50);
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "skill-hub-alt-vision-"));
  const candidates = [];

  const products = await readProducts(client, pageSize);
  for (const product of products) {
    let position = 0;
    for (const media of product.media.nodes.filter(Boolean)) {
      const url = firstImageUrl(media);
      if (url) {
        candidates.push({
          type: "product_media",
          id: media.id,
          url,
          currentAlt: normalizeAlt(media.alt),
          context: { productId: product.id, productTitle: product.title, productHandle: product.handle, position },
        });
      }
      position += 1;
    }
  }

  if (candidates.length < limit) {
    const collections = await readCollections(client, pageSize);
    for (const collection of collections) {
      if (collection.image?.url) {
        candidates.push({
          type: "collection_featured_image",
          id: collection.id,
          url: collection.image.url,
          currentAlt: normalizeAlt(collection.image.altText),
          context: { collectionTitle: collection.title, collectionHandle: collection.handle },
        });
      }
    }
  }

  if (candidates.length < limit) {
    const articles = await readArticles(client, pageSize);
    for (const article of articles) {
      if (article.image?.url) {
        candidates.push({
          type: "article_featured_image",
          id: article.id,
          url: article.image.url,
          currentAlt: normalizeAlt(article.image.altText),
          context: { articleTitle: article.title, articleHandle: article.handle, blogTitle: article.blog?.title },
        });
      }
      for (const image of extractInlineImages(article.body)) {
        if (!image.src) continue;
        candidates.push({
          type: "article_inline_image",
          id: `${article.id}#inline-${image.index}`,
          url: image.src,
          currentAlt: normalizeAlt(image.alt),
          context: { articleId: article.id, articleTitle: article.title, inlineIndex: image.index },
        });
      }
    }
  }

  const samples = [];
  for (const [index, candidate] of candidates.slice(0, limit).entries()) {
    const localPath = path.join(tempDir, `sample-${index + 1}${extensionFromUrl(candidate.url)}`);
    try {
      const bytes = await downloadImage(candidate.url, localPath);
      samples.push({ ...candidate, localPath, bytes, status: "downloaded" });
    } catch (error) {
      samples.push({ ...candidate, localPath, status: "download_failed", error: error.message });
    }
  }

  console.log(JSON.stringify({
    ok: true,
    tempDir,
    cleanupRequired: true,
    instruction: "Open each localPath with the host-native image input or Read/image-view tool, report at least 3 pixel-derived facts, then delete tempDir after the probe.",
    samples,
  }, null, 2));
}

function validatePlan(plan) {
  if (!plan || !Array.isArray(plan.changes)) fail("Plan JSON must contain a changes array.");
  const errors = [];
  for (const [index, change] of plan.changes.entries()) {
    if (!change.type) errors.push(`changes[${index}].type is required`);
    if (!change.id) errors.push(`changes[${index}].id is required`);
    if (typeof change.alt !== "string" || !change.alt.trim()) errors.push(`changes[${index}].alt is required`);
    if (change.alt && change.alt.length > HARD_ALT_LIMIT) errors.push(`changes[${index}].alt exceeds ${HARD_ALT_LIMIT} characters`);
    if (change.source !== "vision") {
      errors.push(`changes[${index}].source must be "vision"; Alt Text optimization requires a verified vision model`);
    } else {
      const evidenceParts = String(change.visualEvidence || "")
        .split(/[.;,\n]+/)
        .map((part) => part.trim())
        .filter(Boolean);
      if (evidenceParts.length < 3) {
        errors.push(`changes[${index}].visualEvidence must include at least 3 pixel-derived facts when source is "vision"`);
      }
    }
  }
  if (errors.length) fail(errors.join("\n"));
}

async function loadPlan(file) {
  let text;
  if (file === "-") {
    text = await new Promise((resolve, reject) => {
      let input = "";
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (chunk) => {
        input += chunk;
      });
      process.stdin.on("end", () => resolve(input));
      process.stdin.on("error", reject);
    });
  } else {
    text = await fs.readFile(file, "utf8").catch(() => null);
  }
  if (!text) fail(`Missing plan input: ${file || "(missing)"}`);
  const plan = JSON.parse(text);
  validatePlan(plan);
  return plan;
}

function replaceInlineAlt(html, inlineIndex, alt) {
  let seen = 0;
  return html.replace(/<img\b[^>]*>/gi, (tag) => {
    if (seen !== inlineIndex) {
      seen += 1;
      return tag;
    }
    seen += 1;
    const escaped = alt.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
    if (/\balt\s*=/i.test(tag)) {
      return tag.replace(/\balt\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))/i, `alt="${escaped}"`);
    }
    return tag.replace(/\/?>$/, (end) => ` alt="${escaped}"${end}`);
  });
}

async function applyPlan(args) {
  const plan = await loadPlan(args.input);
  const execute = Boolean(args.execute);
  if (!execute) {
    console.log(JSON.stringify({ mode: "preview", changes: plan.changes }, null, 2));
    return;
  }
  const env = await loadEnv(args.env || DEFAULT_ENV);
  const client = await resolveAdmin(env);
  const results = [];
  const assertNoUserErrors = (payload, operation) => {
    if (!payload || !Array.isArray(payload.userErrors)) fail(`${operation} did not return userErrors; refusing to report success.`);
    if (payload.userErrors.length) fail(`${operation} failed: ${JSON.stringify(payload.userErrors)}`);
  };

  const fileUpdates = plan.changes
    .filter((change) => change.type === "product_media")
    .map((change) => ({ id: change.id, alt: change.alt }));
  if (fileUpdates.length) {
    const data = await gql(
      client,
      `mutation SkillHubAltTextFileUpdate($files: [FileUpdateInput!]!) {
        fileUpdate(files: $files) {
          files { id alt fileStatus }
          userErrors { field message code }
        }
      }`,
      { files: fileUpdates },
    );
    assertNoUserErrors(data.fileUpdate, "product media update");
    results.push({ type: "product_media", response: data.fileUpdate });
  }

  for (const change of plan.changes.filter((item) => item.type === "collection_featured_image")) {
    const src = change.url || change.src;
    if (!src) fail(`collection change ${change.id} needs url to preserve the image.`);
    const data = await gql(
      client,
      `mutation SkillHubAltTextCollectionUpdate($input: CollectionInput!) {
        collectionUpdate(input: $input) {
          collection { id image { url altText } }
          userErrors { field message }
        }
      }`,
      { input: { id: change.id, image: { src, altText: change.alt } } },
    );
    assertNoUserErrors(data.collectionUpdate, "collection featured image update");
    results.push({ type: "collection_featured_image", id: change.id, response: data.collectionUpdate });
  }

  for (const change of plan.changes.filter((item) => item.type === "article_featured_image")) {
    const url = change.url || change.src;
    if (!url) fail(`article featured image change ${change.id} needs url to preserve the image.`);
    const data = await gql(
      client,
      `mutation SkillHubAltTextArticleImageUpdate($id: ID!, $article: ArticleUpdateInput!) {
        articleUpdate(id: $id, article: $article) {
          article { id image { altText url } }
          userErrors { code field message }
        }
      }`,
      { id: change.id, article: { image: { url, altText: change.alt } } },
    );
    assertNoUserErrors(data.articleUpdate, "article featured image update");
    results.push({ type: "article_featured_image", id: change.id, response: data.articleUpdate });
  }

  const inlineByArticle = new Map();
  for (const change of plan.changes.filter((item) => item.type === "article_inline_image")) {
    if (!change.articleId || !Number.isInteger(change.inlineIndex)) {
      fail(`article inline change ${change.id} needs articleId and numeric inlineIndex.`);
    }
    const list = inlineByArticle.get(change.articleId) || [];
    list.push(change);
    inlineByArticle.set(change.articleId, list);
  }
  for (const [articleId, changes] of inlineByArticle.entries()) {
    const current = await gql(
      client,
      `query SkillHubAltTextArticleBody($id: ID!) { article(id: $id) { id body } }`,
      { id: articleId },
    );
    let body = current.article.body;
    for (const change of changes.sort((left, right) => left.inlineIndex - right.inlineIndex)) {
      body = replaceInlineAlt(body, change.inlineIndex, change.alt);
    }
    const data = await gql(
      client,
      `mutation SkillHubAltTextArticleBodyUpdate($id: ID!, $article: ArticleUpdateInput!) {
        articleUpdate(id: $id, article: $article) {
          article { id }
          userErrors { code field message }
        }
      }`,
      { id: articleId, article: { body } },
    );
    assertNoUserErrors(data.articleUpdate, "article inline image update");
    results.push({ type: "article_inline_image", articleId, changed: changes.length, response: data.articleUpdate });
  }

  console.log(JSON.stringify({ mode: "execute", results }, null, 2));
}

async function connectionCheck(args) {
  const env = await loadEnv(args.env || DEFAULT_ENV);
  const client = await resolveAdmin(env);
  console.log(JSON.stringify({
    ok: true,
    shop: {
      domain: client.shop,
      apiVersion: client.version,
      transport: client.transport,
      name: client.shopInfo.name,
      myshopifyDomain: client.shopInfo.myshopifyDomain,
    },
  }, null, 2));
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  if (!command || command === "--help" || command === "-h" || command === "help") {
    printUsage();
    return;
  }
  const args = parseArgs(rest);
  if (command === "init-env") return initEnv(args);
  if (command === "connection-check") return connectionCheck(args);
  if (command === "target") return target(args);
  if (command === "scan") return scan(args);
  if (command === "vision-sample") return visionSample(args);
  if (command === "apply") return applyPlan(args);
  printUsage();
  fail(`Unknown command: ${command || "(missing)"}`);
}

main().catch((error) => fail(error.stack || error.message));
