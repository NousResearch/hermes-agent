#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { loadShopifyEnv, shopifyCliGraphql as sharedShopifyCliGraphql } from "./lib/shopify-helpers.mjs";

const DEFAULT_ENV = "skill-hub.env";
const REQUIRED_SCOPES = "read_products,read_content,write_content,read_files,write_files";

function parseArgs(argv) {
  const args = {
    command: argv[0],
    env: DEFAULT_ENV,
    input: null,
    articleId: null,
    productPageSize: 50,
    execute: false,
    requireImages: false,
  };

  for (let i = 1; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--env") {
      args.env = value;
      i += 1;
    } else if (key === "--method") {
      args.method = value;
      i += 1;
    } else if (key === "--input") {
      args.input = value;
      i += 1;
    } else if (key === "--article-id") {
      args.articleId = value;
      i += 1;
    } else if (key === "--product-page-size") {
      args.productPageSize = Number(value);
      i += 1;
    } else if (key === "--execute") {
      args.execute = true;
    } else if (key === "--require-images") {
      args.requireImages = true;
    }
  }

  if (!["init-env", "connection-check", "context", "upload-images", "create-draft", "update-draft", "verify"].includes(args.command)) {
    throw new Error(
      "Usage: node shopify-blog-admin.mjs <init-env|connection-check|context|upload-images|create-draft|update-draft|verify> [--env skill-hub.env] [--input file.json] [--article-id gid://shopify/Article/...] [--execute] [--require-images]",
    );
  }

  return args;
}

async function ensureGitignoreLine(line) {
  const gitignore = ".gitignore";
  const existing = await readFile(gitignore, "utf8").catch(() => null);
  if (existing === null) return { updated: false, reason: "missing .gitignore" };
  if (existing.split(/\r?\n/).includes(line)) return { updated: false, reason: "already ignored" };
  const next = existing.endsWith("\n") ? `${existing}${line}\n` : `${existing}\n${line}\n`;
  await writeFile(gitignore, next, "utf8");
  return { updated: true };
}

async function initEnv(args) {
  const envFile = args.env || DEFAULT_ENV;
  const body = `# Skill Hub shared Shopify configuration
# Quick browser connection is the default. For a long-running agent, change the
# method and add the Client ID and Client Secret in this private file.

SKILL_HUB_SHOPIFY_ACCESS_METHOD=shopify_cli_oauth
SKILL_HUB_SHOPIFY_STORE_DOMAIN=
# SKILL_HUB_SHOPIFY_CLIENT_ID=
# SKILL_HUB_SHOPIFY_CLIENT_SECRET=
# SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN=
`;

  const exists = await readFile(envFile, "utf8").then(() => true).catch(() => false);
  if (!exists) await writeFile(envFile, body, "utf8");
  const gitignore = await ensureGitignoreLine(envFile);
  console.log(JSON.stringify({ ok: true, envFile, created: !exists, gitignore, requiredScopes: REQUIRED_SCOPES }, null, 2));
}

async function loadEnv(envPath) {
  return loadShopifyEnv(envPath);
}

async function graphql(env, query, variables = {}) {
  return sharedShopifyCliGraphql(env, query, variables);
}

const CONTEXT_QUERY = `#graphql
query BrandVoiceOverview($blogFirst: Int!, $articleFirst: Int!, $productFirst: Int!, $after: String) {
  shop { name description myshopifyDomain primaryDomain { host url } }
  blogs(first: $blogFirst) { nodes { id title handle } }
  articles(first: $articleFirst, sortKey: PUBLISHED_AT, reverse: true) {
    nodes { id title handle summary publishedAt author { name } blog { id title handle } }
  }
  products(first: $productFirst, after: $after, sortKey: UPDATED_AT, reverse: true) {
    nodes {
      id title handle description productType vendor status tags onlineStoreUrl
      seo { title description }
      collections(first: 5) { nodes { id title handle } }
      featuredMedia { preview { image { url altText width height } } }
      media(first: 5) {
        nodes {
          alt mediaContentType
          preview { image { url altText width height } }
          ... on MediaImage { image { url altText width height } }
        }
      }
      priceRangeV2 {
        minVariantPrice { amount currencyCode }
        maxVariantPrice { amount currencyCode }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}`;

const STAGED_UPLOADS_CREATE = `#graphql
mutation StageImages($input: [StagedUploadInput!]!) {
  stagedUploadsCreate(input: $input) {
    stagedTargets { url resourceUrl parameters { name value } }
    userErrors { field message }
  }
}`;

const FILE_CREATE = `#graphql
mutation CreateFiles($files: [FileCreateInput!]!) {
  fileCreate(files: $files) {
    files {
      id alt fileStatus createdAt
      ... on MediaImage { image { url altText width height } }
    }
    userErrors { field message }
  }
}`;

const FILE_NODE = `#graphql
query FileNode($id: ID!) {
  node(id: $id) {
    id
    ... on MediaImage {
      fileStatus
      alt
      image { url altText width height }
    }
  }
}`;

const ARTICLE_CREATE = `#graphql
mutation CreateArticle($article: ArticleCreateInput!) {
  articleCreate(article: $article) {
    article { id title handle isPublished publishedAt author { name } }
    userErrors { code field message }
  }
}`;

const ARTICLE_UPDATE = `#graphql
mutation UpdateArticle($id: ID!, $article: ArticleUpdateInput!) {
  articleUpdate(id: $id, article: $article) {
    article { id title handle isPublished publishedAt author { name } }
    userErrors { code field message }
  }
}`;

const ARTICLE_VERIFY = `#graphql
query VerifyArticle($id: ID!) {
  article(id: $id) {
    id title handle body summary isPublished publishedAt
    author { name }
    image { altText url }
    metafields(first: 20) { nodes { namespace key type value } }
  }
}`;

const CONNECTION_CHECK = `#graphql
query BlogConnectionCheck {
  shop { name myshopifyDomain }
}`;

async function readJson(filePath) {
  return JSON.parse(await readFile(filePath, "utf8"));
}

function assertNoUserErrors(label, userErrors = []) {
  if (userErrors.length) throw new Error(`${label}: ${JSON.stringify(userErrors, null, 2)}`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assertDraftImages(draft) {
  const body = String(draft.body || "");
  const hasBodyImage = /<img\b/i.test(body) && /cdn\.shopify\.com/i.test(body);
  const hasCoverImage = Boolean(draft.image?.url && /cdn\.shopify\.com/i.test(draft.image.url));
  if (!hasBodyImage && !hasCoverImage) {
    throw new Error(
      "This draft has no Shopify CDN images. Upload images first and insert returned CDN URLs, or rerun without --require-images only if the user explicitly approved a text-only draft.",
    );
  }
}

async function pollMediaImage(env, fileId, attempts = 20) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const data = await graphql(env, FILE_NODE, { id: fileId });
    const node = data.node;
    if (node?.fileStatus === "READY" && node.image?.url) return node;
    if (node?.fileStatus === "FAILED") throw new Error(`Shopify file processing failed for ${fileId}.`);
    await sleep(Math.min(1000 * attempt, 5000));
  }
  throw new Error(`Timed out waiting for Shopify file to become READY: ${fileId}`);
}

async function commandContext(env, args) {
  const products = [];
  let after = null;
  while (true) {
    const data = await graphql(env, CONTEXT_QUERY, {
      blogFirst: 50,
      articleFirst: 10,
      productFirst: args.productPageSize,
      after,
    });
    products.push(...data.products.nodes);
    if (!data.products.pageInfo.hasNextPage) {
      console.log(JSON.stringify({ shop: data.shop, blogs: data.blogs.nodes, recentArticles: data.articles.nodes, products }, null, 2));
      return;
    }
    after = data.products.pageInfo.endCursor;
  }
}

async function commandUploadImages(env, args) {
  if (!args.input) throw new Error("upload-images requires --input manifest.json");
  const manifest = await readJson(args.input);
  const images = Array.isArray(manifest) ? manifest : (manifest.shopifyUploadManifest || manifest.images);
  if (!Array.isArray(images) || images.length === 0) {
    throw new Error("Image manifest must be an array, { images: [...] }, or fetch-wechat-article output with shopifyUploadManifest.");
  }
  for (const image of images) {
    image.path ||= image.download?.filePath;
    image.filename ||= image.download?.filename || (image.path ? path.basename(image.path) : null);
    image.mimeType ||= image.download?.mimeType;
    if (!image.path || !image.filename || !image.mimeType) {
      throw new Error("Each image manifest item must include path, filename, and mimeType. fetch-wechat-article.mjs outputs shopifyUploadManifest for this.");
    }
  }

  if (!args.execute) {
    console.log(JSON.stringify({ previewOnly: true, imageCount: images.length, filenames: images.map((image) => image.filename) }, null, 2));
    return;
  }

  const staged = await graphql(env, STAGED_UPLOADS_CREATE, {
    input: images.map((image) => ({
      filename: image.filename || path.basename(image.path),
      mimeType: image.mimeType,
      resource: "IMAGE",
      httpMethod: "POST",
    })),
  });
  assertNoUserErrors("stagedUploadsCreate", staged.stagedUploadsCreate.userErrors);

  for (let i = 0; i < images.length; i += 1) {
    const target = staged.stagedUploadsCreate.stagedTargets[i];
    const image = images[i];
    const bytes = await readFile(image.path);
    const form = new FormData();
    for (const parameter of target.parameters) form.append(parameter.name, parameter.value);
    form.append("file", new Blob([bytes], { type: image.mimeType }), image.filename || path.basename(image.path));
    const upload = await fetch(target.url, { method: "POST", body: form });
    if (!upload.ok) throw new Error(`Upload failed for ${image.path}: HTTP ${upload.status}`);
    image.originalSource = target.resourceUrl;
  }

  const created = await graphql(env, FILE_CREATE, {
    files: images.map((image) => ({
      originalSource: image.originalSource,
      contentType: "IMAGE",
      alt: image.alt || "",
      filename: image.filename || path.basename(image.path),
    })),
  });
  assertNoUserErrors("fileCreate", created.fileCreate.userErrors);
  const readyFiles = [];
  for (let i = 0; i < created.fileCreate.files.length; i += 1) {
    const file = created.fileCreate.files[i];
    const ready = await pollMediaImage(env, file.id);
    readyFiles.push({
      sourcePath: images[i].path,
      sourceUrl: images[i].sourceUrl,
      filename: images[i].filename,
      id: ready.id,
      alt: ready.alt,
      url: ready.image.url,
      width: ready.image.width,
      height: ready.image.height,
    });
  }
  console.log(JSON.stringify({ files: readyFiles }, null, 2));
}

function normalizeArticleInput(draft, includeBlogId = false) {
  const image = draft.image ? { ...draft.image } : undefined;
  if (image?.alt && !image.altText) image.altText = image.alt;
  if (image) delete image.alt;
  const article = {
    ...(includeBlogId ? { blogId: draft.blogId } : {}),
    title: draft.title,
    author: typeof draft.author === "string" ? { name: draft.author } : draft.author,
    summary: draft.summary,
    body: draft.body,
    image,
    metafields: draft.metafields,
    isPublished: false,
  };
  for (const key of Object.keys(article)) if (article[key] === undefined) delete article[key];
  return article;
}

async function commandConnectionCheck(env) {
  const data = await graphql(env, CONNECTION_CHECK);
  console.log(JSON.stringify({ ok: true, shop: data.shop }, null, 2));
}

async function commandCreateDraft(env, args) {
  if (!args.input) throw new Error("create-draft requires --input draft.json");
  const draft = await readJson(args.input);
  if (!draft.blogId || !draft.title || !draft.body) throw new Error("Draft JSON must include blogId, title, and body.");
  if (args.requireImages) assertDraftImages(draft);

  const article = normalizeArticleInput(draft, true);

  if (!args.execute) {
    console.log(JSON.stringify({ previewOnly: true, article }, null, 2));
    return;
  }

  const created = await graphql(env, ARTICLE_CREATE, { article });
  assertNoUserErrors("articleCreate", created.articleCreate.userErrors);
  console.log(JSON.stringify(created.articleCreate.article, null, 2));
}

async function commandUpdateDraft(env, args) {
  if (!args.articleId) throw new Error("update-draft requires --article-id gid://shopify/Article/...");
  if (!args.input) throw new Error("update-draft requires --input draft.json");
  const draft = await readJson(args.input);
  if (!draft.title && !draft.body && !draft.summary && !draft.image && !draft.metafields) {
    throw new Error("Draft JSON must include at least one article field to update.");
  }
  if (args.requireImages) assertDraftImages(draft);

  const article = normalizeArticleInput(draft);

  if (!args.execute) {
    console.log(JSON.stringify({ previewOnly: true, articleId: args.articleId, article }, null, 2));
    return;
  }

  const updated = await graphql(env, ARTICLE_UPDATE, { id: args.articleId, article });
  assertNoUserErrors("articleUpdate", updated.articleUpdate.userErrors);
  console.log(JSON.stringify(updated.articleUpdate.article, null, 2));
}

async function commandVerify(env, args) {
  if (!args.articleId) throw new Error("verify requires --article-id gid://shopify/Article/...");
  const data = await graphql(env, ARTICLE_VERIFY, { id: args.articleId });
  console.log(JSON.stringify(data.article, null, 2));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.command === "init-env") {
    await initEnv(args);
    return;
  }
  if (args.command === "create-draft" && !args.execute) {
    await commandCreateDraft(null, args);
    return;
  }
  if (args.command === "update-draft" && !args.execute) {
    await commandUpdateDraft(null, args);
    return;
  }
  if (args.requireImages && (args.command === "create-draft" || args.command === "update-draft")) {
    const draft = await readJson(args.input);
    assertDraftImages(draft);
  }

  const env = await loadEnv(args.env);
  if (args.command === "connection-check") await commandConnectionCheck(env);
  if (args.command === "context") await commandContext(env, args);
  if (args.command === "upload-images") await commandUploadImages(env, args);
  if (args.command === "create-draft") await commandCreateDraft(env, args);
  if (args.command === "update-draft") await commandUpdateDraft(env, args);
  if (args.command === "verify") await commandVerify(env, args);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
