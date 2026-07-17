#!/usr/bin/env node

import { loadShopifyEnv, shopifyCliGraphql } from "./lib/shopify-helpers.mjs";

function parseArgs(argv) {
  const args = {
    env: "skill-hub.env",
    articleId: null,
    productPageSize: 50,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--env") {
      args.env = value;
      i += 1;
    } else if (key === "--article-id") {
      args.articleId = value;
      i += 1;
    } else if (key === "--product-page-size") {
      args.productPageSize = Number(value);
      i += 1;
    }
  }

  if (!Number.isInteger(args.productPageSize) || args.productPageSize < 1 || args.productPageSize > 250) {
    throw new Error("--product-page-size must be an integer from 1 to 250.");
  }

  return args;
}

const loadEnv = loadShopifyEnv;
const graphql = shopifyCliGraphql;

function printHelp() {
  console.log("Usage: node shopify-context.mjs --env skill-hub.env [--article-id gid://shopify/Article/...] [--product-page-size 1-250]");
}

function validateSafeUrl(value) {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Invalid protocol: "${parsed.protocol}". Only HTTP and HTTPS are allowed.`);
    }
    const hostname = parsed.hostname.toLowerCase();
    if (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "0.0.0.0" ||
      hostname.startsWith("10.") ||
      hostname.startsWith("192.168.") ||
      (hostname.startsWith("172.") &&
        Number(hostname.split(".")[1]) >= 16 &&
        Number(hostname.split(".")[1]) <= 31)
    ) {
      throw new Error(`Access to private address "${hostname}" is blocked.`);
    }
    return parsed.href;
  } catch (err) {
    throw new Error(`Invalid or unsafe URL "${value}": ${err.message}`);
  }
}

async function fetchHomepageMeta(url) {
  if (!url) return null;
  try {
    validateSafeUrl(url);
  } catch (e) {
    return { url, ok: false, error: `Request blocked: ${e.message}` };
  }
  try {
    const response = await fetch(url, { redirect: "follow" });
    if (!response.ok) return { url, ok: false, status: response.status };
    const html = await response.text();
    const title = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]?.replace(/\s+/g, " ").trim() || "";
    const description =
      html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']*)["'][^>]*>/i)?.[1]?.trim() ||
      html.match(/<meta[^>]+content=["']([^"']*)["'][^>]+name=["']description["'][^>]*>/i)?.[1]?.trim() ||
      "";
    return { url, ok: true, title, description };
  } catch (error) {
    return { url, ok: false, error: error.message };
  }
}

const OVERVIEW_QUERY = `#graphql
query BrandVoiceOverview($blogFirst: Int!, $articleFirst: Int!, $articleId: ID!, $includeArticle: Boolean!, $productFirst: Int!, $productAfter: String) {
  shop {
    name
    description
    myshopifyDomain
    primaryDomain { host url }
  }
  blogs(first: $blogFirst) {
    nodes { id title handle }
  }
  articles(first: $articleFirst, sortKey: PUBLISHED_AT, reverse: true) {
    nodes {
      id
      title
      handle
      summary
      publishedAt
      author { name }
      blog { id title handle }
    }
  }
  article(id: $articleId) @include(if: $includeArticle) {
    id
    title
    handle
    body
    summary
    publishedAt
    author { name }
    image { altText url }
    blog { id title handle }
  }
  products(first: $productFirst, after: $productAfter, sortKey: UPDATED_AT, reverse: true) {
    nodes {
      id
      title
      handle
      description
      productType
      vendor
      status
      tags
      seo { title description }
      collections(first: 5) { nodes { id title handle } }
      onlineStoreUrl
      featuredMedia {
        preview { image { url altText width height } }
      }
      media(first: 5) {
        nodes {
          alt
          mediaContentType
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

const PRODUCTS_QUERY = `#graphql
query ProductPage($first: Int!, $after: String) {
  products(first: $first, after: $after, sortKey: UPDATED_AT, reverse: true) {
    nodes {
      id
      title
      handle
      description
      productType
      vendor
      status
      tags
      seo { title description }
      collections(first: 5) { nodes { id title handle } }
      onlineStoreUrl
      featuredMedia {
        preview { image { url altText width height } }
      }
      media(first: 5) {
        nodes {
          alt
          mediaContentType
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

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv.includes("--help") || argv.includes("-h")) {
    printHelp();
    return;
  }
  const args = parseArgs(argv);
  const env = await loadEnv(args.env);

  const overview = await graphql(env, OVERVIEW_QUERY, {
    blogFirst: 50,
    articleFirst: 10,
    articleId: args.articleId || "gid://shopify/Article/0",
    includeArticle: Boolean(args.articleId),
    productFirst: args.productPageSize,
    productAfter: null,
  });

  const products = [...(overview.products?.nodes || [])];
  let after = overview.products?.pageInfo?.endCursor || null;
  while (after) {
    const page = await graphql(env, PRODUCTS_QUERY, { first: args.productPageSize, after });
    products.push(...page.products.nodes);
    after = page.products.pageInfo?.hasNextPage ? page.products.pageInfo.endCursor : null;
  }

  const homepageUrl =
    overview.shop.primaryDomain?.url ||
    (overview.shop.primaryDomain?.host ? `https://${overview.shop.primaryDomain.host}` : null);
  const homepageMeta = await fetchHomepageMeta(homepageUrl);

  console.log(
    JSON.stringify(
      {
        shop: overview.shop,
        homepageMeta,
        blogs: overview.blogs.nodes,
        recentArticles: overview.articles.nodes,
        products,
        article: args.articleId ? overview.article : null,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
