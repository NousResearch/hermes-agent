#!/usr/bin/env node

import { readFile } from "node:fs/promises";

function parseArgs(argv) {
  const args = { product: null, primaryDomain: null, heading: "Related product" };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--product") {
      args.product = value;
      i += 1;
    } else if (key === "--primary-domain") {
      args.primaryDomain = value;
      i += 1;
    } else if (key === "--heading") {
      args.heading = value;
      i += 1;
    }
  }
  if (!args.product) throw new Error("Use --product <product-json-file>.");
  return args;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function stripHtml(value) {
  return String(value || "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

function productUrl(product, primaryDomain) {
  if (product.onlineStoreUrl) return product.onlineStoreUrl;
  if (primaryDomain && product.handle) {
    const base = primaryDomain.startsWith("http") ? primaryDomain : `https://${primaryDomain}`;
    return `${base.replace(/\/$/, "")}/products/${encodeURIComponent(product.handle)}`;
  }
  return "";
}

function priceLabel(product) {
  const min = product.priceRangeV2?.minVariantPrice;
  const max = product.priceRangeV2?.maxVariantPrice;
  if (!min?.amount || !min?.currencyCode) return "";
  if (max?.amount && max.amount !== min.amount) {
    return `${min.currencyCode} ${min.amount} - ${max.amount}`;
  }
  return `${min.currencyCode} ${min.amount}`;
}

function productImage(product) {
  const featured = product.featuredMedia?.preview?.image;
  if (featured?.url) return featured;

  const mediaNodes = product.media?.nodes || [];
  for (const media of mediaNodes) {
    const image = media.image || media.preview?.image;
    if (media.mediaContentType === "IMAGE" && image?.url) return image;
  }

  return null;
}

function buildBlock(product, primaryDomain, heading) {
  const url = productUrl(product, primaryDomain);
  if (!url) throw new Error("Selected product has no onlineStoreUrl and no primary domain fallback.");

  const image = productImage(product);
  const description = stripHtml(product.description).slice(0, 220);
  const price = priceLabel(product);
  const imageHtml = image?.url
    ? `<a href="${escapeHtml(url)}"><img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.altText || product.title)}"></a>`
    : "";
  const priceHtml = price ? `<p><strong>${escapeHtml(price)}</strong></p>` : "";

  return `<section class="related-product" data-selofy-related-product="true" data-product-id="${escapeHtml(product.id || "")}">
  <h2>${escapeHtml(heading)}</h2>
  ${imageHtml}
  <h3><a href="${escapeHtml(url)}">${escapeHtml(product.title)}</a></h3>
  ${description ? `<p>${escapeHtml(description)}</p>` : ""}
  ${priceHtml}
  <p><a href="${escapeHtml(url)}">View ${escapeHtml(product.title)}</a></p>
</section>`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const product = JSON.parse(await readFile(args.product, "utf8"));
  console.log(buildBlock(product, args.primaryDomain, args.heading));
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
