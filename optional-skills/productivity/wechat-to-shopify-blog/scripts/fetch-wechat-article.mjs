#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fetchPublic } from "./lib/public-fetch.mjs";

function parseArgs(argv) {
  const args = {
    url: null,
    downloadImages: false,
    outputDir: null,
    output: null,
    maxImages: 80,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--url") {
      args.url = value;
      i += 1;
    } else if (key === "--download-images") {
      args.downloadImages = true;
    } else if (key === "--output-dir") {
      args.outputDir = value;
      i += 1;
    } else if (key === "--output") {
      args.output = value;
      i += 1;
    } else if (key === "--max-images") {
      args.maxImages = Number(value);
      i += 1;
    }
  }

  if (!args.url) {
    throw new Error("Usage: node fetch-wechat-article.mjs --url <mp.weixin.qq.com URL> [--download-images --output-dir <temp-dir>] [--output article.json]");
  }
  try {
    validateWeChatUrl(args.url);
    const hostname = new URL(args.url).hostname.toLowerCase();
    if (hostname !== "mp.weixin.qq.com") {
      throw new Error(`Only official WeChat URLs (mp.weixin.qq.com) are allowed. Got "${hostname}".`);
    }
  } catch (e) {
    throw new Error(`CLI argument validation failed: ${e.message}`);
  }
  if (!Number.isInteger(args.maxImages) || args.maxImages < 1 || args.maxImages > 250) {
    throw new Error("--max-images must be an integer from 1 to 250.");
  }

  return args;
}

function validateWeChatUrl(value) {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Invalid protocol: "${parsed.protocol}".`);
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
      throw new Error(`SSRF blocked host: "${hostname}".`);
    }
    const isAllowed =
      hostname === "mp.weixin.qq.com" ||
      hostname === "qpic.cn" ||
      hostname.endsWith(".qpic.cn") ||
      hostname.endsWith(".mp.weixin.qq.com");
    if (!isAllowed) {
      throw new Error(`External domain "${hostname}" is blocked.`);
    }
    return parsed.href;
  } catch (err) {
    throw new Error(`Invalid or unsafe WeChat URL "${value}": ${err.message}`);
  }
}

function decodeEntities(value = "") {
  const named = {
    amp: "&",
    lt: "<",
    gt: ">",
    quot: '"',
    apos: "'",
    nbsp: " ",
  };
  return value
    .replace(/\\x([0-9a-fA-F]{2})/g, (_, hex) => String.fromCharCode(Number.parseInt(hex, 16)))
    .replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, num) => String.fromCodePoint(Number(num)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&([a-zA-Z]+);/g, (all, name) => named[name] ?? all)
    .trim();
}

function normalizeUrl(value, baseUrl) {
  if (!value) return "";
  const decoded = decodeEntities(value).replace(/^\/\//, "https://");
  try {
    return new URL(decoded, baseUrl).toString();
  } catch {
    return "";
  }
}

function attr(tag, name) {
  const pattern = new RegExp(`${name}\\s*=\\s*["']([^"']*)["']`, "i");
  return tag.match(pattern)?.[1] || "";
}

function jsVar(html, name) {
  const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match =
    html.match(new RegExp(`(?:var\\s+)?${escapedName}\\s*=\\s*htmlDecode\\(["']([\\s\\S]*?)["']\\)\\s*;`)) ||
    html.match(new RegExp(`(?:var\\s+)?${escapedName}\\s*=\\s*["']([^"']*)["']\\s*;`));
  return match ? decodeEntities(match[1]) : "";
}

function elementTextById(html, id) {
  const match = html.match(new RegExp(`<[^>]+id=["']${id}["'][^>]*>([\\s\\S]*?)<\\/[^>]+>`, "i"));
  if (!match) return "";
  return decodeEntities(match[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
}

function meta(html, key) {
  const patterns = [
    new RegExp(`<meta[^>]+property=["']${key}["'][^>]+content=["']([^"']*)["'][^>]*>`, "i"),
    new RegExp(`<meta[^>]+content=["']([^"']*)["'][^>]+property=["']${key}["'][^>]*>`, "i"),
    new RegExp(`<meta[^>]+name=["']${key}["'][^>]+content=["']([^"']*)["'][^>]*>`, "i"),
    new RegExp(`<meta[^>]+content=["']([^"']*)["'][^>]+name=["']${key}["'][^>]*>`, "i"),
  ];
  for (const pattern of patterns) {
    const match = html.match(pattern);
    if (match) return decodeEntities(match[1]);
  }
  return "";
}

function extractContentHtml(html) {
  const match = html.match(/<div[^>]+id=["']js_content["'][^>]*>([\s\S]*?)<\/div>\s*<script/i);
  if (match) return match[1];
  const fallback = html.match(/<div[^>]+id=["']js_content["'][^>]*>([\s\S]*?)<\/div>/i);
  return fallback?.[1] || "";
}

function toTextBlocks(contentHtml) {
  return contentHtml
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|section|div|h[1-6]|li)>/gi, "\n")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, "")
    .split(/\n+/)
    .map((line) => decodeEntities(line.replace(/\s+/g, " ")))
    .filter((line) => line.length > 0);
}

function nearbyText(contentHtml, index) {
  const before = contentHtml.slice(Math.max(0, index - 800), index);
  const after = contentHtml.slice(index, Math.min(contentHtml.length, index + 800));
  return toTextBlocks(`${before}\n${after}`).slice(0, 4).join(" ");
}

function extractImages(contentHtml, pageUrl, maxImages) {
  const seen = new Set();
  const images = [];
  const regex = /<img\b[^>]*>/gi;
  let match;

  while ((match = regex.exec(contentHtml)) && images.length < maxImages) {
    const tag = match[0];
    const rawUrl =
      attr(tag, "data-src") ||
      attr(tag, "data-original") ||
      attr(tag, "data-backsrc") ||
      attr(tag, "src");
    const url = normalizeUrl(rawUrl, pageUrl);
    if (!url || seen.has(url)) continue;
    seen.add(url);
    images.push({
      index: images.length + 1,
      url,
      alt: decodeEntities(attr(tag, "alt")),
      width: attr(tag, "data-w") || attr(tag, "width"),
      height: attr(tag, "data-ratio") || attr(tag, "height"),
      nearbyText: nearbyText(contentHtml, match.index),
    });
  }

  return images;
}

function extensionFromMime(mime) {
  if (mime.includes("jpeg")) return "jpg";
  if (mime.includes("png")) return "png";
  if (mime.includes("gif")) return "gif";
  if (mime.includes("webp")) return "webp";
  return "bin";
}

async function downloadImage(image, outputDir) {
  try {
    validateWeChatUrl(image.url);
  } catch (e) {
    throw new Error(`Image download blocked: ${e.message}`);
  }
  const response = await fetchPublic(image.url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      Referer: "https://mp.weixin.qq.com/",
    },
  }, { allowedHosts: ["qpic.cn", "mp.weixin.qq.com"], timeoutMs: 30000 });
  if (!response.ok) throw new Error(`Image ${image.index} failed with HTTP ${response.status}`);
  const mimeType = response.headers.get("content-type")?.split(";")[0] || "application/octet-stream";
  const ext = extensionFromMime(mimeType);
  const filename = `wechat-image-${String(image.index).padStart(2, "0")}.${ext}`;
  const filePath = path.join(outputDir, filename);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(filePath, bytes);
  return { filePath, filename, mimeType, bytes: bytes.length };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const response = await fetchPublic(args.url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
  }, { allowedHosts: ["mp.weixin.qq.com"], timeoutMs: 30000 });
  if (!response.ok) throw new Error(`WeChat request failed with HTTP ${response.status}`);

  const html = await response.text();
  const contentHtml = extractContentHtml(html);
  if (!contentHtml) throw new Error("Could not find WeChat article body container #js_content.");

  const images = extractImages(contentHtml, args.url, args.maxImages);
  const coverImage =
    normalizeUrl(jsVar(html, "msg_cdn_url"), args.url) ||
    normalizeUrl(jsVar(html, "msg_cover"), args.url) ||
    normalizeUrl(meta(html, "og:image"), args.url) ||
    normalizeUrl(meta(html, "twitter:image"), args.url);

  let outputDir = null;
  if (args.downloadImages) {
    outputDir = args.outputDir || path.join(tmpdir(), `wechat-article-${Date.now()}`);
    await mkdir(outputDir, { recursive: true });
    for (const image of images) {
      try {
        image.download = await downloadImage(image, outputDir);
      } catch (error) {
        image.downloadError = error.message;
      }
    }
  }

  const result = {
        sourceUrl: args.url,
        fetchedAt: new Date().toISOString(),
        title: elementTextById(html, "activity-name") || jsVar(html, "msg_title") || meta(html, "og:title"),
        description:
          jsVar(html, "msg_desc") ||
          meta(html, "og:description") ||
          meta(html, "twitter:description") ||
          meta(html, "description"),
        author: elementTextById(html, "js_author_name") || jsVar(html, "msg_author"),
        accountName: jsVar(html, "nickname"),
        publishDate: elementTextById(html, "publish_time") || jsVar(html, "ct"),
        coverImage,
        imageCount: images.length,
        images,
        shopifyUploadManifest: images
          .filter((image) => image.download && !image.downloadError)
          .map((image) => ({
            path: image.download.filePath,
            filename: image.download.filename,
            mimeType: image.download.mimeType,
            alt: image.alt || image.nearbyText || `WeChat article image ${image.index}`,
            sourceUrl: image.url,
            nearbyText: image.nearbyText,
          })),
        textBlocks: toTextBlocks(contentHtml),
        bodyHtml: contentHtml,
        outputDir,
  };
  const output = JSON.stringify(result, null, 2);
  if (args.output) {
    await mkdir(path.dirname(path.resolve(args.output)), { recursive: true });
    await writeFile(args.output, output, "utf8");
    console.log(JSON.stringify({ ok: true, output: args.output, imageCount: images.length }, null, 2));
    return;
  }
  console.log(output);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
