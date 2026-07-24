#!/usr/bin/env node

import fs from "node:fs"
import path from "node:path"
import https from "node:https"
import http from "node:http"
import dns from "node:dns/promises"
import net from "node:net"
import { fileURLToPath } from "node:url"
import { createRequire } from "node:module"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REQUEST_TIMEOUT = 30000
const MAX_REDIRECTS = 5

function fail(message) {
  console.error(`ERROR: ${message}`)
  process.exit(1)
}

const args = parseArgs(process.argv.slice(2))
const storeUrl = args.store || fail("Missing required argument: --store <url>")
const outputDir = args.output || process.cwd()
const filter = args.filter || "all"
const overwrite = args.overwrite === "true"
const yes = args.yes === "true"
const dryRun = args["dry-run"] === "true"
const webp = args.webp || "false"
const rename = args.rename === "true"

const domain = extractDomain(storeUrl)
const baseUrl = `https://${domain}`
try {
  validateSafeUrl(baseUrl)
} catch (e) {
  fail(e.message)
}

console.log(`\n=== Shopify Product Images Downloader ===`)
console.log(`Store: ${domain}`)
console.log(`Filter: ${filter}`)
console.log(`Output: ${path.resolve(outputDir)}`)
console.log(`Overwrite: ${overwrite ? "yes" : "no (skip existing)"}`)
if (webp === "true") console.log(`Format: WebP (converted)`)
if (rename) console.log(`Naming: product-handle-N (smart rename)`)
if (dryRun) console.log(`Mode: dry-run (no files will be downloaded)`)
console.log("")

// ── Step 1: Verify Shopify store ──
console.log("Step 1: Verifying Shopify store...")
const isShopify = await verifyShopifyStore(baseUrl)
if (!isShopify) {
  fail(`"${domain}" does not appear to be a Shopify store. No Shopify indicators found (cdn.shopify.com, myshopify.com, Shopify.shop).`)
}
console.log("  Confirmed: Shopify store ✓\n")

// ── Step 2: Discover products ──
console.log("Step 2: Discovering products...")
let products
if (filter.startsWith("collection:")) {
  const handle = filter.slice("collection:".length).trim()
  console.log(`  Filter: collection "${handle}"`)
  products = await fetchCollectionProducts(handle)
} else if (filter.startsWith("product:")) {
  const handle = filter.slice("product:".length).trim()
  console.log(`  Filter: single product "${handle}"`)
  const product = await fetchSingleProduct(handle)
  products = product ? [product] : []
} else {
  console.log(`  Filter: all products`)
  products = await fetchAllProducts()
}

if (products.length === 0) {
  fail("No products found.")
}

console.log(`  Found ${products.length} products\n`)

// ── Step 3: Fetch full image data ──
console.log("Step 3: Fetching image data for all products...")
const imageTasks = products.map(async (p) => {
  const full = await fetchSingleProduct(p.handle)
  const images = (full && full.images) || []
  return { title: p.title, handle: p.handle, images }
})

const productImageData = await Promise.all(imageTasks)

let totalImages = 0
for (const p of productImageData) {
  totalImages += p.images.length
}
console.log(`  Total images: ${totalImages}\n`)

// ── Gibberish filename analysis ──
console.log("Analyzing image filenames...")
let totalGibberish = 0
for (const p of productImageData) {
  for (const img of p.images) {
    const name = path.basename(new URL(img.src).pathname)
    if (isGibberishFilename(name)) totalGibberish++
  }
}
if (totalGibberish > 0) {
  const pct = Math.round((totalGibberish / totalImages) * 100)
  console.log(`  ${totalGibberish} of ${totalImages} images (${pct}%) have auto-generated IDs as filenames`)
} else {
  console.log("  All images have meaningful filenames")
}
console.log("")

// ── Step 4: Preview ──
const storeFolderName = sanitizeFolderName(domain.replace(/^www\./, ""))
const rootFolder = path.resolve(outputDir, storeFolderName)
const formatLabel = webp === "true" ? "WebP" : "original format"

console.log("Preview:")
console.log(`  Root folder: ${rootFolder}`)
console.log(`  Products: ${products.length}`)
console.log(`  Images: ${totalImages}`)
if (totalGibberish > 0) {
  console.log(`  Gibberish filenames: ${totalGibberish} (${Math.round((totalGibberish / totalImages) * 100)}%)`)
}
console.log(`  Format: ${formatLabel}`)
if (rename) console.log(`  Rename: to product-handle-N pattern`)
console.log("")

if (!yes) {
  console.log("To proceed with download, run again with --yes flag.")
  console.log("The agent will confirm with the user before proceeding.\n")
  process.exit(0)
}

if (dryRun) {
  console.log("Dry-run complete. No files were downloaded.\n")
  process.exit(0)
}

// ── WebP setup ──
let sharp = null
if (webp === "true") {
  console.log("Setting up WebP conversion...")
  sharp = await ensureSharp()
  console.log("  Sharp ready ✓\n")
}

// ── Step 5: Download images ──
console.log("Downloading images...")

let downloaded = 0
let skipped = 0
let failed = 0
const failedItems = []

for (const product of productImageData) {
  const productFolder = path.join(rootFolder, sanitizeFolderName(product.title))
  fs.mkdirSync(productFolder, { recursive: true })
  let imageIndex = 0

  for (const img of product.images) {
    imageIndex++

    let filename
    if (rename) {
      const ext = sharp ? ".webp" : path.extname(new URL(img.src).pathname) || ".jpg"
      filename = `${product.handle}-${imageIndex}${ext}`
    } else {
      const originalName = path.basename(new URL(img.src).pathname)
      filename = sharp ? replaceExt(originalName, ".webp") : originalName
    }

    const filePath = path.join(productFolder, filename)

    if (fs.existsSync(filePath) && !overwrite) {
      skipped++
      continue
    }

    try {
      if (sharp) {
        await downloadAndConvertToWebp(img.src, filePath, sharp)
      } else {
        await downloadFile(img.src, filePath)
      }
      downloaded++
      process.stdout.write(".")
    } catch (err) {
      failed++
      failedItems.push({ product: product.title, url: img.src, error: err.message })
      process.stdout.write("x")
    }
  }
}

console.log("\n")

// ── Step 6: Summary ──
const summaryLines = [
  "=== Download Summary ===",
  `Store: ${domain}`,
  `Filter: ${filter}`,
  `Format: ${formatLabel}`,
  `Root folder: ${rootFolder}`,
  `Products processed: ${products.length}`,
  `Total images found: ${totalImages}`,
  `Downloaded: ${downloaded}`,
  `Skipped (already exist): ${skipped}`,
  `Failed: ${failed}`,
  `Rename: ${rename ? "yes (product-handle-N pattern)" : "no (original filenames)"}`,
  "",
]

if (failedItems.length > 0) {
  summaryLines.push("Failed items:")
  for (const item of failedItems) {
    summaryLines.push(`  - ${item.product}: ${item.url} (${item.error})`)
  }
  summaryLines.push("")
}

summaryLines.push(`Done at ${new Date().toISOString()}`)

const summary = summaryLines.join("\n")
console.log(summary)

const summaryFilename = `download-summary-${formatTimestamp()}.txt`
const summaryPath = path.resolve(outputDir, summaryFilename)
fs.writeFileSync(summaryPath, summary, "utf8")
console.log(`Summary saved: ${summaryPath}`)

// ════════════════════════════════════════════
// Helper functions
// ════════════════════════════════════════════

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

function isBlockedIp(value) {
  const ip = String(value || "").replace(/^\[|\]$/g, "").toLowerCase().split("%")[0]
  if (net.isIPv4(ip)) {
    const parts = ip.split(".").map(Number)
    const number = (((parts[0] * 256 + parts[1]) * 256 + parts[2]) * 256 + parts[3]) >>> 0
    return [[0, 0x00ffffff], [0x0a000000, 0x0affffff], [0x7f000000, 0x7fffffff], [0xa9fe0000, 0xa9feffff], [0xac100000, 0xac1fffff], [0xc0a80000, 0xc0a8ffff], [0xe0000000, 0xffffffff]].some(([start, end]) => number >= start && number <= end)
  }
  if (!net.isIPv6(ip)) return false
  return ip === "::" || ip === "::1" || ip.startsWith("fc") || ip.startsWith("fd") || ip.startsWith("fe8") || ip.startsWith("fe9") || ip.startsWith("fea") || ip.startsWith("feb") || (ip.startsWith("::ffff:") && isBlockedIp(ip.slice(7)))
}

async function resolveSafeAddress(value) {
  const parsed = new URL(validateSafeUrl(value))
  if (isBlockedIp(parsed.hostname)) throw new Error(`Access to private address "${parsed.hostname}" is blocked.`)
  if (net.isIP(parsed.hostname)) return { address: parsed.hostname, family: net.isIP(parsed.hostname) }
  const addresses = await dns.lookup(parsed.hostname, { all: true, verbatim: true })
  if (!addresses.length || addresses.some(({ address }) => isBlockedIp(address))) throw new Error(`DNS resolved to a private or local address: ${parsed.hostname}`)
  return addresses[0]
}

function validateDownloadUrl(value, targetDomain) {
  validateSafeUrl(value);
  const host = new URL(value).hostname.toLowerCase();
  const allowed =
    host === "cdn.shopify.com" ||
    host.endsWith(".shopifycdn.com") ||
    host.endsWith(".shopifycdn.net") ||
    host === targetDomain.toLowerCase() ||
    host.endsWith(".myshopify.com");
  if (!allowed) {
    throw new Error(`Forbidden CDN or download target: "${host}".`);
  }
}

function parseArgs(argv) {
  const result = {}
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2)
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : "true"
      result[key] = val
      if (val !== "true") i++
    }
  }
  return result
}

function extractDomain(url) {
  url = url.trim()
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return new URL(url).hostname
  }
  return url
}

function isGibberishFilename(filename) {
  const name = path.parse(filename).name
  return /^\d+$/.test(name)
}

async function verifyShopifyStore(url) {
  try {
    const html = await fetchText(url)
    const indicators = [
      "cdn.shopify.com",
      "myshopify.com",
      'Shopify.shop = "',
      "shopifycdn",
      "/cdn/shop/",
      "Shopify.theme",
    ]
    return indicators.some((ind) => html.includes(ind))
  } catch {
    return false
  }
}

async function fetchJSON(url) {
  const text = await fetchText(url)
  return JSON.parse(text)
}

async function fetchText(url, redirectDepth = 0) {
  const safeAddress = await resolveSafeAddress(url)
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http
    const req = client.get(url, { headers: { "User-Agent": "Shopify-Image-Downloader/1.0" }, lookup: (_hostname, _options, callback) => callback(null, safeAddress.address, safeAddress.family) }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        if (redirectDepth >= MAX_REDIRECTS) return reject(new Error(`Too many redirects while fetching ${url}`))
        const redirectUrl = new URL(res.headers.location, url).href
        res.resume()
        return resolve(fetchText(redirectUrl, redirectDepth + 1))
      }
      if (res.statusCode !== 200) {
        res.resume()
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`))
      }
      let data = ""
      res.on("data", (chunk) => (data += chunk))
      res.on("end", () => resolve(data))
    })
    req.setTimeout(REQUEST_TIMEOUT, () => {
      req.destroy(new Error(`Request timeout: ${url}`))
    })
    req.on("error", reject)
  })
}

async function fetchAllProducts() {
  const products = []
  let page = 1
  while (true) {
    const url = `${baseUrl}/products.json?limit=250&page=${page}`
    const json = await fetchJSON(url)
    if (!json.products || json.products.length === 0) break
    for (const p of json.products) {
      products.push({ id: p.id, title: p.title, handle: p.handle })
    }
    page++
  }
  return products
}

async function fetchCollectionProducts(handle) {
  const products = []
  let page = 1
  while (true) {
    const url = `${baseUrl}/collections/${handle}/products.json?limit=250&page=${page}`
    const json = await fetchJSON(url)
    if (!json.products || json.products.length === 0) break
    for (const p of json.products) {
      products.push({ id: p.id, title: p.title, handle: p.handle })
    }
    page++
  }
  return products
}

async function fetchSingleProduct(handle) {
  try {
    const url = `${baseUrl}/products/${handle}.json`
    const json = await fetchJSON(url)
    return json.product || null
  } catch {
    return null
  }
}

function sanitizeFolderName(name) {
  return name
    .replace(/[<>:"/\\|?*]/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\.+/, "")
    .replace(/\.+$/, "")
    .substring(0, 200) || "untitled"
}

function replaceExt(filename, newExt) {
  const dot = filename.lastIndexOf(".")
  if (dot === -1) return filename + newExt
  return filename.slice(0, dot) + newExt
}

async function downloadFile(url, dest, redirectDepth = 0) {
  validateDownloadUrl(url, domain)
  const safeAddress = await resolveSafeAddress(url)
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http
    const req = client.get(url, { headers: { "User-Agent": "Shopify-Image-Downloader/1.0" }, lookup: (_hostname, _options, callback) => callback(null, safeAddress.address, safeAddress.family) }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        if (redirectDepth >= MAX_REDIRECTS) return reject(new Error(`Too many redirects while downloading ${url}`))
        const redirectUrl = new URL(res.headers.location, url).href
        res.resume()
        return resolve(downloadFile(redirectUrl, dest, redirectDepth + 1))
      }
      if (res.statusCode !== 200) {
        res.resume()
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`))
      }
      const file = fs.createWriteStream(dest)
      res.pipe(file)
      file.on("finish", () => file.close(resolve))
      file.on("error", (err) => {
        file.close()
        reject(err)
      })
    })
    req.setTimeout(REQUEST_TIMEOUT, () => {
      req.destroy(new Error(`Download timeout: ${url}`))
    })
    req.on("error", reject)
  })
}

async function downloadAndConvertToWebp(url, dest, sharpInstance, redirectDepth = 0) {
  validateDownloadUrl(url, domain)
  const safeAddress = await resolveSafeAddress(url)
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http
    const req = client.get(url, { headers: { "User-Agent": "Shopify-Image-Downloader/1.0" }, lookup: (_hostname, _options, callback) => callback(null, safeAddress.address, safeAddress.family) }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        if (redirectDepth >= MAX_REDIRECTS) return reject(new Error(`Too many redirects while converting ${url}`))
        const redirectUrl = new URL(res.headers.location, url).href
        res.resume()
        return resolve(downloadAndConvertToWebp(redirectUrl, dest, sharpInstance, redirectDepth + 1))
      }
      if (res.statusCode !== 200) {
        res.resume()
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`))
      }
      const transformer = sharpInstance().webp({ quality: 82 })
      const file = fs.createWriteStream(dest)
      res.pipe(transformer).pipe(file)
      file.on("finish", () => file.close(resolve))
      file.on("error", (err) => {
        file.close()
        reject(err)
      })
      transformer.on("error", (err) => {
        file.close()
        reject(err)
      })
    })
    req.setTimeout(REQUEST_TIMEOUT, () => {
      req.destroy(new Error(`Download timeout: ${url}`))
    })
    req.on("error", reject)
  })
}

async function ensureSharp() {
  const require = createRequire(import.meta.url)
  try {
    require.resolve("sharp")
    return require("sharp")
  } catch {
    fail("WEBP_CONVERSION_UNAVAILABLE: install the pinned sharp prerequisite before using --webp; original-format downloads remain available without it.")
  }
}

function formatTimestamp() {
  const now = new Date()
  const y = now.getFullYear()
  const m = String(now.getMonth() + 1).padStart(2, "0")
  const d = String(now.getDate()).padStart(2, "0")
  const h = String(now.getHours()).padStart(2, "0")
  const min = String(now.getMinutes()).padStart(2, "0")
  return `${y}${m}${d}-${h}${min}`
}
