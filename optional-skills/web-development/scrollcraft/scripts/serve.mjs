#!/usr/bin/env node
/**
 * scrollcraft static server.
 *
 * A scrollcraft page cannot be verified from file://. The engine fetches each
 * clip as a Blob, and file:// fetches are blocked by CORS in every browser, so
 * the page silently falls back to posters and looks fine while proving nothing.
 * Serve it.
 *
 *   node serve.mjs --root builds/perkform --port 4500
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i > -1 && argv[i + 1] ? argv[i + 1] : d; };

const ROOT = path.resolve(arg("--root", "."));
const PORT = parseInt(arg("--port", "4500"), 10);

const TYPES = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".json": "application/json",
  ".mp4": "video/mp4", ".webm": "video/webm",
  ".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg",
  ".svg": "image/svg+xml", ".woff2": "font/woff2",
};

http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  let file = path.join(ROOT, url === "/" ? "/index.html" : url);

  // Refuse to serve outside the root even if the path walks up.
  if (!file.startsWith(ROOT)) { res.writeHead(403).end("forbidden"); return; }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!fs.existsSync(file)) { res.writeHead(404).end("not found"); return; }

  const ext = path.extname(file).toLowerCase();
  const stat = fs.statSync(file);
  res.writeHead(200, {
    "Content-Type": TYPES[ext] || "application/octet-stream",
    "Content-Length": stat.size,
    // No caching: verification loops re-shoot the same URLs after edits, and a
    // cached clip or stylesheet makes you screenshot the previous build.
    "Cache-Control": "no-store",
    "Accept-Ranges": "bytes",
  });
  fs.createReadStream(file).pipe(res);
}).listen(PORT, () => {
  console.log(`scrollcraft: ${ROOT}\n  http://localhost:${PORT}`);
});
