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
const ROOT_REAL = fs.realpathSync(ROOT);
const PORT = parseInt(arg("--port", "4500"), 10);

const TYPES = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".json": "application/json",
  ".mp4": "video/mp4", ".webm": "video/webm",
  ".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg",
  ".svg": "image/svg+xml", ".woff2": "font/woff2",
};

const containedBy = (root, target) => {
  const relative = path.relative(root, target);
  return relative === "" || (
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
};

const fail = (res, status, message) => res.writeHead(status).end(message);

function sendFile(candidate, res) {
  fs.stat(candidate, (statError, candidateStat) => {
    if (statError) { fail(res, 404, "not found"); return; }
    const target = candidateStat.isDirectory() ? path.join(candidate, "index.html") : candidate;

    fs.realpath(target, (realpathError, realTarget) => {
      if (realpathError) { fail(res, 404, "not found"); return; }
      if (!containedBy(ROOT_REAL, realTarget)) { fail(res, 403, "forbidden"); return; }

      fs.open(realTarget, "r", (openError, fd) => {
        if (openError) { fail(res, 404, "not found"); return; }
        fs.fstat(fd, (fstatError, stat) => {
          if (fstatError || !stat.isFile()) {
            fs.close(fd, () => {});
            fail(res, 404, "not found");
            return;
          }

          const ext = path.extname(realTarget).toLowerCase();
          res.writeHead(200, {
            "Content-Type": TYPES[ext] || "application/octet-stream",
            "Content-Length": stat.size,
            // No caching: verification loops re-shoot the same URLs after edits, and a
            // cached clip or stylesheet makes you screenshot the previous build.
            "Cache-Control": "no-store",
            "Accept-Ranges": "bytes",
          });
          const stream = fs.createReadStream(realTarget, { fd, autoClose: true });
          stream.on("error", () => res.destroy());
          stream.pipe(res);
        });
      });
    });
  });
}

const server = http.createServer((req, res) => {
  let url;
  try {
    url = decodeURIComponent(req.url.split("?")[0]);
  } catch {
    fail(res, 400, "bad request");
    return;
  }
  // Node rejects embedded NUL before invoking filesystem callbacks.
  if (url.includes("\0")) { fail(res, 400, "bad request"); return; }
  const requested = url === "/" ? "index.html" : url.replace(/^[/\\]+/, "");
  const file = path.resolve(ROOT, requested);

  // Refuse to serve outside the root even if the path walks up.
  if (!containedBy(ROOT, file)) { fail(res, 403, "forbidden"); return; }
  sendFile(file, res);
});

server.listen(PORT, "127.0.0.1", () => {
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : PORT;
  console.log(`scrollcraft: ${ROOT}\n  http://127.0.0.1:${port}`);
});
