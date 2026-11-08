/**
 * GitNexus reverse proxy — serves production web UI + proxies /api/* to backend.
 * Zero dependencies, Node.js built-ins only.
 *
 * Usage: node proxy.mjs <dist-dir> [port]
 *   dist-dir: path to gitnexus-web/dist (production build)
 *   port: listen port (default: 8888)
 *
 * Environment:
 *   API_PORT: GitNexus serve backend port (default: 4747)
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const API_PORT = parseInt(process.env.API_PORT || '4747');
const DIST_DIR = process.argv[2] || './dist';
const PORT = parseInt(process.argv[3] || '8888');

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.wasm': 'application/wasm',
  '.ttf': 'font/ttf',
  '.map': 'application/json',
};

function proxyToApi(req, res) {
  const opts = {
    hostname: '127.0.0.1',
    port: API_PORT,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: `127.0.0.1:${API_PORT}` },
  };
  const proxy = http.request(opts, (upstream) => {
    res.writeHead(upstream.statusCode, upstream.headers);
    upstream.pipe(res, { end: true });
  });
  proxy.on('error', () => {
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('GitNexus backend unavailable — is `npx gitnexus serve` running?');
  });
  req.pipe(proxy, { end: true });
}

function serveStatic(req, res) {
  const urlPath = req.url.split('?')[0];
  let filePath = path.join(DIST_DIR, urlPath === '/' ? 'index.html' : urlPath);

  // SPA fallback: if file doesn't exist and isn't a static asset, serve index.html
  if (!fs.existsSync(filePath) && !path.extname(filePath)) {
    filePath = path.join(DIST_DIR, 'index.html');
  }

  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, {
      'Content-Type': mime,
      'Cache-Control': ext === '.html' ? 'no-cache' : 'public, max-age=86400',
    });
    res.end(data);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not found');
  }
}

const server = http.createServer((req, res) => {
  if (req.url.startsWith('/api')) {
    proxyToApi(req, res);
  } else {
    serveStatic(req, res);
  }
});

server.listen(PORT, () => {
  console.log(`GitNexus proxy listening on http://localhost:${PORT}`);
  console.log(`  Web UI: http://localhost:${PORT}/`);
  console.log(`  API:    http://localhost:${PORT}/api/repos`);
  console.log(`  Backend: http://127.0.0.1:${API_PORT}`);
});
