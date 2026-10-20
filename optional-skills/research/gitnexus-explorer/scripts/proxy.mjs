/**
 * GitNexus reverse proxy — serves production web UI + proxies /api/* to backend.
 * Zero dependencies, Node.js built-ins only.
 *
 * Usage: node proxy.mjs <dist-dir> [port]
 *   dist-dir: path to gitnexus-web/dist (production build)
 *   port: listen port (default: 8888)
 *
 * Environment:
 *   API_PORT:       GitNexus serve backend port (default: 4747)
 *   GITNEXUS_USER:  Enable Basic Auth (required if GITNEXUS_PASS is set)
 *   GITNEXUS_PASS:  Enable Basic Auth (required if GITNEXUS_USER is set)
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const API_PORT = parseInt(process.env.API_PORT || '4747');
const DIST_DIR = process.argv[2] || './dist';
const PORT = parseInt(process.argv[3] || '8888');
const AUTH_USER = process.env.GITNEXUS_USER || '';
const AUTH_PASS = process.env.GITNEXUS_PASS || '';
const AUTH_ENABLED = !!(AUTH_USER && AUTH_PASS);

function checkAuth(req, res) {
  const header = req.headers.authorization || '';
  const [scheme, encoded] = header.split(' ');
  if (scheme !== 'Basic') return false;

  const [user, pass] = Buffer.from(encoded, 'base64').toString().split(':');
  // Constant-time comparison to prevent timing attacks
  const userOk = crypto.timingSafeEqual
    ? crypto.timingSafeEqual(Buffer.from(user), Buffer.from(AUTH_USER))
    : user === AUTH_USER;
  const passOk = crypto.timingSafeEqual
    ? crypto.timingSafeEqual(Buffer.from(pass), Buffer.from(AUTH_PASS))
    : pass === AUTH_PASS;
  return userOk && passOk;
}

function sendAuthRequired(res) {
  res.writeHead(401, {
    'WWW-Authenticate': 'Basic realm="GitNexus Explorer", charset="UTF-8"',
    'Content-Type': 'text/plain',
  });
  res.end('Authentication required');
}

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
  if (AUTH_ENABLED && !checkAuth(req, res)) {
    sendAuthRequired(res);
    return;
  }
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
  if (AUTH_ENABLED) console.log(`  Auth:   Basic (user=${AUTH_USER})`);
  else console.log('  Auth:   none (set GITNEXUS_USER + GITNEXUS_PASS to enable)');
});
