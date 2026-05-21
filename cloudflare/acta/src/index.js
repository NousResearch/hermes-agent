const TEXT_ENCODER = new TextEncoder();
const COOKIE_NAME = "vesta_session";
const SESSION_DAYS = 30;

const SECURITY_HEADERS = {
  "Content-Type": "text/html; charset=utf-8",
  "Content-Security-Policy": "default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
  "X-Robots-Tag": "noindex, nofollow, noarchive",
  "Cache-Control": "no-store, no-transform",
};

const INTERACTIVE_DASHBOARD_HEADERS = {
  ...SECURITY_HEADERS,
  "Content-Security-Policy": "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};

const LOGIN_HEADERS = {
  ...SECURITY_HEADERS,
  "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
};

function headersForKey(key) {
  // The stable Acta dashboard is an app-like reader/inbox and needs a tiny
  // inline script for local read/unread state and swipe gestures. Signed detail
  // reports and archive pages remain scriptless by default.
  if (key === "public/index.html") return INTERACTIVE_DASHBOARD_HEADERS;
  return SECURITY_HEADERS;
}

function jsonResponse(payload, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      ...extraHeaders,
    },
  });
}

function base64url(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

async function hmac(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    TEXT_ENCODER.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return base64url(await crypto.subtle.sign("HMAC", key, TEXT_ENCODER.encode(message)));
}

function base64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

async function sha256Base64(value) {
  return base64(await crypto.subtle.digest("SHA-256", TEXT_ENCODER.encode(value)));
}

function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  let diff = a.length ^ b.length;
  const max = Math.max(a.length, b.length);
  for (let i = 0; i < max; i += 1) {
    diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }
  return diff === 0;
}

function cleanKey(rawKey) {
  const key = rawKey.replace(/^\/+/, "");
  const signedReport = /^r\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.html$/.test(key);
  const publicPage = /^public\/(?:[A-Za-z0-9._-]+\/)*[A-Za-z0-9._-]+\.html$/.test(key);
  if (!signedReport && !publicPage) return "";
  if (key.includes("..")) return "";
  return key;
}

function publicKeyForPath(pathname) {
  if (pathname === "/" || pathname === "") return "public/index.html";
  if (pathname === "/jobs" || pathname === "/jobs/") return "public/jobs/index.html";
  if (pathname === "/archive" || pathname === "/archive/") return "public/archive/index.html";
  const match = pathname.match(/^\/archive\/([0-9]{4}-[0-9]{2}-[0-9]{2})\/?$/);
  if (match) return `public/archive/${match[1]}.html`;
  return "";
}

function isPublicKey(key) {
  return key.startsWith("public/");
}

function safeDecodePath(value) {
  try {
    return decodeURIComponent(value);
  } catch (_) {
    return "";
  }
}

function publicBase(env) {
  const base = String(env.PUBLIC_BASE_URL || "").replace(/\/+$/, "");
  try {
    const parsed = new URL(base);
    if (parsed.protocol !== "https:" || parsed.hostname !== "acta.imperatr.com") return "";
  } catch (_) {
    return "";
  }
  return base;
}

async function publicUrl(env, key) {
  const base = publicBase(env);
  if (!base) return "";
  if (key === "public/index.html") return `${base}/`;
  if (key === "public/jobs/index.html") return `${base}/jobs`;
  if (key === "public/archive/index.html") return `${base}/archive`;
  const archiveMatch = key.match(/^public\/archive\/([0-9]{4}-[0-9]{2}-[0-9]{2})\.html$/);
  if (archiveMatch) return `${base}/archive/${archiveMatch[1]}`;
  return `${base}/${key}`;
}

async function signedUrl(env, key) {
  if (isPublicKey(key)) return publicUrl(env, key);
  const ttlDays = Number.parseInt(env.DEFAULT_TTL_DAYS || "90", 10);
  const ttlSeconds = Math.max(1, Math.min(Number.isFinite(ttlDays) ? ttlDays : 90, 365)) * 24 * 60 * 60;
  const exp = Math.floor(Date.now() / 1000) + ttlSeconds;
  const path = `/${key}`;
  const sig = await hmac(env.ACTA_SIGNING_SECRET, `${path}\n${exp}`);
  const base = publicBase(env);
  if (!base) return "";
  return `${base}${path}?exp=${exp}&sig=${sig}`;
}

async function verifySignature(env, url, key) {
  const expRaw = url.searchParams.get("exp") || "";
  const sig = url.searchParams.get("sig") || "";
  const exp = Number.parseInt(expRaw, 10);
  if (!Number.isFinite(exp) || exp < Math.floor(Date.now() / 1000)) return false;
  if (!sig || !/^[A-Za-z0-9_-]{32,}$/.test(sig)) return false;
  const expected = await hmac(env.ACTA_SIGNING_SECRET, `/${key}\n${exp}`);
  return timingSafeEqual(sig, expected);
}

async function requireUser(request, env) {
  if (!env.VESTA_DB) return null;
  const token = getCookie(request, COOKIE_NAME);
  if (!token) return null;
  const tokenHash = await sha256Base64(token);
  const user = await env.VESTA_DB.prepare(
    `SELECT users.id, users.email, users.forwarding_username, users.created_at, users.updated_at
     FROM sessions
     JOIN users ON users.id = sessions.user_id
     WHERE sessions.token_hash = ? AND sessions.expires_at > ?`,
  )
    .bind(tokenHash, new Date().toISOString())
    .first();
  return user || null;
}

function getCookie(request, name) {
  const cookie = request.headers.get("cookie") || "";
  for (const part of cookie.split(";")) {
    const [rawName, ...rest] = part.trim().split("=");
    if (rawName === name) return decodeURIComponent(rest.join("="));
  }
  return "";
}

function ssoCookie(token) {
  return `${COOKIE_NAME}=${encodeURIComponent(token)}; Path=/; Domain=.imperatr.com; HttpOnly; SameSite=Lax; Secure; Max-Age=${SESSION_DAYS * 24 * 60 * 60}`;
}

function clearSsoCookie() {
  return `${COOKIE_NAME}=; Path=/; Domain=.imperatr.com; HttpOnly; SameSite=Lax; Secure; Max-Age=0`;
}

function loginRedirect(url) {
  const next = `${url.pathname}${url.search}` || "/";
  return new Response("redirecting", { status: 302, headers: { Location: `/login?next=${encodeURIComponent(next)}`, "Cache-Control": "no-store" } });
}

function normalizeEmail(value) {
  return String(value || "").trim().toLowerCase();
}

function publicUser(user) {
  return { id: user.id, email: user.email, forwardingUsername: user.forwarding_username || null };
}

async function handleSalt(url, env) {
  if (!env.VESTA_DB) return jsonResponse({ error: "database not configured" }, 500);
  const email = normalizeEmail(url.searchParams.get("email"));
  const user = email ? await env.VESTA_DB.prepare("SELECT password_salt FROM users WHERE email = ?").bind(email).first() : null;
  if (!user) return jsonResponse({ error: "Email or password did not match." }, 404);
  return jsonResponse({ salt: user.password_salt });
}

async function handleLogin(request, env) {
  if (!env.VESTA_DB) return jsonResponse({ error: "database not configured" }, 500);
  const body = await request.json().catch(() => ({}));
  const email = normalizeEmail(body.email);
  const passwordVerifier = String(body.passwordVerifier || "");
  const user = await env.VESTA_DB.prepare("SELECT * FROM users WHERE email = ?").bind(email).first();
  if (!user || !(await timingSafeEqual(await sha256Base64(passwordVerifier), user.password_hash))) {
    return jsonResponse({ error: "Email or password did not match." }, 401);
  }
  const token = base64(crypto.getRandomValues(new Uint8Array(32)));
  const tokenHash = await sha256Base64(token);
  const now = new Date();
  const expires = new Date(now.getTime() + SESSION_DAYS * 24 * 60 * 60 * 1000);
  await env.VESTA_DB.prepare("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)")
    .bind(tokenHash, user.id, now.toISOString(), expires.toISOString())
    .run();
  return jsonResponse({ user: publicUser(user) }, 200, { "Set-Cookie": ssoCookie(token) });
}

async function handleLogout(request, env) {
  const token = getCookie(request, COOKIE_NAME);
  if (token && env.VESTA_DB) {
    await env.VESTA_DB.prepare("DELETE FROM sessions WHERE token_hash = ?").bind(await sha256Base64(token)).run();
  }
  return jsonResponse({ ok: true }, 200, { "Set-Cookie": clearSsoCookie() });
}

function renderLoginPage(url) {
  const next = url.searchParams.get("next") || "/";
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<title>Sign in to Acta</title>
<style>
:root{color-scheme:dark;--bg:#000;--panel:#070707;--line:#252525;--text:#fff;--muted:#9b9b9b;--accent:#f5a400;--mono:SFMono-Regular,ui-monospace,Menlo,monospace;--ui:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#000;color:var(--text);font:15px/1.45 var(--ui);padding:20px}.card{width:min(420px,100%);border:1px solid var(--line);background:var(--panel);padding:22px}.kicker{font:700 11px var(--mono);letter-spacing:.16em;color:var(--accent);text-transform:uppercase}h1{font:700 38px/1.02 Georgia,serif;letter-spacing:-.04em;margin:10px 0 8px}.lede{color:var(--muted);margin:0 0 18px}label{display:block;color:var(--muted);font:700 11px var(--mono);text-transform:uppercase;margin:12px 0 6px}input,button{width:100%;height:44px;border-radius:0}input{background:#000;border:1px solid var(--line);color:#fff;padding:0 12px;font:15px var(--ui)}button{margin-top:16px;border:0;background:var(--accent);color:#000;font:900 12px var(--mono);letter-spacing:.12em}button:disabled{opacity:.55}.msg{min-height:20px;margin-top:12px;color:#d05a4e;font:12px var(--mono)}
</style>
</head>
<body>
<form class="card" id="loginForm">
  <div class="kicker">ACTA / IMPERATR SSO</div>
  <h1>Sign in to Acta</h1>
  <p class="lede">Use your Vesta credentials. One login for the imperatr apps.</p>
  <label for="email">Email</label><input id="email" type="email" autocomplete="email" required autofocus>
  <label for="password">Password</label><input id="password" type="password" autocomplete="current-password" required>
  <button id="submit" type="submit">ENTER ACTA</button>
  <div class="msg" id="msg"></div>
</form>
<script>
const next=${JSON.stringify(next)};
const enc=new TextEncoder();
function b64(bytes){let s=''; for(const b of bytes)s+=String.fromCharCode(b); return btoa(s)}
function b64ToBytes(v){return Uint8Array.from(atob(v), c=>c.charCodeAt(0))}
async function verifier(password,saltB64){const key=await crypto.subtle.importKey('raw',enc.encode(password),'PBKDF2',false,['deriveBits']); const bits=await crypto.subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt:b64ToBytes(saltB64),iterations:210000},key,256); return b64(new Uint8Array(bits))}
document.getElementById('loginForm').addEventListener('submit',async ev=>{ev.preventDefault(); const msg=document.getElementById('msg'); const btn=document.getElementById('submit'); msg.textContent=''; btn.disabled=true; try{const email=document.getElementById('email').value.trim().toLowerCase(); const password=document.getElementById('password').value; const saltRes=await fetch('/auth/salt?email='+encodeURIComponent(email)); const salt=await saltRes.json(); if(!saltRes.ok) throw new Error(salt.error||'Sign in failed.'); const passwordVerifier=await verifier(password,salt.salt); const res=await fetch('/auth/login',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email,passwordVerifier})}); const body=await res.json(); if(!res.ok) throw new Error(body.error||'Sign in failed.'); location.href=next && next.startsWith('/') ? next : '/';}catch(e){msg.textContent=e.message||'Sign in failed.'; btn.disabled=false;}});
</script>
</body>
</html>`;
}

async function handleUpload(request, env, url) {
  if (!env.ACTA_UPLOAD_TOKEN || !env.ACTA_SIGNING_SECRET) {
    return jsonResponse({ error: "server not configured" }, 500);
  }
  const supplied = request.headers.get("X-Acta-Upload-Token") || "";
  if (!timingSafeEqual(supplied, env.ACTA_UPLOAD_TOKEN)) {
    return jsonResponse({ error: "unauthorized" }, 401);
  }
  const key = cleanKey(safeDecodePath(url.pathname.replace(/^\/__upload\//, "")));
  if (!key) return jsonResponse({ error: "invalid key" }, 400);
  const contentType = request.headers.get("Content-Type") || "";
  if (!contentType.toLowerCase().startsWith("text/html")) {
    return jsonResponse({ error: "content-type must be text/html" }, 415);
  }
  const maxBytes = 512 * 1024;
  const body = await request.arrayBuffer();
  if (body.byteLength > maxBytes) {
    return jsonResponse({ error: "artifact too large" }, 413);
  }
  await env.REPORTS.put(key, body, {
    httpMetadata: { contentType: "text/html; charset=utf-8" },
    customMetadata: { uploadedAt: new Date().toISOString() },
  });
  const urlForArtifact = await signedUrl(env, key);
  if (!urlForArtifact) return jsonResponse({ error: "server not configured" }, 500);
  return jsonResponse({ ok: true, key, url: urlForArtifact });
}

async function handleRead(request, env, url) {
  if (!env.ACTA_SIGNING_SECRET) return new Response("server not configured", { status: 500 });
  const publicKey = publicKeyForPath(url.pathname);
  const key = publicKey || cleanKey(safeDecodePath(url.pathname));
  if (!key) return new Response("not found", { status: 404 });
  if (!isPublicKey(key) && !(await verifySignature(env, url, key))) return new Response("forbidden", { status: 403 });
  const object = await env.REPORTS.get(key);
  if (!object) return new Response("not found", { status: 404 });
  return new Response(object.body, { status: 200, headers: headersForKey(key) });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/healthz") {
      return jsonResponse({ ok: true, service: "acta" });
    }
    if (request.method === "PUT" && url.pathname.startsWith("/__upload/")) {
      return handleUpload(request, env, url);
    }
    if (request.method === "GET" && url.pathname === "/login") {
      return new Response(renderLoginPage(url), { status: 200, headers: LOGIN_HEADERS });
    }
    if (request.method === "GET" && url.pathname === "/auth/salt") return handleSalt(url, env);
    if (request.method === "POST" && url.pathname === "/auth/login") return handleLogin(request, env);
    if (request.method === "POST" && url.pathname === "/auth/logout") return handleLogout(request, env);
    if (request.method === "GET") {
      const user = await requireUser(request, env);
      if (!user) return loginRedirect(url);
      return handleRead(request, env, url);
    }
    return new Response("method not allowed", { status: 405, headers: { Allow: "GET, PUT, POST" } });
  },
};
