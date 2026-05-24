import test from "node:test";
import assert from "node:assert/strict";

import worker from "../src/index.js";

function base64(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)));
}

async function sha256Base64(value) {
  const data = new TextEncoder().encode(value);
  return base64(await crypto.subtle.digest("SHA-256", data));
}

function makeReports(entries = {}) {
  return {
    async get(key) {
      if (!(key in entries)) return null;
      return { body: entries[key] };
    },
    async put() {},
  };
}

function makeVestaDb({ token = "valid-token", user = { id: "u1", email: "p@example.com", forwarding_username: "p" } } = {}) {
  return {
    async tokenHash() {
      return sha256Base64(token);
    },
    prepare(sql) {
      return {
        bind: (...args) => ({
          async first() {
            if (sql.includes("FROM sessions") && sql.includes("JOIN users")) {
              const suppliedHash = args[0];
              if (suppliedHash === await sha256Base64(token)) return user;
              return null;
            }
            if (sql.includes("SELECT password_salt FROM users")) {
              return { password_salt: "salt" };
            }
            if (sql.includes("SELECT * FROM users WHERE email")) {
              return null;
            }
            return null;
          },
          async run() {
            return { success: true };
          },
        }),
      };
    },
  };
}

function makeEnv(overrides = {}) {
  return {
    ACTA_SIGNING_SECRET: "test-secret",
    ACTA_UPLOAD_TOKEN: "upload-token",
    PUBLIC_BASE_URL: "https://acta.imperatr.com",
    REPORTS: makeReports({
      "public/index.html": "<html><body>Acta Private Feed</body></html>",
      "public/outputs/index.html": "<html><body>Acta Outputs</body></html>",
      "public/outputs/pe-principal-automation-roadmap.html": "<html><body><script>window.ok=true</script>PE OS</body></html>",
    }),
    VESTA_DB: makeVestaDb(),
    ...overrides,
  };
}

test("Acta redirects unauthenticated dashboard requests to login", async () => {
  const response = await worker.fetch(new Request("https://acta.imperatr.com/"), makeEnv());

  assert.equal(response.status, 302);
  assert.equal(response.headers.get("location"), "/login?next=%2F");
});

test("Acta serves dashboard when a valid imperatr SSO vesta_session cookie is present", async () => {
  const response = await worker.fetch(
    new Request("https://acta.imperatr.com/", { headers: { cookie: "vesta_session=valid-token" } }),
    makeEnv(),
  );

  assert.equal(response.status, 200);
  assert.equal(await response.text(), "<html><body>Acta Private Feed</body></html>");
});

test("Acta login page is reachable without an existing SSO cookie", async () => {
  const response = await worker.fetch(new Request("https://acta.imperatr.com/login"), makeEnv());
  const html = await response.text();

  assert.equal(response.status, 200);
  assert.match(html, /Sign in to Acta/);
  assert.match(html, /\/auth\/login/);
});


test("Acta serves Outputs as a first-class authenticated module route", async () => {
  const response = await worker.fetch(
    new Request("https://acta.imperatr.com/outputs", { headers: { cookie: "vesta_session=valid-token" } }),
    makeEnv(),
  );

  assert.equal(response.status, 200);
  assert.equal(await response.text(), "<html><body>Acta Outputs</body></html>");
});

test("Acta serves output slugs with interactive CSP so artifact JavaScript works", async () => {
  const response = await worker.fetch(
    new Request("https://acta.imperatr.com/outputs/pe-principal-automation-roadmap", {
      headers: { cookie: "vesta_session=valid-token" },
    }),
    makeEnv(),
  );
  const body = await response.text();

  assert.equal(response.status, 200);
  assert.match(body, /PE OS/);
  assert.match(response.headers.get("content-security-policy"), /script-src 'unsafe-inline'/);
});
