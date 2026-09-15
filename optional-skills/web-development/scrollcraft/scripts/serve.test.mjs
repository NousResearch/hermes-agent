import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { afterEach, test } from "node:test";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const children = new Set();

afterEach(() => {
  for (const child of children) child.kill();
  children.clear();
});

async function unusedPort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function startServer(root) {
  const port = await unusedPort();
  const child = spawn(process.execPath, [path.join(HERE, "serve.mjs"), "--root", root, "--port", String(port)], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  children.add(child);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("server did not start")), 5_000);
    child.once("exit", (code) => reject(new Error(`server exited with ${code}`)));
    child.stdout.on("data", (chunk) => {
      if (chunk.toString().includes(`:${port}`)) {
        clearTimeout(timer);
        resolve();
      }
    });
  });
  return { child, port };
}

function request(host, port, requestPath) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host, port, path: requestPath }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => resolve({ status: res.statusCode, body }));
    });
    req.setTimeout(2_000, () => req.destroy(new Error("request timed out")));
    req.on("error", reject);
  });
}

test("server contains decoded traversal within the configured root", { timeout: 10_000 }, async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "scrollcraft-serve-"));
  const root = path.join(base, "site");
  const sibling = path.join(base, "site-private");
  fs.mkdirSync(root);
  fs.mkdirSync(sibling);
  fs.writeFileSync(path.join(root, "index.html"), "public");
  fs.writeFileSync(path.join(sibling, "secret.txt"), "secret");

  const { port } = await startServer(root);
  assert.deepEqual(await request("127.0.0.1", port, "/"), { status: 200, body: "public" });
  const escaped = await request("127.0.0.1", port, "/%2e%2e/site-private/secret.txt");
  assert.equal(escaped.status, 403);
  assert.equal(escaped.body, "forbidden");
});

test("server resolves links before enforcing root containment", { timeout: 10_000 }, async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "scrollcraft-links-"));
  const root = path.join(base, "site");
  const assets = path.join(root, "assets");
  const outside = path.join(base, "private");
  fs.mkdirSync(assets, { recursive: true });
  fs.mkdirSync(outside);
  fs.writeFileSync(path.join(root, "index.html"), "public");
  fs.writeFileSync(path.join(assets, "allowed.txt"), "allowed");
  fs.writeFileSync(path.join(outside, "secret.txt"), "secret");
  const linkType = process.platform === "win32" ? "junction" : "dir";
  fs.symlinkSync(assets, path.join(root, "inside-link"), linkType);
  fs.symlinkSync(outside, path.join(root, "outside-link"), linkType);

  const { port } = await startServer(root);
  assert.deepEqual(await request("127.0.0.1", port, "/inside-link/allowed.txt"), {
    status: 200,
    body: "allowed",
  });
  const escaped = await request("127.0.0.1", port, "/outside-link/secret.txt");
  assert.equal(escaped.status, 403);
  assert.equal(escaped.body, "forbidden");
});

test("invalid URL paths return 400 without stopping the server", { timeout: 10_000 }, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scrollcraft-url-"));
  fs.writeFileSync(path.join(root, "index.html"), "public");
  const { port } = await startServer(root);

  for (const requestPath of ["/%ZZ", "/%00"]) {
    const malformed = await request("127.0.0.1", port, requestPath);
    assert.equal(malformed.status, 400);
    assert.equal(malformed.body, "bad request");
  }
  assert.equal((await request("127.0.0.1", port, "/")).status, 200);
});

test("server accepts loopback connections only", { timeout: 10_000 }, async (t) => {
  const external = Object.values(os.networkInterfaces())
    .flat()
    .find((address) => address?.family === "IPv4" && !address.internal)?.address;
  if (!external) return t.skip("host has no non-loopback IPv4 interface");

  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scrollcraft-bind-"));
  fs.writeFileSync(path.join(root, "index.html"), "public");
  const { port } = await startServer(root);

  assert.equal((await request("127.0.0.1", port, "/")).status, 200);
  await assert.rejects(request(external, port, "/"));
});
