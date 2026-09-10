// Driver-side wrapper around tests-js/scripts/mock-server.ts (the desktop
// E2E suite's OpenAI-compatible mock): starts the server as a LIBRARY
// (importing, not executing, so the dev-launcher block never runs) and
// publishes its URL to a file for the shell driver to consume.
//
// Usage: node mock-provider.mjs <url-file>
//   Writes "<url-file>" with the base URL (http://127.0.0.1:<port>) once
//   the server is listening, then stays alive until stdin closes.

// @ts-check
import fs from 'node:fs';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const mockUrl = fileURLToPath(new URL('../../../tests-js/scripts/mock-server.ts', import.meta.url));

const { startMockServer } = await import(mockUrl);
const mock = await startMockServer();

const urlFile = process.argv[2];
if (!urlFile) {
  console.error('usage: node mock-provider.mjs <url-file>');
  process.exit(1);
}
fs.writeFileSync(urlFile, mock.url);
console.log(`[mock-provider] listening at ${mock.url}`);

// Live until killed. NOT gated on stdin closing: a backgrounded process's
// stdin is already at EOF, so an end-event exit would fire immediately
// and the server would die right after writing its URL.
await new Promise(() => {});
