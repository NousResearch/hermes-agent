// Driver-side wrapper around tests-js/scripts/mock-server.ts (the desktop
// E2E suite's OpenAI-compatible mock): starts the server as a LIBRARY
// (importing, not executing, so the dev-launcher block never runs) and
// publishes its URL to a file for the shell driver to consume.
//
// Usage: node mock-provider.mjs <url-file>
//   Writes "<url-file>" with the base URL (http://127.0.0.1:<port>) once
//   the server is listening, then stays alive until SIGINT or SIGTERM.

// @ts-check
import fs from 'node:fs';
import process from 'node:process';
import { startMockServer } from '../../../tests-js/scripts/mock-server.ts';

const urlFile = process.argv[2];
if (!urlFile) {
  console.error('usage: node mock-provider.mjs <url-file>');
  process.exit(1);
}
const mock = await startMockServer();
fs.writeFileSync(urlFile, mock.url);
console.log(`[mock-provider] listening at ${mock.url}`);

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, async () => {
    await mock.close();
    process.exit(0);
  });
}

// A background shell starts with stdin at EOF; signals own this lifetime.
await new Promise(() => {});
