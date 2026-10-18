// Read-only source checkpoint adapter. Tooling is resolved from the CI checkout,
// never from the historical installation. The standalone driver owns the app.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: {
  root: { type: 'string' }, home: { type: 'string' },
  'user-data': { type: 'string' }, out: { type: 'string' },
  phase: { type: 'string' }, 'expect-commit': { type: 'string' },
  desktop: { type: 'string' }, method: { type: 'string' },
  'mock-url': { type: 'string' },
} });
for (const key of ['root', 'home', 'user-data', 'out', 'phase', 'expect-commit', 'desktop', 'method']) {
  if (!values[key]) throw new Error(`--${key} is required`);
}
if (!['old', 'new'].includes(values.phase)) throw new Error('expected --phase old|new');
if (!['absent', 'present'].includes(values.desktop)) throw new Error('expected --desktop absent|present');
fs.mkdirSync(values.out, { recursive: true });
if (values.desktop === 'absent') {
  fs.writeFileSync(path.join(values.out, `desktop-chat-${values.phase}.json`), JSON.stringify({
    status: 'not-applicable', phase: values.phase, method: values.method,
    expectedCommit: values['expect-commit'], origin: 'source', reason: 'selected route does not install desktop',
  }, null, 2) + '\n');
} else try {
  fs.rmSync(path.join(values.out, `desktop-chat-${values.phase}.json`), { force: true });
  const release = path.join(values.root, 'apps', 'desktop', 'release');
  const candidates = {
    linux: ['linux-unpacked/Hermes', 'linux-unpacked/hermes'],
    darwin: [process.arch === 'arm64' ? 'mac-arm64/Hermes.app/Contents/MacOS/Hermes' : 'mac/Hermes.app/Contents/MacOS/Hermes'],
    win32: [process.arch === 'arm64' ? 'win-arm64-unpacked/Hermes.exe' : 'win-unpacked/Hermes.exe'],
  }[process.platform];
  if (!candidates) throw new Error(`unsupported source smoke host: ${process.platform}`);
  const executables = candidates.map(p => path.join(release, p)).filter(p => fs.existsSync(p));
  if (executables.length !== 1) throw new Error(`expected exactly one installed desktop executable, found ${executables.length} under ${release}`);
  const driver = fileURLToPath(new URL('./desktop-smoke.ts', import.meta.url));
  const args = [driver, '--exe', executables[0], '--origin', 'source'];
  for (const key of ['root', 'home', 'user-data', 'out', 'phase', 'expect-commit']) args.push(`--${key}`, values[key]);
  const mockUrl = values['mock-url'] || process.env.HERMES_E2E_MOCK_URL;
  if (mockUrl) args.push('--mock-url', mockUrl);
  const result = spawnSync(process.execPath, args, { stdio: 'inherit', cwd: values.root });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Desktop smoke process failed: exit=${result.status}, signal=${result.signal}`);
} catch (error) {
  const receiptPath = path.join(values.out, `desktop-chat-${values.phase}.json`);
  const prior = fs.existsSync(receiptPath) ? JSON.parse(fs.readFileSync(receiptPath, 'utf8')) : {};
  fs.writeFileSync(receiptPath, JSON.stringify({ ...prior, status: 'failed', phase: values.phase,
    origin: 'source', expectedCommit: values['expect-commit'], method: values.method,
    error: String(error) }, null, 2) + '\n');
  console.error(error);
  process.exitCode = 1;
}