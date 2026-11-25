/** npm's allowScripts is keyed by exact `name@version`: a dependency bump that
 * orphans a pin, or a new package with an install script, fails here, in the
 * docs lane that changes the lockfile (tests-js runs only in the frontend lane). */

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const allow = JSON.parse(readFileSync(join(docsRoot, 'package.json'), 'utf8')).allowScripts ?? {};
const packages = Object.entries(
  JSON.parse(readFileSync(join(docsRoot, 'package-lock.json'), 'utf8')).packages ?? {},
).filter(([lockPath, entry]) => lockPath !== '' && entry.version);

const nameOf = (lockPath, entry) => entry.name ?? lockPath.slice(lockPath.lastIndexOf('node_modules/') + 13);

test('every versioned allowScripts pin matches a version in the lockfile', () => {
  const installed = new Set(packages.map(([lockPath, entry]) => `${nameOf(lockPath, entry)}@${entry.version}`));
  const stale = Object.keys(allow).filter((key) => /^.+@[^@]+$/.test(key) && !installed.has(key));
  assert.deepEqual(stale, [], 'npm silently blocks the install script of a pin the lockfile no longer resolves');
});

test('every package with an install script has an allowScripts decision', () => {
  const uncovered = packages
    .filter(([, entry]) => entry.hasInstallScript)
    .map(([lockPath, entry]) => [nameOf(lockPath, entry), entry.version])
    .filter(([name, version]) => !(`${name}@${version}` in allow) && !(name in allow))
    .map(([name, version]) => `${name}@${version}`);
  assert.deepEqual(uncovered, [], 'add "<name>@<version>": true to allow it, or "<name>": false to deny it');
});
