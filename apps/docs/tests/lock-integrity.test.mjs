/** Registry downloads need committed tarball URLs and SHA-512 integrity.
 * Root/link entries are local; bundled members ship in their parent's tarball. */

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const docsRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const lock = JSON.parse(readFileSync(join(docsRoot, 'package-lock.json'), 'utf8'));

const TRUSTED_REGISTRY = 'https://registry.npmjs.org/';
const SRI_SHA512 = /^sha512-[A-Za-z0-9+/]+={0,2}$/;
const SHA512_DIGEST_BYTES = 64;

/** The package name for a lockfile path, e.g. `@emnapi/core` (nested paths take the last segment). */
function packageName(lockPath) {
  return lockPath.slice(lockPath.lastIndexOf('node_modules/') + 'node_modules/'.length);
}

function unscopedName(name) {
  return name.startsWith('@') ? name.slice(name.indexOf('/') + 1) : name;
}

/** The tarball URL the npm registry serves for an exact `name@version`. */
function expectedTarball(name, version) {
  return `${TRUSTED_REGISTRY}${name}/-/${unscopedName(name)}-${version}.tgz`;
}

const entries = Object.entries(lock.packages ?? {}).filter(([lockPath]) => lockPath !== '');
const linked = entries.filter(([, entry]) => entry.link === true);
const bundled = entries.filter(([, entry]) => entry.inBundle === true);
// Everything else is a registry download: dev, optional and platform
// (`os`/`cpu`/`libc`) entries are fetched and verified by npm ci like any
// other package, so none of those flags excuses missing identity.
const fetched = entries.filter(
  ([, entry]) => entry.link !== true && entry.inBundle !== true,
);

test('every registry package carries a reviewed resolved URL and integrity digest', () => {
  const missing = fetched
    .filter(([, entry]) => typeof entry.resolved !== 'string' || typeof entry.integrity !== 'string')
    .map(([lockPath]) => `${lockPath} (${packageName(lockPath)}@${lock.packages[lockPath].version})`);
  assert.deepEqual(
    missing,
    [],
    `${missing.length} of ${fetched.length} registry packages in package-lock.json lack ` +
      `'resolved'/'integrity', so npm ci trusts unreviewed install-time bytes for them.\n` +
      `${missing.slice(0, 15).map((p) => `  ${p}`).join('\n')}\n` +
      'Regenerate the lockfile with full registry metadata (npm resolution with the exact ' +
      'pinned versions) — never hand-write URLs or digests.',
  );
});

test('resolved tarball URLs point at the trusted registry for the pinned name and version', () => {
  const offenders = fetched
    .filter(([, entry]) => typeof entry.resolved === 'string')
    .filter(([lockPath, entry]) => entry.resolved !== expectedTarball(packageName(lockPath), entry.version))
    .map(([lockPath, entry]) => `  ${lockPath}: ${entry.resolved}`);
  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} resolved URLs are not the trusted registry tarball for their pinned ` +
      `name@version:\n${offenders.join('\n')}\n` +
      `Every resolved URL must be ${TRUSTED_REGISTRY}<name>/-/<name>-<version>.tgz over https.`,
  );
});

test('integrity digests are well-formed sha512 SRI values', () => {
  const offenders = fetched
    .filter(([, entry]) => typeof entry.integrity === 'string')
    .map(([lockPath, entry]) => [lockPath, entry.integrity])
    .filter(([lockPath, integrity]) => {
      if (!SRI_SHA512.test(integrity)) return true;
      const digest = Buffer.from(integrity.slice('sha512-'.length), 'base64');
      return digest.length !== SHA512_DIGEST_BYTES;
    })
    .map(([lockPath, integrity]) => `  ${lockPath}: ${integrity}`);
  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} integrity digests are not sha512-<base64 of 64 bytes>:\n` +
      `${offenders.join('\n')}`,
  );
});

test('metadata exemptions are exactly the root, link nodes and bundled members', () => {
  // The root entry is the docs app itself.
  assert.ok('' in lock.packages, 'the root "" entry must exist in the lockfile');

  const withoutIdentity = entries
    .filter(([, entry]) => !entry.resolved || !entry.integrity)
    .map(([lockPath]) => lockPath)
    .sort();
  const excused = [...linked, ...bundled]
    .map(([lockPath]) => lockPath)
    .sort();
  assert.deepEqual(
    withoutIdentity,
    excused,
    'Entries without resolved/integrity must be exactly the link and bundled nodes — ' +
      'dev, optional and platform packages are still fetched and must stay pinned. ' +
      'Unexpected exemptions:\n' +
      withoutIdentity.filter((p) => !excused.includes(p)).map((p) => `  ${p}`).join('\n'),
  );

  const nakedExemptions = bundled
    .filter(([, entry]) => typeof entry.version !== 'string')
    .map(([lockPath]) => lockPath);
  assert.deepEqual(
    nakedExemptions,
    [],
    `Bundled entries must still pin a version:\n${nakedExemptions.join('\n')}`,
  );

  // A bundled member only exists inside its bundling parent's tarball, so npm
  // records it nested under the parent. A top-level inBundle flag would be a
  // blanket exemption, not a registry fact.
  const topLevelBundled = bundled
    .map(([lockPath]) => lockPath)
    .filter((lockPath) => lockPath.slice('node_modules/'.length).includes('node_modules/') === false);
  assert.deepEqual(
    topLevelBundled,
    [],
    `inBundle entries must sit inside their bundling parent, not at the top level:\n${topLevelBundled.join('\n')}`,
  );
});
