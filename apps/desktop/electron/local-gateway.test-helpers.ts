/**
 * Temp homes for the local-gateway suites that bind real AF_UNIX control sockets.
 *
 * sun_path is 104 bytes on macOS (108 on Linux). There os.tmpdir() is
 * /var/folders/xx/<28 chars>/T (~49 chars, /private/var/... once realpath'd),
 * and vitest.run-tmp.ts nests a vitest-XXXXXX root under it, so
 * `<home>/gateway.sock` and `<runtime>/hermes-gw-<16 hex>/control.sock` overflow.
 * On darwin the base is /tmp (/private/tmp); never the user's home.
 */

import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

/** Canonical (realpath'd: the gateway canonicalises profile_id) temp dir with a short prefix. */
export async function shortSocketTmpDir(prefix: string): Promise<string> {
  // no-tmp: ok — macOS AF_UNIX sun_path is 104 bytes; os.tmpdir() there is too long for the socket.
  const base = os.platform() === 'darwin' ? '/tmp' : os.tmpdir()

  return fs.realpath(await fs.mkdtemp(path.join(base, prefix)))
}
