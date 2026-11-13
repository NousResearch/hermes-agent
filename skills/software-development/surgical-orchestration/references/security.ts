import * as path from 'node:path';
import * as fs from 'node:fs';

/**
 * Resolves physical path handling non-existent target files by checking
 * the nearest existing parent directory realpath to prevent symlink bypass.
 */
export function resolveRealPath(targetPath: string): string {
  const normalizedPath = path.normalize(targetPath);
  if (fs.existsSync(normalizedPath)) {
    return fs.realpathSync(normalizedPath);
  }

  const parentDir = path.dirname(normalizedPath);
  const baseName = path.basename(normalizedPath);

  if (fs.existsSync(parentDir)) {
    return path.join(fs.realpathSync(parentDir), baseName);
  }

  return path.resolve(normalizedPath);
}

/**
 * Asserts that targetPath resides strictly inside allowedScope.
 * Throws security violation error if boundary is breached.
 */
export function assertScopeBoundary(targetPath: string, allowedScope: string, agentId: string): void {
  const absTarget = resolveRealPath(targetPath);
  const absScope = resolveRealPath(allowedScope);

  const relative = path.relative(absScope, absTarget);
  const isOutside = relative.startsWith('..') || path.isAbsolute(relative);

  if (isOutside && absTarget !== absScope) {
    throw new Error(
      `[SECURITY_VIOLATION] Subagent '${agentId}' attempted unauthorized path access: '${targetPath}' (resolved: '${absTarget}'). Locked Scope: '${absScope}'`
    );
  }
}