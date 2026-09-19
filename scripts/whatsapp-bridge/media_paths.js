import fs from 'node:fs';
import path from 'node:path';

export function resolveAllowedMediaPath(candidate, roots, maxBytes) {
  if (typeof candidate !== 'string' || !path.isAbsolute(candidate)) return null;
  const absoluteCandidate = path.resolve(candidate);

  for (const root of roots) {
    const normalizedRoot = path.resolve(root);
    if (absoluteCandidate !== normalizedRoot && !absoluteCandidate.startsWith(`${normalizedRoot}${path.sep}`)) continue;

    try {
      const resolvedRoot = fs.realpathSync(normalizedRoot);
      const resolved = fs.realpathSync(absoluteCandidate);
      // Reject symlink escapes before accessing the target's metadata.
      if (!resolved.startsWith(`${resolvedRoot}${path.sep}`)) continue;
      const metadata = fs.statSync(resolved);
      if (metadata.isFile() && metadata.size <= maxBytes) {
        return resolved;
      }
    } catch {
      // Try the next configured cache root when a path or cache is unavailable.
    }
  }

  return null;
}
