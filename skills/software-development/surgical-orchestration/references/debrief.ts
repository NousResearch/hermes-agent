import * as crypto from 'node:crypto';
import * as path from 'node:path';

export interface DebriefPayload {
  directory: string;
  modifiedFiles: string[];
  diffHash: string;
  errorSignature?: string;
}

/**
 * Produces a canonicalized, deterministic SHA-256 hash of a debrief payload.
 * Normalizes paths, sorts file arrays, and strips dynamic metadata.
 */
export function computeDebriefHash(payload: DebriefPayload): string {
  const canonicalObject = {
    directory: path.normalize(payload.directory).replace(/\\/g, '/'),
    modifiedFiles: [...payload.modifiedFiles]
      .map(f => path.normalize(f).replace(/\\/g, '/'))
      .sort(),
    diffHash: payload.diffHash.trim(),
    errorSignature: (payload.errorSignature || '').trim()
  };

  const canonicalJson = JSON.stringify(canonicalObject);
  return crypto.createHash('sha256').update(canonicalJson).digest('hex');
}