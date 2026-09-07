// Voice-attachment upload naming for the Photon sidecar.
//
// spectrum-ts's iMessage `voice()` path transcodes non-m4a audio to an m4a
// container before upload, but uploads it under `content.name` — which
// defaults to the basename of the path Hermes passes (e.g. `1234.mp3` from
// the TTS cache). iOS then receives a voice note whose .mp3 file name
// disagrees with its m4a payload and renders it as a short, unplayable
// bubble (#88083). Naming the upload what the payload actually is (.m4a)
// matches the manually-verified direct-sidecar path, where a real .m4a is
// delivered intact.
//
// This lives in its own module (rather than inline in index.mjs) so tests can
// execute the real naming logic under node instead of grepping source — see
// tests/plugins/platforms/photon/test_voice_attachment_name.py.

import { basename, extname } from "node:path";

const M4A_EXTENSIONS = new Set([".m4a", ".m4b", ".m4p"]);

/**
 * Name spectrum-ts should upload a voice note under. spectrum-ts re-encodes
 * non-m4a audio to m4a, so any non-m4a extension must be rewritten to .m4a
 * to keep the upload name and the payload container in agreement.
 *
 * @param {string} path the local audio path Hermes delivers
 * @param {string | undefined} name explicit name from the request, if any
 * @returns {string | undefined} name to pass to the voice() builder
 */
export function voiceAttachmentName(path, name) {
  const base = String(name || basename(String(path)));
  const ext = extname(base).toLowerCase();
  if (!ext) {
    return base + ".m4a";
  }
  if (M4A_EXTENSIONS.has(ext)) {
    return name || undefined;
  }
  return base.slice(0, base.length - ext.length) + ".m4a";
}
