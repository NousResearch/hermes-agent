import { blobatar, type BlobatarOptions } from "./blobatar";

/**
 * A `data:` URI suitable for `<img src>` or `background-image`.
 *
 * Percent-encoded rather than base64: base64 inflates payloads ~33%, while SVG
 * markup is mostly characters that survive percent-encoding untouched. Only the
 * characters that actually break inside an attribute are escaped.
 */
export function blobatarUri(name: string, opts?: BlobatarOptions): string {
  const svg = blobatar(name, opts)
    .replace(/"/g, "'")
    .replace(/[%#<>{}|\\^[\]`]/g, c => "%" + c.charCodeAt(0).toString(16).toUpperCase());
  return "data:image/svg+xml," + svg.replace(/\s+/g, " ");
}
