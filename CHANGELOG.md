# Changelog

All notable user-facing changes to hermes-agent are tracked here.

## Unreleased

### Fixes

- **gateway/telegram**: image files uploaded as Telegram *documents*
  (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`) are now downloaded, cached, and
  routed through the same vision/photo handling path as native Telegram
  photos instead of being rejected with `Unsupported document type`. The
  document-type error message also now lists the supported image
  extensions. (#20128)
