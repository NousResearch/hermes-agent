# Changelog

All notable changes to the GitHub Pull Requests dashboard plugin.

## [1.0.1] - 2026-08-02

### Fixed

- PR detail loading (HTTP 502 → 200): `DETAIL_FIELDS` was built from the
  `gh search prs` field list (`SUMMARY_FIELDS`), which contains `repository`
  and `commentsCount` — fields `gh pr view --json` rejects. Every `/detail`
  request failed with `RuntimeError` → HTTP 502, so the desktop UI showed
  "Could not load PR details" for every PR. `DETAIL_FIELDS` is now an explicit
  `gh pr view`-compatible field list. The repository is still passed via
  `--repo`; `commentsCount` is not rendered in the detail view.

## [1.0.0] - 2026-08-02

### Added

- Initial release: Created / Review requested / Closed pull request views
  backed by the user's authenticated `gh` CLI. Read-only.
