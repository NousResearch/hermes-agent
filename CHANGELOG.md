# Changelog

All notable changes in this contribution are documented here.

## [Unreleased]

### Fixed

- Read project skill allowlists as UTF-8 with an optional BOM, preserving the first skill and treating BOM-only or comment-only files as no narrowing. Add regression coverage for LF and CRLF line endings.
