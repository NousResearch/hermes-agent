# Changelog

All notable changes to this skill will be documented in this file.

## [1.2.18] - 2026-04-22

### Changed
- Optimized `claw.json` description

## [1.2.17] - 2026-04-15

### Fixed
- Unified `growth` unit description to percentage (`11.5 = 11.5%`) in schema to remove conflicting definitions
- Updated range filter example from `--min-growth 0.2` to `--min-growth 20` to match percentage semantics
- Updated CLI `--min-growth` help text to percentage format for consistency with API behavior

### Changed
- Bumped skill version from `1.2.16` to `1.2.17` across `SKILL.md`, `claw.json`, and `scripts/query.py`

## [1.2.16] - 2026-04-06

### Changed
- Optimized `claw.json` description

## [1.2.15] - 2026-04-02

### Added
- Added "Quickest Way" setup: paste ClawHub URL + token directly to OpenClaw agent for automatic install & config
- Added OpenClaw CLI install command (`openclaw skills install`) to Setup section
- Added `openclaw.json` config option (Option A) for persistent token storage alongside shell `export` (Option B)
- Updated data coverage from 14M+ to 10M+ across all files

### Changed
- Restructured Setup section into three tiers: Quickest Way → Manual CLI → Token config

## [1.2.14] - 2026-04-02

### Changed
- Optimized `claw.json` description

## [1.2.13] - 2026-04-02

### Changed
- Optimized `claw.json` description for vector similarity search — naturally embedded target keywords (ecommerce store, ecommerce website, ecommerce data, shopify store)
- Expanded `claw.json` tags with compound keyword tags: `ecommerce-store`, `ecommerce-website`, `ecommerce-data`, `shopify-store`, `online-store`, `lead-generation`
- Updated `SKILL.md` frontmatter description to include target search phrases
- Updated `README.md` title and opening paragraph for better keyword coverage
- Standardized all descriptions to use "ecommerce" (no hyphen) consistently

## [1.2.12] - 2026-04-01

### Fixed
- Fixed "Adress" typo → "Address" in SKILL.md agent tips
- Fixed "bigcommerce" → "BigCommerce" capitalization in SKILL.md platform list
- Aligned platform list between README.md and SKILL.md (removed Magento from README to match)
- Fixed `growth` field description in schema.md: was "decimal, 0.25 = 25%", now "percentage, 11.5 = 11.5%" to match actual API behavior and `fmt_growth` formatting

## [1.2.11] - 2026-04-01

### Added
- Description on EcCompass sub-skill structure. 

## [1.2.10] - 2026-04-01

### Fixed
- Fixed social media followers display: values of 0 were incorrectly skipped due to falsy check; now uses explicit `None` comparison
- Fixed `estimatedMonthlySales` raw number output in domain analytics; now formatted with `fmt_money()` for consistent display
- Fixed `estimatedMonthlySales` type in schema example: was string `"500000"`, now long `500000` matching the documented type

## [1.2.9] - 2026-03-31

### Fixed
  - Fixed description.

## [1.2.8] - 2026-03-31

### Fixed

- **Monthly API quota now enforced at user level instead of per-token** — prevents users from bypassing limits by creating multiple tokens
  - `isMonthlyLimitExceeded` now sums `monthly_used` across all tokens for the same `user_id`
  - Frontend `monthlyUsed` field now reflects user-level total usage
  - Redis rate limit key for monthly dimension changed from `api:ratelimit:month:{apiKey}` to `api:ratelimit:month:user:{userId}`
  - Short-cycle limits (minute/hour/day) remain per-token to prevent single-token abuse

## [1.2.7] - 2026-03-30

### Added
  - Add platform support.

## [1.2.6] - 2026-03-26

### Added
  - Updated SKILL.md document.

## [1.2.4] - 2026-03-26

### Fixed

- Resolved ClawHub security flag: metadata inconsistencies across files
- Aligned `version`, `author`, `requires.bins`, `requires.env` between `claw.json`, `SKILL.md`, and `query.py`
- Removed false `curl` dependency — skill only requires `python3`
- Added `requires` section to `claw.json` to explicitly declare `bins` and `env`
- Fixed data update frequency inconsistency (README said "Weekly", now all files say "Monthly")

## [1.2.3] - 2026-03-26

### Added

- **3 new API endpoints**:
  - `GET /historical/{domain}` — monthly GMV, UV, PV, and average price history from 2023 onwards
  - `GET /installed-apps/{domain}` — installed apps/plugins with ratings, install counts, vendor info, and pricing plans
  - `GET /contacts/{domain}` — verified LinkedIn contacts (name, position, email) for a domain's company
- **Exists filter** (`exists`): query fields that must be present and not empty (e.g. `["tiktokUrl", "emails"]`)
- **Multi-value OR filters**: pass comma-separated values for the same field (e.g. `region: "Europe,Africa"`)
- **Complete field reference** in SKILL.md — all available ES fields documented for filters, ranges, and exists
- Case-insensitive keyword filters for platform, country, and other Keyword-type fields
- Keyword search now covers `installed_apps` and `platform` fields
- CLI commands: `historical`, `apps`, `contacts` subcommands in `query.py`
- CLI flag `--exists` for exists filter in search

### Changed

- Search API changed from `GET /search/{keyword}` to `POST /search` with JSON body
- Search keyword is now optional (can search by filters only)
- Range queries use `gte`/`lte` JSON syntax for Elasticsearch compatibility
- `installed-apps` and `contacts` endpoints return empty data instead of 400 error when no results found

### Fixed

- Elasticsearch `[range] query does not support [from]` error — switched to `wrapperQuery` with manual JSON
- `estimatedMonthlySales` and `estimatedSalesYearly` field types corrected from String to Long
- `avgPriceUsd` field type corrected from String to Integer
- Historical data API double-wrapping issue (nested `data.data`)
- Range query `Double.toString()` scientific notation risk — now uses `BigDecimal.toPlainString()`
- Removed duplicate database queries in API key validation flow
- Removed API whitelist mechanism from rate limit aspect

## [1.2.1] - 2026-03-19

### Added

- Keyword search across 10M+ e-commerce domains via ECcompass REST API
- Full domain analytics with 100+ fields (GMV, traffic, social media, tech stack, etc.)
- Python CLI tool (`scripts/query.py`) with `search` and `domain` subcommands
- JSON export support (`--json` flag)
- Paginated search results (up to 100 per page)
- Complete API response schema documentation
- Usage examples covering real-world scenarios

