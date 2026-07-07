# Anime Inquiry Sheet Skill

This directory contains the `anime-inquiry-sheet` Hermes skill. It researches anime/manga/game IPs, writes inquiry-priority rows to the Google Sheet `作品リストfor問い合わせ`, and returns the conclusion, reason, and Spreadsheet URL.

## Files

- `SKILL.md` — the runtime skill instructions.
- `README.md` — setup and operating notes for repository users.

## Google Sheets Target

The current skill is configured for:

- Spreadsheet title: `作品リストfor問い合わせ`
- Spreadsheet ID: `1ZZ9qF5UFr2GGO5IGiWJCBgcUqAhu3PDEQYD-m5NlD-g`
- Sheet name: `シート1`
- Expected columns: `A:N` / 14 columns

If you use this skill with a different sheet, update the Sheet ID, sheet name, and column assumptions in `SKILL.md` before use.

## Required Google Access

The agent account must be able to read and edit the target spreadsheet. Minimum required capability:

- Google Sheets read access
- Google Sheets write access
- Permission to apply cell formatting via Sheets API `batchUpdate`

For the configured workflow, the account should be able to:

1. Read existing rows from `シート1!A:N`.
2. Append new 14-column rows.
3. Clear the appended row background color.
4. Set the appended row text color to black.
5. Read the appended rows back for verification.

## Authentication Setup

This skill relies on the `google-workspace` skill and its Google API wrapper.

Before using the skill, verify Google auth:

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
$GSETUP --check
```

A usable setup prints `AUTHENTICATED`.

If auth is missing or expired, follow the `google-workspace` skill setup flow:

1. Create or select a Google Cloud project.
2. Enable the Google Sheets API and Google Drive API.
3. Create a Desktop OAuth client.
4. Run the `google-workspace` setup script to authorize the account.
5. Re-run `$GSETUP --check` until it reports `AUTHENTICATED`.

The user/account you authorize must also have edit access to the configured spreadsheet.

## Credential Safety

Do not commit Google credentials or token files.

Never add these to git:

- `google_token.json`
- `google_client_secret.json`
- OAuth refresh tokens
- access tokens
- service account JSON files
- exported `gws` credential files

If logs or command output contain credential material, redact it before sharing or committing.

## Optional `gws` CLI Notes

Some environments also have the standalone `gws` CLI authenticated. That can be useful for Sheets formatting operations such as `batchUpdate`.

Verify availability:

```bash
command -v gws
```

Verify auth if using `gws` directly:

```bash
gws auth status
```

The skill should still treat authentication as an environment prerequisite, not as something stored in the repository.

## Preflight Checklist

Before running a real inquiry task:

- [ ] `google-workspace` auth check returns `AUTHENTICATED`.
- [ ] The authorized Google account can edit the target spreadsheet.
- [ ] The sheet has the expected 14 columns in `A:N`.
- [ ] Existing rows can be read before research for duplicate detection.
- [ ] Appended rows can be formatted: background cleared and text set to black.
- [ ] The skill file is loaded from the repository path `skills/productivity/anime-inquiry-sheet/SKILL.md`.

## Operational Notes

- For a single work title, the skill checks the spreadsheet first. If the title is already present, it reports the existing row number and does not research or append a duplicate.
- For a source URL containing multiple works, the skill extracts work candidates, deduplicates against the spreadsheet, researches only new works, and appends new rows in batches where possible.
- Batch research should avoid putting raw HTML, full search-result pages, or long logs into the conversation context. Keep only extracted titles, compact 14-column row data, and final results.
- Spreadsheet writes are centralized in the main session; delegated research tasks should return row data only and should not write to Google Sheets directly.
