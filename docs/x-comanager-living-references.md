# Personal X co-manager: living references

## Behaviour

- Runtime voice references come from observed, authored Sahil_Saghir publications in `$HERMES_HOME/data/x-analytics/observations.json`, never the deleted legacy archive. Refresh reads the own-account replies timeline, which also contains authored posts and quote posts. Quoted third-party text is not attributed to Sahil. Truncated, future-observed and invalid records are excluded. Missing optional references do not block ordinary opinions or questions.
- References influence wording and perspective. They do not become rigid positions, claims of personal experience, or permission to publish. User-approved conversational calibration is optional and explicitly separate from published history.
- Runtime guidance is the narrow `sahil-twitter-voice/references/runtime-voice.md`, with the source-controlled `content_engine/x_voice_runtime.md` as the default fallback. The old, contradictory full skill document is not pasted into model prompts.
- Grounding retrieves relevant approved SahilBlog material, local commit records, memory hints, and owned GitHub repositories including private repositories. GitHub indexes all owned repositories, then retrieves a bounded relevant subset of recent commits. Local discovery covers roots and two grouping levels; it does not pretend to exhaust arbitrary disk trees.
- A commit is evidence of committed work, not passing tests, deployment or measured benefit. Memory is a fallible hint. Private material remains internal: generated public text must not disclose private names, paths, URLs, secrets, client information or unpublished details without separate approval. Credential-looking excerpts are dropped, including complete PEM-containing chunks.
- Personal X uses its owning runtime's configured model and fallbacks through the existing native provider/pool client. It no longer silently borrows the content-strategist profile's broken route. No profile, credential or gateway configuration is modified by this change.
- Feed scrolling measures progress by IDs, not a virtualised DOM's constant article count. A worthwhile batch may be smaller than three; no filler is generated to reach a quota.
- Freshness and source identity are checked during collection, staging and immediately before delivery. Expired or invalid receipts cannot be revived. All generated output remains pending human approval.

## Controls and scope

- `X_REFRESH_OWN_REFERENCES=0`: internal isolation/diagnostic override to suppress browser refresh; defaults on. Unit tests set this; live verification does not.
- `X_OBSERVATIONS_PATH`: optional isolated observation-file override; production defaults to the path above.
- `X_KNOWLEDGE_ROOTS_JSON`: JSON list of local repository roots; defaults to `~/repos` and `~/worktrees`.
- `X_GITHUB_KNOWLEDGE=0`: disables optional remote repository retrieval. Default uses the authenticated `gh` account's owned public/private repositories, with a 24-hour catalog and one-hour commit cache. Coverage and unavailable selections are reported; inaccessible repositories are not invented.
- Memory retrieval reads a curated `research/x-voice/memory-hints.json` when present, otherwise relevant stable `memories/MEMORY.md` paragraphs. It is not an exhaustive Mnemosyne search or proof that remembered outcomes are still true.
- Real generation/staging verification uses an isolated queue. A labelled preview can be sent once through scheduler delivery to the existing review channel without enabling a cron or publishing to X.

## Verification

Run from the candidate checkout using the runtime venv:

    scripts/run_tests.sh -j 4 content_engine/tests tests/cron -q -W error::RuntimeWarning

The suite owns its policy fixtures, temporary previews and external-source doubles. Real browser, GitHub, model and Discord evidence is captured separately; fixtures are never described as live evidence.

Detailed receipts and real model outputs are under `/home/kensei/x-comanager-repair-evidence/living-run/`. The Discord test delivery is read back by message ID and its downloaded attachment is SHA-256 matched to the actual report. No activation, publishing or other profile changes are part of build verification.
