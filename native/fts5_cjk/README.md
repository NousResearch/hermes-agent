# fts5_cjk — cjk_unicode61 FTS5 tokenizer

unicode61 + CJK character bigrams (Lucene CJKAnalyzer semantics). Fixes
2-char Korean/CJK terms falling through to LIKE full-table scans.

Build & install to ~/.hermes/lib/:

    ./build.sh

Then run `scripts/fts_v2_migrate.py` to create + backfill messages_fts_v2;
reads cut over by default once the migration verifies the index. Set
`agent.fts_v2_read: false` in ~/.hermes/config.yaml to fall back to the
legacy tables. Override the .so location with `HERMES_FTS5_CJK_SO`.
