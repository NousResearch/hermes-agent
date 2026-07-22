# fts5_cjk — cjk_unicode61 FTS5 tokenizer

unicode61 + CJK character bigrams (Lucene CJKAnalyzer semantics). Fixes
2-char Korean/CJK terms falling through to LIKE full-table scans.

Build & install to ~/.hermes/lib/:

    ./build.sh

Then run `scripts/fts_v2_migrate.py` to create + backfill messages_fts_v2,
and set `HERMES_FTS_V2_READ=1` in ~/.hermes/.env to cut reads over.
Override the .so location with `HERMES_FTS5_CJK_SO`.
