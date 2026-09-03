# Governed content routing TDD evidence

RED command (2026-09-03, before implementation):

```text
pytest -q content_engine/tests/test_content_llm_topics.py::test_llm_configs_use_governed_config_and_runtime_resolution content_engine/tests/test_content_llm_topics.py::test_llm_configs_do_not_filter_routes_by_environment content_engine/tests/test_blog_illustrator.py::test_codex_cli_uses_isolated_pool_accounts_and_fails_over content_engine/tests/test_governed_cron_scripts.py
```

Expected pre-fix failures: the content module has no Hermes config/runtime seams,
the illustrator has no native Codex pool seam, and the four canonical cron script
sources are absent from the repository.

Observed: 3 failures (`_load_hermes_config` missing twice and
`_load_codex_pool` missing); the script-source assertion was initially hidden by
the file-granular runner's combined `-k` handling and was subsequently retained
as a permanent behavioural/source contract.
