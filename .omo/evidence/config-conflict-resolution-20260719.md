# `hermes_cli/config.py` conflict resolution evidence

## Scope and intent

The four conflict regions were resolved with the upstream implementation as
the base. The upstream nested config accessors and `config get`/`config unset`
handlers were retained. The fork's duplicate API-key list in `set_config_value`
was removed in favor of the upstream `_is_env_config_key` helper. The verified
fork advantages retained are `_normalize_model_api_key_for_save` (called by
`save_config`) and the existing `OPTIONAL_ENV_VARS` catalog.

## Verification

* `python -m py_compile hermes_cli/config.py` — exit 0.
* `python -c "import hermes_cli.config as c; print('config_import_ok', callable(c._is_env_config_key), callable(c._normalize_model_api_key_for_save))"` — `config_import_ok True True`.
* `scripts\\run_tests.sh tests\\hermes_cli\\test_config.py -q` — exit 0.
* `scripts\\run_tests.sh tests\\hermes_cli\\test_clear_stale_base_url.py -q` — exit 0.
* `git diff --check --cached -- hermes_cli/config.py` — clean.

The resolved file is staged as `M  hermes_cli/config.py`.
