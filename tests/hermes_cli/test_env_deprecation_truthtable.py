"""Truth-table regression test for the 'Deprecated .env settings' warning.

Behavior contract (pins the fix for the TERMINAL_CWD false-positive class:
#88829, #89016, #89389, #90299 — four reports in two weeks of the startup
warning firing when TERMINAL_CWD was merely in the inherited process
environment, e.g. bridged from ``terminal.cwd`` in config.yaml or set by
session restoration / cron / the gateway):

    warn_deprecated_cwd_env_vars() emits the warning IFF the deprecated key
    (TERMINAL_CWD or MESSAGING_CWD) has a genuinely non-empty value inside
    the user's ~/.hermes/.env FILE.

    * Process environment is IRRELEVANT — runtime bridges legitimately set
      TERMINAL_CWD in os.environ (fix commit 31561e37e).
    * config.yaml ``terminal.cwd`` is IRRELEVANT — it neither triggers nor
      suppresses the warning (fix commit a93f1b2be removed suppression).
    * Commented-out or empty-valued .env entries do NOT warn.
    * There is NO quiet/non-interactive gating in the scanner itself; the
      gateway calls it before setting HERMES_QUIET (gateway/run.py), so the
      warning fires regardless of HERMES_QUIET.

Exercises the real scanner against a real temp HERMES_HOME .env file.
"""

import pytest


REAL_PATH = "/home/user/projects/demo"

# Truth table rows:
#   env_file:    None = no .env file at all; str = literal .env file bytes
#   proc_env:    dict of process-environment vars to set (inherited, NOT .env)
#   config_yaml: None = no config.yaml; str = literal config.yaml contents
#   warn_keys:   key names that MUST appear in the warning ((), i.e. empty,
#                means the scanner must stay completely silent)
TRUTH_TABLE = [
    pytest.param(None, {}, None, (), id="all-absent-silent"),
    pytest.param(
        None, {"TERMINAL_CWD": REAL_PATH}, None, (),
        id="no-envfile-procenv-only-silent",
    ),
    pytest.param(
        "API_KEY=x\n", {"TERMINAL_CWD": "."}, None, (),
        id="procenv-dot-only-silent-88829",
    ),
    pytest.param(
        "API_KEY=x\n", {"TERMINAL_CWD": REAL_PATH}, None, (),
        id="procenv-realpath-only-silent-89016",
    ),
    pytest.param(
        "API_KEY=x\n",
        {"TERMINAL_CWD": REAL_PATH},
        f"terminal:\n  cwd: {REAL_PATH}\n",
        (),
        id="config-bridged-procenv-silent-89389",
    ),
    pytest.param(
        "API_KEY=x\n", {}, f"terminal:\n  cwd: {REAL_PATH}\n", (),
        id="config-only-silent-90299",
    ),
    pytest.param(
        "TERMINAL_CWD=.\n", {}, None, ("TERMINAL_CWD",),
        id="dotenv-dot-warns",
    ),
    pytest.param(
        f"TERMINAL_CWD={REAL_PATH}\n", {}, None, ("TERMINAL_CWD",),
        id="dotenv-realpath-warns",
    ),
    pytest.param(
        "TERMINAL_CWD=.\n", {}, "terminal:\n  cwd: /other/path\n",
        ("TERMINAL_CWD",),
        id="dotenv-dot-warns-despite-config",
    ),
    pytest.param(
        f"TERMINAL_CWD={REAL_PATH}\n",
        {"TERMINAL_CWD": REAL_PATH},
        f"terminal:\n  cwd: {REAL_PATH}\n",
        ("TERMINAL_CWD",),
        id="dotenv-warns-despite-procenv-and-config",
    ),
    pytest.param(
        "TERMINAL_CWD=\n", {"TERMINAL_CWD": REAL_PATH}, None, (),
        id="dotenv-empty-value-silent",
    ),
    pytest.param(
        "# TERMINAL_CWD=.\n", {"TERMINAL_CWD": "."}, None, (),
        id="dotenv-commented-silent",
    ),
    pytest.param(
        f"export TERMINAL_CWD={REAL_PATH}\n", {}, None, ("TERMINAL_CWD",),
        id="dotenv-export-prefix-warns",
    ),
    pytest.param(
        "TERMINAL_CWD=.\n", {"HERMES_QUIET": "1"}, None, ("TERMINAL_CWD",),
        id="quiet-mode-no-gating-dotenv-warns",
    ),
    pytest.param(
        "API_KEY=x\n",
        {"HERMES_QUIET": "1", "TERMINAL_CWD": "."},
        None,
        (),
        id="quiet-mode-procenv-only-silent",
    ),
    pytest.param(
        "MESSAGING_CWD=/msg/path\n", {}, None, ("MESSAGING_CWD",),
        id="dotenv-messaging-cwd-warns",
    ),
    pytest.param(
        "API_KEY=x\n", {"MESSAGING_CWD": "/msg/path"}, None, (),
        id="procenv-messaging-cwd-silent",
    ),
    pytest.param(
        f"MESSAGING_CWD=/msg/path\nTERMINAL_CWD={REAL_PATH}\n",
        {},
        None,
        ("MESSAGING_CWD", "TERMINAL_CWD"),
        id="dotenv-both-keys-warn",
    ),
]


@pytest.mark.parametrize("env_file, proc_env, config_yaml, warn_keys", TRUTH_TABLE)
def test_deprecated_env_warning_truth_table(
    env_file, proc_env, config_yaml, warn_keys, monkeypatch, tmp_path, capsys
):
    """Warning fires only for genuine .env-file entries — never for
    inherited process environment or config.yaml (#88829 #89016 #89389 #90299).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    if env_file is not None:
        (hermes_home / ".env").write_bytes(env_file.encode("utf-8"))
    if config_yaml is not None:
        (hermes_home / "config.yaml").write_text(config_yaml, encoding="utf-8")

    # Baseline: deprecated keys absent from process env unless the row says so.
    for key in ("TERMINAL_CWD", "MESSAGING_CWD", "HERMES_QUIET"):
        monkeypatch.delenv(key, raising=False)
    for key, value in proc_env.items():
        monkeypatch.setenv(key, value)

    import hermes_cli.config as config_module

    # load_env() memoises on (path, mtime, size); invalidate so this row's
    # freshly written file is what the scanner reads.
    config_module.invalidate_env_cache()

    config_module.warn_deprecated_cwd_env_vars()

    err = capsys.readouterr().err
    if warn_keys:
        assert "Deprecated .env settings detected" in err
        for key in warn_keys:
            assert f"{key}=" in err, f"expected {key} in warning:\n{err}"
        assert "config.yaml" in err  # migration hint present
    else:
        assert err == "", f"expected silence, got:\n{err}"
