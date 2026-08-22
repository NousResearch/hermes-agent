"""Execute the Desktop SSH argv classifier. No /proc, no mocked OWNED strings."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = REPO_ROOT / "apps" / "desktop" / "electron" / "ssh_pid_owned.py"

HOME = "/home/u"
HERMES_HOME = f"{HOME}/.hermes"
WRAPPER = f"{HOME}/.local/bin/hermes"
PYTHON = f"{HERMES_HOME}/hermes-agent/venv/bin/python"
REPO_HERMES = f"{HERMES_HOME}/hermes-agent/hermes"
VENV_HERMES = f"{HERMES_HOME}/hermes-agent/venv/bin/hermes"
NONCE = "0123456789abcdef"
OWNERSHIP_ID = "0123456789abcdef0123456789abcdef"
TOKEN = f"{HERMES_HOME}/desktop-ssh/{OWNERSHIP_ID}/{NONCE}.token"
PROFILE = "cto"


def this_host_args(**over: object) -> list[str]:
    args = [
        str(over.get("argv0", PYTHON)),
        str(over.get("argv1", REPO_HERMES)),
        "--profile",
        str(over.get("profile", PROFILE)),
        "serve",
        "--isolated",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--ssh-session-token-file",
        str(over.get("token", TOKEN)),
        "--ssh-owner-nonce",
        str(over.get("nonce", NONCE)),
    ]
    if over.get("drop_isolated"):
        args.remove("--isolated")
    return args


def classify_legacy_4arg(args: list[str], expected: str, nonce: str) -> str:
    """HEAD / running Desktop 4-arg rule (direct || python_entry vs hermesPath)."""
    try:
        serve = args.index("serve")
        owner = args.index("--ssh-owner-nonce", serve + 1)
        direct = args[0] == expected
        python_entry = (
            len(args) > 1
            and args[1] == expected
            and os.path.basename(args[0]).startswith("python")
        )
        ok = (
            (direct or python_entry)
            and "--isolated" in args[serve + 1 :]
            and args[owner + 1] == nonce
        )
    except (ValueError, IndexError):
        ok = False
    return "OWNED" if ok else "FOREIGN"


def load_classifier():
    spec = importlib.util.spec_from_file_location("ssh_pid_owned", CLASSIFIER_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(CLASSIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_this_host_shape_is_foreign_under_old_4arg_rule():
    assert (
        classify_legacy_4arg(this_host_args(), WRAPPER, NONCE) == "FOREIGN"
    ), "python + hermes-agent/hermes must be FOREIGN when expected is the wrapper path"


def test_classify_dashboard_argv_exists():
    module = load_classifier()
    assert callable(module.classify_dashboard_argv)


def test_this_host_shape_is_owned_without_spawn_proof():
    module = load_classifier()
    result = module.classify_dashboard_argv(
        this_host_args(),
        WRAPPER,
        NONCE,
        hermes_home=HERMES_HOME,
        token_path="",
        profile=PROFILE,
        allow_spawn_proof=False,
    )
    assert result == "OWNED"


def test_this_host_wrong_nonce_is_foreign():
    module = load_classifier()
    result = module.classify_dashboard_argv(
        this_host_args(nonce="fedcba9876543210"),
        WRAPPER,
        NONCE,
        hermes_home=HERMES_HOME,
        token_path=TOKEN,
        profile=PROFILE,
    )
    assert result == "FOREIGN"


def test_this_host_without_isolated_is_foreign():
    module = load_classifier()
    result = module.classify_dashboard_argv(
        this_host_args(drop_isolated=True),
        WRAPPER,
        NONCE,
        hermes_home=HERMES_HOME,
        token_path=TOKEN,
        profile=PROFILE,
    )
    assert result == "FOREIGN"


def test_direct_hermes_path_is_owned():
    module = load_classifier()
    args = [WRAPPER, "--profile", PROFILE, "serve", "--isolated", "--ssh-owner-nonce", NONCE]
    assert (
        module.classify_dashboard_argv(
            args, WRAPPER, NONCE, hermes_home=HERMES_HOME, profile=PROFILE
        )
        == "OWNED"
    )


def test_python_plus_hermes_path_is_owned():
    module = load_classifier()
    args = [
        PYTHON,
        WRAPPER,
        "--profile",
        PROFILE,
        "serve",
        "--isolated",
        "--ssh-owner-nonce",
        NONCE,
    ]
    assert (
        module.classify_dashboard_argv(
            args, WRAPPER, NONCE, hermes_home=HERMES_HOME, profile=PROFILE
        )
        == "OWNED"
    )


def test_wrapper_exec_targets_are_owned_without_spawn_proof(tmp_path: Path):
    module = load_classifier()
    wrapper = tmp_path / "hermes"
    script = tmp_path / "custom-hermes"
    script.write_text("# entry\n", encoding="utf-8")
    wrapper.write_text(
        f'#!/usr/bin/env bash\nexec "{PYTHON}" "{script}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    args = [
        PYTHON,
        str(script),
        "--profile",
        PROFILE,
        "serve",
        "--isolated",
        "--ssh-owner-nonce",
        NONCE,
    ]
    assert (
        module.classify_dashboard_argv(
            args,
            str(wrapper),
            NONCE,
            hermes_home="/unrelated/hermes-home",
            profile=PROFILE,
            allow_spawn_proof=False,
        )
        == "OWNED"
    )


def test_unrelated_python_is_foreign():
    module = load_classifier()
    args = [
        "/usr/bin/python3",
        "/opt/other/app.py",
        "serve",
        "--isolated",
        "--ssh-owner-nonce",
        NONCE,
    ]
    assert (
        module.classify_dashboard_argv(
            args, WRAPPER, NONCE, hermes_home=HERMES_HOME, profile=""
        )
        == "FOREIGN"
    )


def test_nonce_and_isolated_without_entry_or_spawn_proof_is_foreign():
    module = load_classifier()
    args = this_host_args()
    assert (
        module.classify_dashboard_argv(
            args,
            WRAPPER,
            NONCE,
            hermes_home="",
            token_path="",
            profile=PROFILE,
            allow_spawn_proof=False,
            include_repo_hermes_entry=False,
            resolve_wrapper=False,
        )
        == "FOREIGN"
    )


def test_sabotage_dropping_repo_hermes_entry_fails_this_host_without_spawn_proof():
    module = load_classifier()
    result = module.classify_dashboard_argv(
        this_host_args(),
        WRAPPER,
        NONCE,
        hermes_home=HERMES_HOME,
        token_path="",
        profile=PROFILE,
        allow_spawn_proof=False,
        include_repo_hermes_entry=False,
        resolve_wrapper=True,
    )
    assert result == "FOREIGN"


def test_sabotage_dropping_only_wrapper_resolve_fails_wrapper_fixture(tmp_path: Path):
    module = load_classifier()
    wrapper = tmp_path / "hermes"
    script = tmp_path / "custom-hermes"
    script.write_text("# entry\n", encoding="utf-8")
    wrapper.write_text(
        f'#!/usr/bin/env bash\nexec "{PYTHON}" "{script}" "$@"\n',
        encoding="utf-8",
    )
    args = [
        PYTHON,
        str(script),
        "--profile",
        PROFILE,
        "serve",
        "--isolated",
        "--ssh-owner-nonce",
        NONCE,
    ]
    result = module.classify_dashboard_argv(
        args,
        str(wrapper),
        NONCE,
        hermes_home="/unrelated/hermes-home",
        profile=PROFILE,
        allow_spawn_proof=False,
        resolve_wrapper=False,
    )
    assert result == "FOREIGN"


def test_existing_venv_entrypoint_stays_owned_without_new_repo_entry():
    module = load_classifier()
    args = [
        PYTHON,
        VENV_HERMES,
        "--profile",
        PROFILE,
        "serve",
        "--isolated",
        "--ssh-owner-nonce",
        NONCE,
    ]
    assert (
        module.classify_dashboard_argv(
            args,
            WRAPPER,
            NONCE,
            hermes_home=HERMES_HOME,
            profile=PROFILE,
            allow_spawn_proof=False,
            include_repo_hermes_entry=False,
        )
        == "OWNED"
    )


def test_spawn_proof_still_owns_this_host_when_new_entries_disabled():
    module = load_classifier()
    result = module.classify_dashboard_argv(
        this_host_args(),
        WRAPPER,
        NONCE,
        hermes_home=HERMES_HOME,
        token_path=TOKEN,
        profile=PROFILE,
        allow_spawn_proof=True,
        include_repo_hermes_entry=False,
        resolve_wrapper=False,
    )
    assert result == "OWNED"


def test_remote_lifecycle_embeds_classifier_source():
    ts = (REPO_ROOT / "apps" / "desktop" / "electron" / "remote-lifecycle.ts").read_text(
        encoding="utf-8"
    )
    py = CLASSIFIER_PATH.read_text(encoding="utf-8")
    assert py in ts, "pidIsOurDashboard must embed ssh_pid_owned.py verbatim"
