"""hc-472 followup: CN mirror index env silently defeats hash-verified installs.

``apexnodes_apply_cn_mirror_env`` (scripts/lib/apexnodes-region-detect.sh) /
``Set-ApexCnMirrorEnv`` (scripts/lib/apexnodes-region-detect.ps1) export a CN
mirror default package index (``UV_DEFAULT_INDEX`` etc, Tsinghua TUNA) when
``HERMES_CN_MIRRORS=1``. ``uv.lock`` records the ACTUAL registry each package
was resolved against (``source = { registry = "https://pypi.org/simple" }``),
so once that env is active, ``uv sync --extra all --locked`` is a DIFFERENT
identity than the lock expects and uv refuses outright with "The lockfile ...
needs to be updated" -- confirmed against the vendored uv 0.11 by running
``uv sync --extra all --locked --dry-run`` with the mirror env set: it exits
1 with exactly that message, after a real ~30s network re-resolve against the
mirror (vs. ~0.2s and zero network calls when the index matches the lock).

install.sh's ``install_deps`` / install.ps1's ``Install-Dependencies`` treat
that failure as "lockfile stale" and fall through to the tiered
``uv pip install`` fallback -- which has NO lock and NO hash verification at
all. Net effect: every CN-mirror install (the region this repo goes out of
its way to route through mirrors for) silently skipped hash verification,
every time, not just on a stale lock.

scripts/build-runtime-bundle.mjs already discovered and fixed this for the
*bundle build* path (it deletes the index-selecting env before its own
``uv sync --locked``). This test pins the same fix applied to the actual
*installer* consumption path: ``_uv_sync_locked`` (install.sh) /
``Invoke-UvSyncLocked`` (install.ps1) strip ONLY the package-index-selecting
env for the ``--locked`` calls specifically, leaving every other CN mirror
(npm/node/electron/playwright, and the unlocked pip-fallback tiers, which
DO benefit from a mirror since they re-resolve fresh) untouched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
BUILD_BUNDLE = REPO_ROOT / "scripts" / "build-runtime-bundle.mjs"

# The env vars that select/redirect uv's package index. Must match the set
# scripts/build-runtime-bundle.mjs already strips for its own --locked sync.
_INDEX_ENV_VARS = (
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)


def _extract_install_sh_function(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?\n\}}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def _extract_install_ps1_function(name: str) -> str:
    text = INSTALL_PS1.read_text()
    match = re.search(rf"function {re.escape(name)} \{{(?P<body>[\s\S]*?)^\}}", text, re.MULTILINE)
    assert match is not None, f"function {name} not found in install.ps1"
    return match.group("body")


# ---------------------------------------------------------------------------
# Static: the sanitizing helpers exist and strip the right (and only the
# right) env vars.
# ---------------------------------------------------------------------------


def test_install_sh_uv_sync_locked_helper_strips_every_index_env_var() -> None:
    body = _extract_install_sh_function("_uv_sync_locked")
    for var in _INDEX_ENV_VARS:
        assert f"-u {var}" in body, (
            f"_uv_sync_locked must strip {var} (env -u {var}) before the "
            "--locked uv sync, or a CN mirror default index re-keys the "
            "lock's recorded registry and --locked always refuses"
        )
    assert "--locked" in body
    assert 'UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv"' in body


def test_install_ps1_invoke_uv_sync_locked_helper_strips_every_index_env_var() -> None:
    body = _extract_install_ps1_function("Invoke-UvSyncLocked")
    for var in _INDEX_ENV_VARS:
        assert f"$env:{var} = $null" in body, (
            f"Invoke-UvSyncLocked must clear $env:{var} before the --locked "
            "uv sync, or a CN mirror default index re-keys the lock's "
            "recorded registry and --locked always refuses"
        )
        # Must also be RESTORED afterward (finally) -- other CN-mirror
        # consumers later in Install-Dependencies still need it.
        assert f"$env:{var} = $savedIndexEnv.{var}" in body, (
            f"Invoke-UvSyncLocked must restore {var} in its finally block "
            "so later non-locked tiers keep the CN mirror"
        )
    assert "finally" in body
    assert "--locked" in body


# ---------------------------------------------------------------------------
# Static: both --locked call sites in each installer actually route through
# the sanitizing helper (not the raw, unsanitized invocation).
# ---------------------------------------------------------------------------


def test_install_sh_call_sites_route_through_uv_sync_locked() -> None:
    fn = _extract_install_sh_function("install_deps")
    assert "_uv_sync_locked --check" in fn, (
        "the fast-path 'already satisfied' probe must go through "
        "_uv_sync_locked so a CN mirror index can't make it a permanent "
        "false negative"
    )
    assert "if _uv_sync_locked; then" in fn, (
        "the hash-verified tier must go through _uv_sync_locked"
    )
    # The pre-fix raw literal must not remain as a live call site (it still
    # legitimately appears once, inside _uv_sync_locked's own definition,
    # which is outside this extracted install_deps() body).
    raw = 'UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv" $UV_CMD sync --extra all --locked'
    assert raw not in fn, (
        "install_deps() must not call uv sync --locked directly -- route "
        "through _uv_sync_locked so the index env is sanitized"
    )


def test_install_ps1_call_sites_route_through_invoke_uv_sync_locked() -> None:
    fn = _extract_install_ps1_function("Install-Dependencies")
    assert "Invoke-UvSyncLocked -Check" in fn, (
        "the fast-path 'already satisfied' probe must go through "
        "Invoke-UvSyncLocked so a CN mirror index can't make it a "
        "permanent false negative"
    )
    # The bare (non -Check) call for the hash-verified tier: a standalone
    # line naming only the function, no -Check switch.
    assert re.search(r"^\s*Invoke-UvSyncLocked\s*$", fn, re.MULTILINE), (
        "the hash-verified tier must call bare Invoke-UvSyncLocked (no "
        "-Check) on its own line"
    )
    raw = "& $UvCmd sync --extra all --locked"
    assert raw not in fn, (
        "Install-Dependencies must not call uv sync --locked directly -- "
        "route through Invoke-UvSyncLocked so the index env is sanitized"
    )


def test_build_runtime_bundle_and_installers_agree_on_the_index_env_list() -> None:
    """Keep the env-var list in step with the bundle-build fix that first
    discovered this (scripts/build-runtime-bundle.mjs)."""
    bundle_src = BUILD_BUNDLE.read_text()
    for var in _INDEX_ENV_VARS:
        assert f"'{var}'" in bundle_src, (
            f"build-runtime-bundle.mjs no longer strips {var} -- update "
            "_uv_sync_locked / Invoke-UvSyncLocked (and this test) to match, "
            "or update the bundle script if the var list is intentionally "
            "narrower there"
        )


# ---------------------------------------------------------------------------
# Behavioral: _uv_sync_locked actually strips the index env at run time (and
# leaves unrelated CN-mirror env alone) -- exercised against a fake `uv`.
# Only these two need bash; the static tests above run everywhere.
# ---------------------------------------------------------------------------

_needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@_needs_bash
def test_uv_sync_locked_strips_index_env_but_preserves_other_mirrors_and_args(
    tmp_path: Path,
) -> None:
    fake_uv = tmp_path / "fake-uv.sh"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'echo "ARGS:$*"\n'
        'echo "UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX-<unset>}"\n'
        'echo "UV_INDEX_URL=${UV_INDEX_URL-<unset>}"\n'
        'echo "UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL-<unset>}"\n'
        'echo "UV_INDEX=${UV_INDEX-<unset>}"\n'
        'echo "PIP_INDEX_URL=${PIP_INDEX_URL-<unset>}"\n'
        'echo "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL-<unset>}"\n'
        'echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT-<unset>}"\n'
        'echo "npm_config_registry=${npm_config_registry-<unset>}"\n'
        "exit 0\n"
    )
    fake_uv.chmod(0o755)

    script = (
        "set -e\n"
        f'INSTALL_DIR="{tmp_path}"\n'
        f'UV_CMD="{fake_uv}"\n'
        # Simulate apexnodes_apply_cn_mirror_env having already run (as it
        # does, early in install.sh) before install_deps() ever calls this.
        'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"\n'
        'export UV_INDEX_URL="$UV_DEFAULT_INDEX"\n'
        'export UV_EXTRA_INDEX_URL="https://example.invalid/extra/simple"\n'
        'export UV_INDEX="pypi=$UV_DEFAULT_INDEX"\n'
        'export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"\n'
        'export PIP_EXTRA_INDEX_URL="https://example.invalid/extra/simple"\n'
        # A DIFFERENT class of CN-mirror env (non-index) that other tiers
        # still need -- must NOT be touched by this helper.
        'export npm_config_registry="https://registry.npmmirror.com"\n'
        f"{_extract_install_sh_function('_uv_sync_locked')}\n"
        "_uv_sync_locked --check\n"
    )
    res = subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "ARGS:sync --extra all --locked --check" in out, (
        f"expected the fixed 'sync --extra all --locked' plus the forwarded "
        f"--check arg, got: {out!r}"
    )
    for var in _INDEX_ENV_VARS:
        assert f"{var}=<unset>" in out, (
            f"{var} leaked into the --locked uv invocation: {out}"
        )
    assert f"UV_PROJECT_ENVIRONMENT={tmp_path}/venv" in out
    assert "npm_config_registry=https://registry.npmmirror.com" in out, (
        "the helper must scope its strip to the package-index env only -- "
        "unrelated CN mirrors (npm/node/electron/playwright) stay set for "
        "every other call site"
    )


@_needs_bash
def test_uv_sync_locked_restores_index_env_for_subsequent_commands(
    tmp_path: Path,
) -> None:
    """The strip must be scoped to the one invocation -- the unlocked
    fallback tiers later in install_deps() still want the CN mirror."""
    fake_uv = tmp_path / "fake-uv.sh"
    fake_uv.write_text("#!/bin/bash\nexit 1\n")  # force the --locked tier to "fail"
    fake_uv.chmod(0o755)

    script = (
        f'INSTALL_DIR="{tmp_path}"\n'
        f'UV_CMD="{fake_uv}"\n'
        'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"\n'
        f"{_extract_install_sh_function('_uv_sync_locked')}\n"
        "_uv_sync_locked --check || true\n"
        'echo "AFTER=${UV_DEFAULT_INDEX-<unset>}"\n'
    )
    res = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "AFTER=https://pypi.tuna.tsinghua.edu.cn/simple" in res.stdout, (
        "a subshell-scoped strip (env -u ... cmd) must not leak into the "
        "calling shell's exported env -- the fallback uv pip install tiers "
        "run later in the SAME shell and still need the mirror"
    )
