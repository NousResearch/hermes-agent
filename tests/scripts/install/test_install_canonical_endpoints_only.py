"""Guard: the install path only ever targets canonical endpoints.

The installer must never ship, select, or fall back on a third-party mirror.
A built-in mirror table (package index, GitHub proxy, Electron mirror) silently
redirects every user's download through an operator the project has no
relationship with: it is a supply-chain risk for the user and unbounded load
for that operator. A user who needs a mirror opts in through the standard
knobs (``UV_INDEX_URL`` / ``PIP_INDEX_URL``, git's ``url.<base>.insteadOf``,
``ELECTRON_MIRROR``, ``HTTPS_PROXY``), and those must pass through untouched.

Coverage = every file that runs during install or update.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

INSTALL_PATH_FILES = (
    "scripts/install.sh",
    "scripts/install.ps1",
    "scripts/install.cmd",
    "setup-hermes.sh",
    "hermes_cli/managed_uv.py",
)

# Hosts the install path is allowed to talk to. Canonical upstreams and the
# project's own domain only. A third-party mirror NEVER belongs here -- if a
# host is not an upstream vendor or the project's own, the right answer is to
# drop it and let the user opt in, not to widen this list.
CANONICAL_HOSTS = frozenset(
    {
        "astral.sh",  # uv installer
        "docs.astral.sh",
        "duckduckgo.com",  # network reachability probe only
        "git-scm.com",
        "github.com",  # canonical source + release host
        "hermes-agent.nousresearch.com",  # project's own bootstrap domain
        "nodejs.org",
        "pypi.org",  # canonical package index
        "raw.githubusercontent.com",
    }
)

# Names of the mirror/proxy families this code must never carry again, checked
# against the raw text (not just URLs) so a mention in a comment or a message
# cannot slip through either.
BANNED_NAMES = (
    "npmmirror",
    "ghfast",
    "gh-proxy",
    "ghproxy",
    "mirrors.aliyun",
    "mirrors.tuna",
    "mirrors.ustc",
    "mirrors.cloud.tencent",
    "mirrors.huaweicloud",
    "registry.npmmirror",
)

HOST_RE = re.compile(r"https?://([A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)")

# Endpoint variables the user owns. The install path may read and mention them,
# never clear or override them, and never pass a conflicting CLI flag.
ENDPOINT_VARS = (
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)
ENDPOINT_VAR_ALTERNATION = "|".join(ENDPOINT_VARS)
FORBIDDEN_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:"
    r"(?:export\s+)?(?:%s)\s*="  # shell assignment
    r"|(?:unset|Remove-Item)\b.*(?:%s)"  # shell / PowerShell clearing
    r"|\$env:(?:%s)\s*="  # PowerShell assignment
    r")"
    % (ENDPOINT_VAR_ALTERNATION, ENDPOINT_VAR_ALTERNATION, ENDPOINT_VAR_ALTERNATION)
)
FORBIDDEN_FLAGS = ("--index-url", "--extra-index-url", "--default-index")


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _block_after(text: str, sentinel: str, lines: int = 12) -> str:
    """The `lines` lines starting at `sentinel` -- so an assertion is scoped to
    the failure path that prints it, not the file at large."""
    index = text.find(sentinel)
    assert index != -1, f"install path no longer contains {sentinel!r}"
    return "\n".join(text[index:].splitlines()[:lines])


def _block_around(text: str, sentinel: str, before: int = 8, after: int = 4) -> str:
    """The guidance printed around `sentinel` (messages that precede a `throw`
    belong to it)."""
    index = text.find(sentinel)
    assert index != -1, f"install path no longer contains {sentinel!r}"
    start = max(0, len(text[:index].splitlines()) - before)
    return "\n".join(text.splitlines()[start : start + before + after])


def test_install_path_references_only_canonical_hosts() -> None:
    offenders: dict[str, set[str]] = {}
    for relative in INSTALL_PATH_FILES:
        for host in HOST_RE.findall(_read(relative)):
            if host not in CANONICAL_HOSTS:
                offenders.setdefault(relative, set()).add(host)

    assert not offenders, (
        "The install path points at non-canonical hosts. A third-party mirror "
        "or proxy must not be built into the installer -- users opt in with "
        "UV_INDEX_URL / PIP_INDEX_URL / ELECTRON_MIRROR / git insteadOf "
        f"instead. Offending hosts: {offenders}"
    )


def test_install_path_carries_no_mirror_or_proxy_names() -> None:
    for relative in INSTALL_PATH_FILES:
        text = _read(relative)
        for name in BANNED_NAMES:
            assert name not in text, (
                f"{relative} mentions {name!r}. Mirror and proxy services must "
                "not be hardcoded into software everyone installs."
            )


def test_install_path_never_sets_or_clears_user_endpoints() -> None:
    for relative in INSTALL_PATH_FILES:
        for number, line in enumerate(_read(relative).splitlines(), 1):
            assert not FORBIDDEN_ASSIGNMENT_RE.search(line), (
                f"{relative}:{number} overrides or clears a user endpoint "
                f"variable: {line.strip()!r}. The user's own index must win."
            )
            for flag in FORBIDDEN_FLAGS:
                assert flag not in line, (
                    f"{relative}:{number} passes {flag} to a package manager: "
                    f"{line.strip()!r}. That overrides whatever the user chose."
                )


_SH_LOCKED_SYNC_START = "run_locked_uv_sync() {\n"


def _sh_locked_sync_helper(relative: str = "scripts/install.sh") -> str:
    text = _read(relative)
    _, marker, rest = text.partition(_SH_LOCKED_SYNC_START)
    assert marker, f"{relative} is missing run_locked_uv_sync()"
    body, end, _ = rest.partition("\n}\n")
    assert end, f"{relative} has an unterminated run_locked_uv_sync()"
    return marker + body + end


def _bash_path(path: Path) -> str:
    # MSYS/git-bash and POSIX both accept the forward-slash native form.
    return path.as_posix()


def test_locked_sync_passes_user_index_through_untouched(tmp_path: Path) -> None:
    """The hash-verified tier must inherit the index the user configured."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")

    record = tmp_path / "uv-env.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'UV_INDEX_URL=%s\\n' "${UV_INDEX_URL:-UNSET}"
            printf 'PIP_INDEX_URL=%s\\n' "${PIP_INDEX_URL:-UNSET}"
            printf 'args=%s\\n' "$*"
            """,
        ),
        encoding="utf-8",
        newline="\n",
    )

    harness = tmp_path / "harness.sh"
    harness.write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            set -eu
            UV_CMD="sh $1"
            export UV_CMD
            """,
        )
        + _sh_locked_sync_helper()
        + textwrap.dedent(
            """\
            run_locked_uv_sync /tmp/hermes-venv
            test "$UV_INDEX_URL" = "https://user-index.test/simple/"
            """,
        ),
        encoding="utf-8",
        newline="\n",
    )

    env = {
        **os.environ,
        "UV_INDEX_URL": "https://user-index.test/simple/",
        "PIP_INDEX_URL": "https://user-pip-index.test/simple/",
    }
    result = subprocess.run(
        [bash, _bash_path(harness), _bash_path(fake_uv)],
        env={**env, "RECORD": str(record)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "UV_INDEX_URL=https://user-index.test/simple/" in result.stdout
    assert "PIP_INDEX_URL=https://user-pip-index.test/simple/" in result.stdout
    assert "args=sync --extra all --locked" in result.stdout


def test_failed_unix_install_points_at_the_standard_opt_in() -> None:
    text = _read("scripts/install.sh")

    deps = _block_after(text, "Package installation failed even with no extras.")
    assert "UV_INDEX_URL" in deps, "the pip/uv failure hint must name UV_INDEX_URL"
    assert "PIP_INDEX_URL" in deps, "the failure hint must name PIP_INDEX_URL"

    clone = _block_after(text, "Failed to clone repository")
    assert "insteadOf" in clone, (
        "a blocked github.com must point at git's url.<base>.insteadOf opt-in"
    )

    locked = _block_after(text, "uv.lock is pinned to the canonical PyPI registry", 4)
    assert "UV_INDEX_URL" in locked, (
        "when a custom index is set, the locked tier must say why --locked "
        "cannot use it instead of blaming a stale lockfile"
    )


def test_failed_windows_install_points_at_the_standard_opt_in() -> None:
    text = _read("scripts/install.ps1")

    deps = _block_around(
        text, "Failed to install hermes-agent package even with no extras."
    )
    assert "UV_INDEX_URL" in deps, "the pip/uv failure hint must name UV_INDEX_URL"
    assert "PIP_INDEX_URL" in deps, "the failure hint must name PIP_INDEX_URL"

    clone = _block_around(text, "Failed to download repository (tried git clone")
    assert "insteadOf" in clone, (
        "a blocked github.com must point at git's url.<base>.insteadOf opt-in"
    )

    locked = _block_after(text, "uv.lock is pinned to the canonical PyPI registry", 4)
    assert "UV_INDEX_URL" in locked, (
        "when a custom index is set, the locked tier must say why --locked "
        "cannot use it instead of blaming a stale lockfile"
    )


def test_docs_document_opt_in_without_shipping_a_mirror_table() -> None:
    install_doc = _read("website/docs/getting-started/installation.md")
    assert "### Package and source mirrors (opt-in)" in install_doc
    section = _block_after(
        install_doc, "### Package and source mirrors (opt-in)", 16
    ).lower()
    for knob in ("uv_index_url", "pip_index_url", "insteadof", "electron_mirror"):
        assert knob in section, f"the mirror doc must document {knob}"

    for relative in (
        "website/docs/getting-started/installation.md",
        "website/docs/user-guide/desktop.md",
    ):
        text = _read(relative)
        for name in BANNED_NAMES:
            assert name not in text, (
                f"{relative} names {name!r}; docs must use a placeholder "
                "(<mirror-base-url>) so nobody is pointed at a third party."
            )
