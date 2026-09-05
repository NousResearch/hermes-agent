"""Regression tests for stage2-run.sh not exporting a dead npm_config_nodedir.

stage 1 (dev-sandbox.sh) resolves NODE_DIR from the HOST's ``command -v
node`` and stage 2 exported it as ``npm_config_nodedir`` unconditionally.
But stage 2's own mount plan hides most host paths: /usr/local is always
shadowed by the sandbox's near-empty copy, and anything outside the mounted
runtime prefixes (/nix, or /usr /bin /sbin /lib /lib64) does not exist
inside at all. On GitHub runners the host node is /usr/local/bin/node, so
every node-gyp build inside the sandbox failed with

    gyp: /usr/local/common.gypi not found (cwd: .../node_modules/node-pty)

which kept the Install & Update E2E installer legs red (see the
verification thread on #87635). With no nodedir at all, node-gyp downloads
version-matched headers through the sandbox proxy -- what a real install
does -- so the fix is to pass the nodedir through only when the directory
is actually visible inside the sandbox and contains the bundled headers.

The visibility decision itself is a pure function (``nodedir_visible``) at
the top of stage2-run.sh, table-tested here by sourcing the script. The
end-to-end tests drive the real stage2-run.sh with a fake ``bwrap`` on PATH
that records its argv, and assert on the --setenv plan it would have
executed.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2 = REPO_ROOT / "scripts" / "sandbox" / "stage2-run.sh"

pytestmark = pytest.mark.linux_only


def _run_stage2(
    tmp_path: Path, node_dir: str
) -> tuple[list[str], subprocess.CompletedProcess]:
    """Run stage2-run.sh against a minimal sandbox root and a fake bwrap.

    The fake bwrap dumps its argv (one arg per line) and exits, so the test
    observes exactly the mount/setenv plan stage 2 execs -- no namespaces,
    no network, no real bubblewrap needed.
    """
    sandbox = tmp_path / "sandbox"
    root = sandbox / "root"
    for sub in (
        "logs",
        "usr/local",
        "usr/bin",
        "bin",
        "lib64",
        "repo",
        "certs",
        "http",
    ):
        (root / sub).mkdir(parents=True)
    # Non-empty ready file skips the wait for the (absent) network helper.
    (root / "logs" / "slirp.ready").write_text("1\n", encoding="utf-8")
    (sandbox / "etc").mkdir()
    (sandbox / "home").mkdir()

    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    argv_file = tmp_path / "bwrap-argv"
    fake_bwrap = bindir / "bwrap"
    fake_bwrap.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > ' + str(argv_file) + "\nexit 0\n",
        encoding="utf-8",
    )
    fake_bwrap.chmod(0o755)

    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "DEV_SANDBOX_ROOT": str(sandbox),
        "DEV_SANDBOX_BASH": "/usr/bin/bash",
        "DEV_SANDBOX_INTERACTIVE": "false",
        "DEV_SANDBOX_USER": "hermes",
        "DEV_SANDBOX_HOME": "/home/hermes",
        "DEV_SANDBOX_NODE_DIR": node_dir,
    }
    proc = subprocess.run(
        ["bash", str(STAGE2), "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert argv_file.exists(), (
        f"stage2-run.sh never reached its bwrap exec "
        f"(rc={proc.returncode}):\n{proc.stderr}"
    )
    return argv_file.read_text(encoding="utf-8").splitlines(), proc


def _nodedir_settings(argv: list[str]) -> list[str]:
    """The values stage 2 planned to --setenv npm_config_nodedir to."""
    return [
        argv[i + 2]
        for i, arg in enumerate(argv)
        if arg == "--setenv" and argv[i + 1] == "npm_config_nodedir"
    ]


def _nodedir_visible(node_dir: str, use_host_runtime: bool) -> bool:
    """Call stage2-run.sh's nodedir_visible without running stage 2."""
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && nodedir_visible "$2" "$3"',
            "_",
            str(STAGE2),
            node_dir,
            "true" if use_host_runtime else "false",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode in (0, 1), proc.stderr
    return proc.returncode == 0


@pytest.mark.parametrize(
    ("node_dir", "host_runtime", "visible"),
    [
        # /usr/local: shadowed by the sandbox's own copy in both modes.
        ("/usr/local", True, False),
        ("/usr/local", False, False),
        ("/usr/local/n/versions/22", True, False),
        # Host mode ro-binds the runtime prefixes; nix mode does not.
        ("/usr", True, True),
        ("/usr/lib/node", True, True),
        ("/usr", False, False),
        # Nix mode binds /nix; host mode does not.
        ("/nix/store/abc-nodejs-22", False, True),
        ("/nix/store/abc-nodejs-22", True, False),
        # Nothing binds the host HOME (the sandbox HOME is a fresh directory),
        # hostedtoolcache, Homebrew, or a bare prefix that only shares a name.
        ("/home/dev/.nvm/versions/node/v22.0.0", True, False),
        ("/opt/hostedtoolcache/node/22.0.0/x64", True, False),
        ("/home/linuxbrew/.linuxbrew", True, False),
        ("/usrlocal", True, False),
        ("/nixos", False, False),
    ],
)
def test_nodedir_visible_table(
    node_dir: str, host_runtime: bool, visible: bool
) -> None:
    """Both branches of the visibility decision, hermetically."""
    assert _nodedir_visible(node_dir, host_runtime) is visible


def test_usr_local_nodedir_is_dropped(tmp_path: Path) -> None:
    """/usr/local is always shadowed inside the sandbox; a host node living
    there (GitHub runners) must not become npm_config_nodedir."""
    argv, _ = _run_stage2(tmp_path, "/usr/local")
    assert _nodedir_settings(argv) == [], (
        "stage 2 exported npm_config_nodedir=/usr/local, but the sandbox "
        "bind-mounts its own near-empty directory over /usr/local -- inside, "
        "node-gyp fails with 'gyp: /usr/local/common.gypi not found'"
    )


def test_unmounted_prefix_nodedir_is_dropped(tmp_path: Path) -> None:
    """A host node outside the mounted runtime prefixes (nvm, Homebrew,
    hostedtoolcache...) is invisible inside the sandbox, headers or not."""
    fake_node = tmp_path / "node-install"
    (fake_node / "include" / "node").mkdir(parents=True)
    (fake_node / "include" / "node" / "common.gypi").write_text(
        "{}\n", encoding="utf-8"
    )
    argv, _ = _run_stage2(tmp_path, str(fake_node))
    assert _nodedir_settings(argv) == [], (
        "stage 2 exported an npm_config_nodedir that no mount in its own "
        "plan makes visible inside the sandbox"
    )


def test_empty_nodedir_stays_absent(tmp_path: Path) -> None:
    """No host node resolved -> no nodedir, before and after the fix."""
    argv, _ = _run_stage2(tmp_path, "")
    assert _nodedir_settings(argv) == []


@pytest.mark.skipif(
    not os.path.isfile("/usr/include/node/common.gypi"),
    reason="host has no node headers under /usr (libnode-dev not installed)",
)
def test_visible_nodedir_with_headers_is_kept(tmp_path: Path) -> None:
    """A distro node under /usr is ro-bound into the sandbox and its headers
    are real: that nodedir must survive, preserving the offline build path.

    Opportunistic: needs root-owned headers, so it only runs on hosts that
    have them. The keep branch of the decision itself is covered
    hermetically by test_nodedir_visible_table; this proves the wiring."""
    argv, _ = _run_stage2(tmp_path, "/usr")
    assert _nodedir_settings(argv) == ["/usr"]
