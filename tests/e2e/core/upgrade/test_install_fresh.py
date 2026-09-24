"""Fresh install through the real `scripts/install.sh` into an empty HOME, then a re-run.

The installer is HEAD's own copy, run the way the documented one-liner runs it (non-interactive,
stdin at EOF, setup wizard skipped) inside the bwrap sandbox with an empty fake HOME that has
only a fresh distro's dotfiles. Its git clone is redirected to a local bare clone of this
checkout (``_install_helpers``); uv comes from the host's warm cache. After the install the
user's shell resolves ``hermes`` from ``~/.local/bin`` (the installer wires PATH into the
shell rc), ``hermes --version`` answers from the installed checkout, and a one-shot turn reaches
the (fake) provider and is persisted.

Re-running the installer on that live install (what users do to "repair" or to update) must be
idempotent: same checkout, user edits to config/.env/SOUL and their sessions byte-identical,
the PATH line not appended twice, and the install still works.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.linux_only,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

RC_FILES = (".bashrc", ".profile")
# A fresh distro's skeleton dotfiles (as /etc/skel ships them): .bashrc bails out early for
# non-interactive shells, .profile sources it.
SKEL = {
    ".bashrc": "# ~/.bashrc\ncase $- in *i*) ;; *) return;; esac\nHISTSIZE=1000\n",
    ".profile": '# ~/.profile\nif [ -n "$BASH_VERSION" ] && [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n',
}


def _path_lines(sb: I.Sandbox) -> dict[str, int]:
    """How many lines in each shell rc put ~/.local/bin on PATH."""
    out = {}
    for name in RC_FILES:
        text = (sb.home / name).read_text(encoding="utf-8")
        out[name] = sum(1 for line in text.splitlines()
                        if ".local/bin" in line and "PATH" in line and not line.lstrip().startswith("#"))
    return out


def _login_shell_hermes(sb: I.Sandbox) -> str:
    """Resolve `hermes` the way a new login shell would: PATH comes only from the rc files."""
    env = dict(sb.env)
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    cp = H.run(["bash", "-lc", "command -v hermes"], env=env, cwd=sb.root, writable=[sb.root], timeout=60)
    return cp.stdout.strip()


def _state(sb: I.Sandbox) -> dict:
    hh = sb.hermes_home
    return {
        "config": (hh / "config.yaml").read_bytes(),
        "env": (hh / ".env").read_bytes(),
        "soul": (hh / "SOUL.md").read_bytes(),
        "db": I.db_state(hh / "state.db"),
        "custom_skill": I.tree_digest(hh / "skills" / "my-own-skill"),
    }


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the install suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def installed(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("fresh-install")
    origin = I.make_origin(root, I.head_sha())
    sb = I.new_sandbox(root / "sb", origin)
    for name, text in SKEL.items():
        (sb.home / name).write_text(text, encoding="utf-8")
    first = I.run_installer(sb)
    return sb, first


def _turn(sb: I.Sandbox, provider: FakeLLMServer, marker: str) -> None:
    n = len(provider.main_requests())
    cp = sb.cli("-z", marker)
    assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    assert provider.default_text in cp.stdout, I.describe(cp)
    new = provider.main_requests()[n:]
    assert len(new) == 1 and marker in json.dumps(new[0]["messages"]), "one-shot turn did not reach the provider once"


def _configure(sb: I.Sandbox, provider: FakeLLMServer) -> None:
    py = str(sb.checkout / "venv" / "bin" / "python")
    ver = sb.run([py, "-c", "from hermes_cli.config_defaults import DEFAULT_CONFIG as D; print(D['_config_version'])"])
    assert ver.returncode == 0, I.describe(ver)
    version = int(ver.stdout.strip().splitlines()[-1])
    (sb.hermes_home / "config.yaml").write_text(I.provider_config(provider.base_url, version), encoding="utf-8")
    (sb.hermes_home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}\n# my note\n", encoding="utf-8")
    (sb.hermes_home / "SOUL.md").write_text("You are my own customised agent.\n", encoding="utf-8")
    skill = sb.hermes_home / "skills" / "my-own-skill"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: mine\n---\nDo my thing.\n", encoding="utf-8")


def test_fresh_install_serves_head_and_runs_a_turn(installed, provider):
    sb, first = installed
    assert first.returncode == 0, "install.sh failed on an empty HOME:\n" + I.describe(first)
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == I.head_sha(), "installed checkout is not the published commit"
    assert I.git("status", "--porcelain", "--untracked-files=no", cwd=sb.checkout) == "", "installer dirtied the checkout"
    # A new login shell finds the command through the rc files the installer edited.
    assert _login_shell_hermes(sb) == sb.hermes, "a new shell does not resolve `hermes` to the installed launcher"
    assert all(n >= 1 for n in _path_lines(sb).values()), f"PATH not wired into the shell rc: {_path_lines(sb)}"
    ver = sb.cli("--version")
    assert ver.returncode == 0 and I.TRACEBACK not in ver.stdout + ver.stderr, I.describe(ver)
    probe = sb.run([str(sb.checkout / "venv" / "bin" / "python"), "-c", "import hermes_cli, run_agent; print(hermes_cli.__file__); print(run_agent.__file__)"])
    assert probe.returncode == 0, I.describe(probe)
    for line in probe.stdout.split():
        assert line.startswith(str(sb.checkout)), f"installed venv imports code from outside the checkout: {line}"
    for rel in ("config.yaml", ".env", "SOUL.md"):
        assert (sb.hermes_home / rel).is_file(), f"installer did not seed ~/.hermes/{rel}"
    _configure(sb, provider)
    _turn(sb, provider, "first turn on a fresh install")
    db = I.db_state(sb.hermes_home / "state.db")
    assert db["integrity"] == [("ok",)] and len(db["sessions"]) == 1 and db["n_messages"] >= 2, db


def test_rerunning_the_installer_is_idempotent(installed, provider):
    sb, first = installed
    assert first.returncode == 0, I.describe(first)
    if not (sb.hermes_home / "state.db").exists():
        _configure(sb, provider)
        _turn(sb, provider, "first turn on a fresh install")
    before = _state(sb)
    rc_before = _path_lines(sb)
    commit = I.git("rev-parse", "HEAD", cwd=sb.checkout)
    second = I.run_installer(sb)
    assert second.returncode == 0, "re-running install.sh on a working install failed:\n" + I.describe(second)
    assert re.search(r"(?i)traceback", second.stdout + second.stderr) is None, I.describe(second)
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == commit, "re-run moved the checkout"
    after = _state(sb)
    for key in before:
        assert after[key] == before[key], f"re-running the installer changed the user's {key}"
    assert _path_lines(sb) == rc_before, f"PATH line appended again: {rc_before} -> {_path_lines(sb)}"
    assert _login_shell_hermes(sb) == sb.hermes
    _turn(sb, provider, "turn after re-running the installer")
    db = I.db_state(sb.hermes_home / "state.db")
    assert set(before["db"]["sessions"]) < set(db["sessions"]), "earlier session lost or new turn not persisted"
