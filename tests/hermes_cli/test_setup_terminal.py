"""Tests for hermes_cli/setup_terminal.py backend wizards."""

import pytest

from hermes_cli.config import save_env_value, get_env_value
from hermes_cli import setup as setup_mod
from hermes_cli import setup_terminal


@pytest.fixture
def ssh_wizard(tmp_path, monkeypatch):
    """Drive ``_setup_backend_ssh`` with scripted answers, SSH test declined."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TERMINAL_SSH_PORT", raising=False)

    def run(answers):
        it = iter(answers)
        monkeypatch.setattr(setup_mod, "prompt", lambda label, default="": next(it))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda label, default=True: False)
        setup_terminal._setup_backend_ssh({})

    return run


def test_ssh_port_reset_to_22_removes_saved_port(ssh_wizard):
    # A stored non-default port must be removable: answering "22" restores the
    # default, it does not silently leave the old value in .env.
    save_env_value("TERMINAL_SSH_PORT", "2222")
    ssh_wizard(["host.example.com", "me", "22", "/tmp/key"])
    assert get_env_value("TERMINAL_SSH_PORT") is None


def test_ssh_port_non_default_still_saves(ssh_wizard):
    # Control: the deliberate skip is only for the default; a real port persists.
    ssh_wizard(["host.example.com", "me", "2222", "/tmp/key"])
    assert get_env_value("TERMINAL_SSH_PORT") == "2222"


def test_ssh_host_pasted_with_quotes_saves_unquoted(ssh_wizard, capsys):
    # The SSH loop shares the quote-stripping contract with the secret prompt:
    # a quoted host paste must reach .env unquoted, warned via the shared helper.
    ssh_wizard(['"10.0.0.5"', "me", "22", "/tmp/key"])
    assert get_env_value("TERMINAL_SSH_HOST") == "10.0.0.5"
    out = capsys.readouterr().out
    assert "Removed the surrounding quotes" in out
    assert "nothing saved" not in out


@pytest.fixture
def secret_env_home(tmp_path, monkeypatch):
    """Isolate .env writes to tmp_path; keep MODAL_* out of os.environ between tests."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_secret_prompt_strips_pasted_quotes(secret_env_home, monkeypatch, capsys):
    # A token pasted with surrounding quotes is a paste artifact; saving it verbatim
    # makes save_env_value escape the quotes, so the loader hands back the quoted
    # string and it silently outranks the vendor's own config (#47264).
    monkeypatch.setattr(setup_mod, "prompt", lambda label, default=None, password=False: '"ak-quoted"')
    setup_terminal._prompt_secret_env("  Modal Token ID", "MODAL_TOKEN_ID")

    from agent.secret_scope import load_env_file
    loaded = load_env_file(secret_env_home / ".env")["MODAL_TOKEN_ID"]
    assert loaded == "ak-quoted"
    assert '"' not in loaded and "'" not in loaded
    assert get_env_value("MODAL_TOKEN_ID") == "ak-quoted"
    assert "Removed the surrounding quotes" in capsys.readouterr().out

    # Quotes-only strips to empty — the same "skip" contract as an empty answer,
    # so nothing is saved and the user is told.
    monkeypatch.setattr(setup_mod, "prompt", lambda label, default=None, password=False: '""')
    setup_terminal._prompt_secret_env("  Modal Token Secret", "MODAL_TOKEN_SECRET")
    assert get_env_value("MODAL_TOKEN_SECRET") is None
    assert "nothing saved" in capsys.readouterr().out


def test_strip_pasted_quotes_contract():
    # Pure contract of the helper shared by every credential/.env prompt site.
    assert setup_terminal._strip_pasted_quotes('"ak-x"') == ("ak-x", True)
    # Doubly-quoted paste: one surviving pair would still be rejected by the
    # vendor, so every matching outer pair must come off.
    assert setup_terminal._strip_pasted_quotes('""ak-x""') == ("ak-x", True)
    assert setup_terminal._strip_pasted_quotes("''ak-x''") == ("ak-x", True)
    assert setup_terminal._strip_pasted_quotes("'ak-x'") == ("ak-x", True)
    assert setup_terminal._strip_pasted_quotes('"ak-x') == ('"ak-x', False)
    assert setup_terminal._strip_pasted_quotes("ak-x") == ("ak-x", False)
    assert setup_terminal._strip_pasted_quotes('""') == ("", True)
    # Whitespace inside artifact quotes is artifact too once the quotes come off:
    # a quoted space stores nothing, a quoted padded value stores the trimmed value.
    assert setup_terminal._strip_pasted_quotes('" "') == ("", True)
    assert setup_terminal._strip_pasted_quotes('" ak-x "') == ("ak-x", True)
    # Unquoted input is returned untouched, whitespace included.
    assert setup_terminal._strip_pasted_quotes(' ') == (' ', False)
    # Length guard: a lone quote character is the value itself, not a paste artifact.
    assert setup_terminal._strip_pasted_quotes('"') == ('"', False)
    # Odd run of quotes: after one pair comes off, a lone surviving quote is a paste
    # artifact, not a credential — it must not reach .env as a truthy leftover.
    assert setup_terminal._strip_pasted_quotes('"""') == ("", True)
    assert setup_terminal._strip_pasted_quotes("'''") == ("", True)


def test_vercel_token_pasted_with_quotes_saves_unquoted(tmp_path, monkeypatch, capsys):
    # The Vercel auth loop is the prompt site the shared helper also guards; drive
    # the whole wizard so deleting the helper call there cannot keep the suite green.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # keep a stray .vercel/project.json out of the defaults
    for key in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID", "TERMINAL_VERCEL_RUNTIME"):
        monkeypatch.delenv(key, raising=False)

    answers = {"    Vercel access token": '"vc-quoted"'}
    # Non-token prompts (runtime/persist/cpu/memory/project/team) answer with their
    # defaults, so the quoted token is the only value the strip logic can touch.
    monkeypatch.setattr(setup_mod, "prompt",
                        lambda label, default="", password=False: answers.get(label, default))
    setup_terminal._prompt_vercel_sandbox_settings({"terminal": {}})

    assert get_env_value("VERCEL_TOKEN") == "vc-quoted"
    out = capsys.readouterr().out
    assert "Removed the surrounding quotes" in out
    assert "nothing saved" not in out


def test_docker_wizard_reports_the_resolved_podman_runtime(monkeypatch, capsys):
    """Podman satisfies the docker backend, so the wizard must name it instead of warning."""
    monkeypatch.setattr(setup_terminal, "find_docker", lambda: "/usr/bin/podman")
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda label, default=True: False)

    setup_terminal._setup_backend_docker({"terminal": {}})

    out = capsys.readouterr().out
    assert "Podman found: /usr/bin/podman" in out
    assert "not found in PATH" not in out
