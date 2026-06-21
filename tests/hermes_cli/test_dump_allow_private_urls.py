"""`hermes dump` must report the EFFECTIVE private-URL (SSRF) policy.

The runtime resolver ``tools.url_safety._global_allow_private_urls`` honors, in
priority order, the ``HERMES_ALLOW_PRIVATE_URLS`` env var, then the preferred
``security.allow_private_urls`` config key, then the legacy
``browser.allow_private_urls`` key.  The dump used to inspect only the legacy
config key, so a user who relaxed SSRF private-IP blocking via the env var or
the modern config key got a dump that hid the override entirely — implying a
locked-down posture while the agent was actually permitting private/internal-IP
fetches.  The dump now reports the effective value via the same resolver.
"""

from pathlib import Path
from types import SimpleNamespace


def _seed(home: Path, *, config_yaml: str = "", env_text: str = "") -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_yaml)
    (home / ".env").write_text(env_text)


def _run(monkeypatch, capsys, tmp_path) -> str:
    from hermes_cli import dump
    from tools import url_safety

    # Resolver caches process-wide; reset so case ordering can't leak.
    url_safety._reset_allow_private_cache()
    # Keep run_dump's project-.env fallback from touching the real repo.
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    dump.run_dump(SimpleNamespace(show_keys=False))
    return capsys.readouterr().out


def test_dump_surfaces_security_config_key(monkeypatch, capsys, tmp_path):
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    _seed(get_hermes_home(), config_yaml="security:\n  allow_private_urls: true\n")

    out = _run(monkeypatch, capsys, tmp_path)

    # Must report under the effective/preferred key, never the legacy one.
    assert "security.allow_private_urls" in out
    assert "browser.allow_private_urls" not in out
    assert "DISABLED" in out


def test_dump_surfaces_env_var(monkeypatch, capsys, tmp_path):
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    _seed(get_hermes_home(), env_text="HERMES_ALLOW_PRIVATE_URLS=true\n")

    out = _run(monkeypatch, capsys, tmp_path)

    # Even when the env var drives the effective policy, the dump must report it
    # under the preferred key, not the legacy one.
    assert "security.allow_private_urls" in out
    assert "browser.allow_private_urls" not in out
    assert "DISABLED" in out


def test_dump_silent_when_locked_down(monkeypatch, capsys, tmp_path):
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    _seed(get_hermes_home())

    out = _run(monkeypatch, capsys, tmp_path)

    assert "allow_private_urls" not in out


def test_dump_reports_unknown_when_resolver_errors(monkeypatch, capsys, tmp_path):
    """A resolver failure must surface as an explicit unknown/error value.

    Previously the resolver call was wrapped in ``except Exception: pass``, so a
    runtime failure resolving the effective SSRF policy was silently hidden —
    defeating the purpose of surfacing this posture during support/security
    triage. The dump must now report an explicit ``unknown (...)`` value.
    """
    from hermes_cli.config import get_hermes_home
    from tools import url_safety

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    _seed(get_hermes_home())

    def _boom() -> bool:
        raise RuntimeError("simulated resolver failure")

    monkeypatch.setattr(url_safety, "_global_allow_private_urls", _boom)

    out = _run(monkeypatch, capsys, tmp_path)

    assert "security.allow_private_urls" in out
    assert "unknown" in out
    assert "RuntimeError" in out
