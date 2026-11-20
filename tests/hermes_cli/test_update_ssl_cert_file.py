"""SSL_CERT_FILE → GIT_SSL_CAINFO for the updater's git children (#124654).

git's libcurl ignores ``SSL_CERT_FILE``; behind a TLS-inspecting proxy whose root lives only
there, the channel read (Python) passed and the next ``git fetch`` failed. The mapping lands in
``os.environ`` so the partial clone's lazy promisor fetches inherit it too.
"""

import os
import subprocess

from hermes_cli import update_cmd


def _git_config(configured: str | None):
    def run(cmd, args, **kwargs):
        assert args[:3] == ["config", "--get", "http.sslCAInfo"], args
        return subprocess.CompletedProcess(cmd, 0 if configured else 1, configured or "", "")
    return run


def test_ssl_cert_file_reaches_network_git_children(monkeypatch, tmp_path):
    bundle = str(tmp_path / "corp-bundle.pem")
    monkeypatch.setenv("SSL_CERT_FILE", bundle)
    monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
    monkeypatch.setattr(update_cmd, "_git_run", _git_config(None))
    update_cmd._map_ssl_cert_file_for_git(["git"])
    assert update_cmd._no_prompt_git_kwargs()["env"]["GIT_SSL_CAINFO"] == bundle
    assert os.environ["GIT_SSL_CAINFO"] == bundle


def test_configured_git_ca_is_not_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "corp-bundle.pem"))
    monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
    monkeypatch.setattr(update_cmd, "_git_run", _git_config(str(tmp_path / "configured.pem")))
    update_cmd._map_ssl_cert_file_for_git(["git"])
    assert "GIT_SSL_CAINFO" not in os.environ
