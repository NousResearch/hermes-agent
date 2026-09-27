"""SSL_CERT_FILE → GIT_SSL_CAINFO mapping for the updater's git children.

git's libcurl ignores ``SSL_CERT_FILE``: GnuTLS builds do, and so do OpenSSL
builds with a compiled-in CA path. git reads only ``GIT_SSL_CAINFO`` /
``http.sslCAInfo``. Behind a TLS-inspecting proxy whose corporate root lives
only in ``SSL_CERT_FILE``, the channel read (Python honors it) succeeded and
the very next ``git fetch`` failed with "certificate signer not trusted",
reported as a "Network error". The updater must keep one CA story across all
of its git children — including the partial-clone checkout's lazy promisor
fetches, which inherit the process environment rather than
``_no_prompt_git_kwargs``.
"""

import subprocess

from hermes_cli import update_cmd


def _git_config_reader(configured: str | None):
    """A ``git`` double answering ``config --get http.sslCAInfo``."""
    calls = []

    def run(cmd, args, **kwargs):
        calls.append(args)
        if args[:3] == ["config", "--get", "http.sslCAInfo"]:
            out = configured if configured is not None else ""
            return subprocess.CompletedProcess(cmd, 0 if configured else 1, out, "")
        raise AssertionError(f"unexpected git call: {args}")

    return run, calls


class TestMapSslCertFileForGit:
    def test_maps_when_only_ssl_cert_file_is_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "corp-bundle.pem"))
        monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
        run, _ = _git_config_reader(None)
        monkeypatch.setattr(update_cmd, "_git_run", run)
        update_cmd._map_ssl_cert_file_for_git(["git"])
        import os
        assert os.environ["GIT_SSL_CAINFO"] == str(tmp_path / "corp-bundle.pem")

    def test_explicit_git_ssl_cainfo_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "corp-bundle.pem"))
        monkeypatch.setenv("GIT_SSL_CAINFO", str(tmp_path / "git-pinned.pem"))
        run, calls = _git_config_reader(None)
        monkeypatch.setattr(update_cmd, "_git_run", run)
        update_cmd._map_ssl_cert_file_for_git(["git"])
        import os
        assert os.environ["GIT_SSL_CAINFO"] == str(tmp_path / "git-pinned.pem")
        assert calls == []  # the env pin short-circuits before git is consulted

    def test_configured_http_sslcainfo_is_not_overridden(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "corp-bundle.pem"))
        monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
        run, _ = _git_config_reader(str(tmp_path / "configured.pem"))
        monkeypatch.setattr(update_cmd, "_git_run", run)
        update_cmd._map_ssl_cert_file_for_git(["git"])
        import os
        assert "GIT_SSL_CAINFO" not in os.environ

    def test_noop_without_ssl_cert_file(self, monkeypatch):
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
        run, calls = _git_config_reader(None)
        monkeypatch.setattr(update_cmd, "_git_run", run)
        update_cmd._map_ssl_cert_file_for_git(["git"])
        import os
        assert "GIT_SSL_CAINFO" not in os.environ
        assert calls == []  # nothing to map — git is never consulted

    def test_mapped_bundle_reaches_network_git_children(self, monkeypatch, tmp_path):
        """The mapping is live in the env of a network git call made after it."""
        import os
        monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "corp-bundle.pem"))
        monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)

        def run(cmd, args, **kwargs):
            if args[:3] == ["config", "--get", "http.sslCAInfo"]:
                return subprocess.CompletedProcess(cmd, 1, "", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(update_cmd, "_git_run", run)
        update_cmd._map_ssl_cert_file_for_git(["git"])
        kw = update_cmd._no_prompt_git_kwargs()
        assert kw["env"]["GIT_SSL_CAINFO"] == str(tmp_path / "corp-bundle.pem")
        assert os.environ["GIT_SSL_CAINFO"] == str(tmp_path / "corp-bundle.pem")
