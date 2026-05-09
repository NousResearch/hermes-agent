"""Tests for hermes_constants.is_termux() — Termux/Android env detection."""

from hermes_constants import is_termux


class TestIsTermux:
    def test_returns_false_on_clean_env(self, monkeypatch):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/usr")

        assert is_termux() is False

    def test_returns_true_when_termux_version_set(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
        monkeypatch.setenv("PREFIX", "/usr")

        assert is_termux() is True

    def test_returns_true_for_termux_prefix_path(self, monkeypatch):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")

        assert is_termux() is True

    def test_returns_false_for_unrelated_prefix(self, monkeypatch):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/opt/homebrew")

        assert is_termux() is False

    def test_returns_false_when_prefix_unset(self, monkeypatch):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.delenv("PREFIX", raising=False)

        assert is_termux() is False

    def test_termux_version_takes_precedence_over_unrelated_prefix(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
        monkeypatch.setenv("PREFIX", "/usr/local")

        assert is_termux() is True
