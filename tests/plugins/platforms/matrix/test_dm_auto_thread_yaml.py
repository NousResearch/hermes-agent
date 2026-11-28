"""``dm_auto_thread`` obeys the per-profile YAML bridge like its siblings ``auto_thread``
and ``dm_mention_threads``.

Commit 545e74d0ea bridged "matrix flags" extra-first but skipped this one line, so
``gateway.platforms.matrix.dm_auto_thread: true`` was silently ignored (env-only read).
"""

from gateway.config import PlatformConfig


def _adapter_with(extra, monkeypatch, env=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    monkeypatch.delenv("MATRIX_DM_AUTO_THREAD", raising=False)
    if env is not None:
        monkeypatch.setenv("MATRIX_DM_AUTO_THREAD", env)
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
            **extra,
        },
    )
    return MatrixAdapter(config)


def test_yaml_dm_auto_thread_enables_dm_threading(monkeypatch):
    """The YAML key alone turns DM auto-threading on (red on base: YAML was ignored)."""
    assert _adapter_with({"dm_auto_thread": True}, monkeypatch)._dm_auto_thread is True


def test_scoped_env_beats_yaml_and_default_stays_off(monkeypatch):
    """Precedence matches the siblings: explicit env → YAML → default False."""
    assert _adapter_with({"dm_auto_thread": True}, monkeypatch, env="false")._dm_auto_thread is False
    assert _adapter_with({"dm_auto_thread": False}, monkeypatch, env="true")._dm_auto_thread is True
    assert _adapter_with({}, monkeypatch)._dm_auto_thread is False
