"""Compose #99182's identity guard with generated-cache and routed-policy trust."""
import os
import time

from gateway import media_policy
from gateway.platforms import base
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_generated_cache_and_routed_operator_trust_do_not_grant_secret_identity(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / ".hermes"
    root.mkdir(parents=True)
    exports = home / "exports"
    exports.mkdir()
    exported = exports / "report.txt"
    exported.write_bytes(b"ordinary exported artifact\n")
    old = time.time() - 25 * 3600
    os.utime(exported, (old, old))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(base, "_HERMES_HOME", root)
    monkeypatch.setattr(base, "_HERMES_ROOT", root)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", ())
    # Isolate scratch's enclosing /root collision, not credential denials.
    monkeypatch.setattr(base, "_MEDIA_DELIVERY_DENIED_PREFIXES", tuple(
        p for p in base._MEDIA_DELIVERY_DENIED_PREFIXES if p != "/root"))
    envs = ("HERMES_MEDIA_DELIVERY_STRICT", "HERMES_MEDIA_TRUST_RECENT_FILES", "HERMES_MEDIA_ALLOW_DIRS")
    for env in envs:
        monkeypatch.delenv(env, raising=False)
    profiles = [root / "profiles" / name for name in ("alpha", "beta")]
    for profile in profiles:
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(
            "gateway:\n  strict: true\n  trust_recent_files: false\n"
            f"  media_delivery_allow_dirs:\n    - {exports}\n")
        generated = profile / "cache" / "generated" / "images"
        generated.mkdir(parents=True)
        ordinary = generated / "report.png"
        ordinary.write_bytes(b"ordinary generated artifact\n")
        old = time.time() - 25 * 3600
        os.utime(ordinary, (old, old))
    for profile in (profiles[0], profiles[1], profiles[0]):
        token = set_hermes_home_override(profile)
        try:
            media_policy.apply_media_policy_env()
            assert all(env not in os.environ for env in envs)
            assert media_policy.media_delivery_strict() is True
            assert media_policy.media_delivery_trust_recent() is False
            assert media_policy.media_delivery_allow_dirs() == str(exports)
            generated = profile / "cache" / "generated" / "images"
            ordinary = generated / "report.png"
            assert base.validate_media_delivery_path(str(ordinary)) == str(ordinary.resolve())
            assert base.validate_media_delivery_path(str(exported)) == str(exported.resolve())
            for name in (".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials"):
                secret = home / name
                secret.write_bytes(b"placeholder credential bytes\n")
                alias = generated / (name + ".png")
                exported_alias = exports / (name + ".txt")
                os.link(secret, alias)
                os.link(secret, exported_alias)
                try:
                    assert base.BasePlatformAdapter.filter_local_delivery_paths([
                        str(secret), str(alias), str(exported_alias), str(ordinary), str(exported),
                    ]) == [str(ordinary.resolve()), str(exported.resolve())]
                finally:
                    alias.unlink()
                    exported_alias.unlink()
        finally:
            reset_hermes_home_override(token)
    assert all(env not in os.environ for env in envs)
