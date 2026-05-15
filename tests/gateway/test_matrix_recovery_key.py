import logging
import stat
import sys
import types

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

from gateway.platforms import matrix


def test_record_generated_recovery_key_does_not_log_secret(tmp_path, monkeypatch, caplog):
    secret = "EsTc LWJ6 V6LE TWXQ 9EDH J7XF D52C PQSF K9MA S8BK Q3N2"
    recovery_path = tmp_path / "matrix" / "recovery_key.txt"

    def fake_save(recovery_key):
        recovery_path.parent.mkdir(parents=True)
        recovery_path.write_text(recovery_key + "\n")
        return recovery_path

    monkeypatch.setattr(matrix, "_save_generated_recovery_key", fake_save)

    with caplog.at_level(logging.WARNING, logger=matrix.logger.name):
        saved_path = matrix._record_generated_recovery_key("@bot:example.org", secret)

    assert saved_path == recovery_path
    assert secret not in caplog.text
    assert str(recovery_path) in caplog.text
    assert "MATRIX_RECOVERY_KEY" in caplog.text


def test_save_generated_recovery_key_uses_owner_only_permissions(tmp_path):
    secret = "EsTc LWJ6 V6LE TWXQ 9EDH J7XF D52C PQSF K9MA S8BK Q3N2"
    recovery_path = tmp_path / "store" / "recovery_key.txt"

    saved_path = matrix._save_generated_recovery_key(secret, recovery_path)

    assert saved_path == recovery_path
    assert recovery_path.read_text() == secret + "\n"
    assert stat.S_IMODE(recovery_path.stat().st_mode) == 0o600
