import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backup-health-check.py"


def _module():
    spec = importlib.util.spec_from_file_location("backup_health_check", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_backup_emits_alert_without_cron_failure(tmp_path, capsys):
    module = _module()
    setattr(module, "BACKUP_ROOT", tmp_path / "missing-backups")

    result = module.main()

    assert result == 0
    assert capsys.readouterr().out == (
        f"ALERT: No backup archives found in {module.BACKUP_ROOT}\n"
    )
