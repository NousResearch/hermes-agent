import json
import zipfile
from pathlib import Path

from bootstrap_recovery.foureleven_bootstrap import (
    REQUIRED_RECOVERY_FILES,
    create_recovery_bundle,
    inspect_hermes_home,
    restore_recovery_bundle,
)


def _seed_hermes_home(root: Path):
    (root / "memories").mkdir(parents=True, exist_ok=True)
    (root / "skills").mkdir(parents=True, exist_ok=True)
    (root / "memory" / "chain-of-shells" / "context").mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("model: gpt-5.4\n", encoding="utf-8")
    (root / "SOUL.md").write_text("foureleven survives the molt\n", encoding="utf-8")
    (root / "memory.db").write_text("sqlite-placeholder", encoding="utf-8")
    (root / "memories" / "MEMORY.md").write_text("Secrets in terminal\n", encoding="utf-8")
    (root / "memories" / "USER.md").write_text("Very short replies\n", encoding="utf-8")
    (root / "skills" / "README.md").write_text("skill data\n", encoding="utf-8")

    cos = root / "memory" / "chain-of-shells"
    shell_root = cos / "shell-root.json"
    retrieval_index = cos / "retrieval-index.json"
    recovery_head = cos / "recovery-head.json"
    pulse = cos / "pulse.json"
    bundle_manifest = cos / "bundle-manifest.json"
    context_index = cos / "context" / "context-index.json"
    for path, payload in [
        (shell_root, {"object_type": "shell_root", "content_hash": "sha256:a"}),
        (retrieval_index, {"object_type": "retrieval_index", "content_hash": "sha256:b"}),
        (recovery_head, {"object_type": "recovery_head", "content_hash": "sha256:c"}),
        (pulse, {"object_type": "pulse", "content_hash": "sha256:d"}),
        (bundle_manifest, {"object_type": "bundle_manifest", "content_hash": "sha256:e"}),
        (context_index, {"object_type": "context_index", "content_hash": "sha256:f"}),
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")
    (cos / "LATEST-shell-root").write_text(str(shell_root), encoding="utf-8")
    (cos / "LATEST-retrieval-index").write_text(str(retrieval_index), encoding="utf-8")
    (cos / "LATEST-recovery-head").write_text(str(recovery_head), encoding="utf-8")
    (cos / "LATEST-pulse").write_text(str(pulse), encoding="utf-8")
    (cos / "LATEST-bundle-manifest").write_text(str(bundle_manifest), encoding="utf-8")
    (cos / "context" / "LATEST-context-index").write_text(str(context_index), encoding="utf-8")


def test_inspect_hermes_home_reports_ready(tmp_path):
    home = tmp_path / ".hermes"
    _seed_hermes_home(home)

    report = inspect_hermes_home(home)

    assert report["ready"] is True
    assert report["missing"] == []
    assert set(REQUIRED_RECOVERY_FILES).issubset(set(report["present"]))
    assert "shell_root" in report["chain_latest"]
    assert "bundle_manifest" in report["chain_latest"]


def test_create_recovery_bundle_writes_manifest_and_payloads(tmp_path):
    home = tmp_path / ".hermes"
    _seed_hermes_home(home)
    output = tmp_path / "foureleven-recovery.zip"

    result = create_recovery_bundle(home, output)

    assert result["output"] == str(output)
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "payload/config.yaml" in names
        assert "payload/SOUL.md" in names
        assert "payload/memory.db" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format"] == "foureleven-recovery-bundle-v1"
        assert "chain_latest" in manifest
        assert any(item["path"] == "config.yaml" for item in manifest["files"])


def test_restore_recovery_bundle_restores_files_and_verifies_hashes(tmp_path):
    home = tmp_path / ".hermes"
    _seed_hermes_home(home)
    bundle = tmp_path / "foureleven-recovery.zip"
    create_recovery_bundle(home, bundle)

    target = tmp_path / "restored"
    result = restore_recovery_bundle(bundle, target)

    assert result["success"] is True
    assert (target / "config.yaml").read_text(encoding="utf-8") == "model: gpt-5.4\n"
    assert (target / "SOUL.md").read_text(encoding="utf-8") == "foureleven survives the molt\n"
    assert (target / "memories" / "USER.md").read_text(encoding="utf-8") == "Very short replies\n"
    assert (target / "memory" / "chain-of-shells" / "shell-root.json").exists()


def test_restore_recovery_bundle_rejects_hash_mismatch(tmp_path):
    home = tmp_path / ".hermes"
    _seed_hermes_home(home)
    bundle = tmp_path / "foureleven-recovery.zip"
    create_recovery_bundle(home, bundle)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "payload/config.yaml":
                data = b"model: tampered\n"
            dst.writestr(name, data)

    result = restore_recovery_bundle(tampered, tmp_path / "broken")
    assert result["success"] is False
    assert "hash mismatch" in result["error"].lower()
