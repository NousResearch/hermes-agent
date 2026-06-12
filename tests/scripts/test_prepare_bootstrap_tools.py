"""Tests for the bootstrap tool archive preparation helper."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_bootstrap_tools.py"
    spec = importlib.util.spec_from_file_location("_prepare_bootstrap_tools", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareBootstrapToolsTests(unittest.TestCase):
    """Validate archive naming logic used by the release preparation helper."""

    def test_select_latest_node_archive_filters_major_and_arch(self):
        module = _load_script_module()
        html = """
            <a href="node-v22.18.0-win-x64.zip">node-v22.18.0-win-x64.zip</a>
            <a href="node-v22.19.1-win-arm64.zip">node-v22.19.1-win-arm64.zip</a>
            <a href="node-v22.19.0-win-x64.zip">node-v22.19.0-win-x64.zip</a>
            <a href="node-v21.99.0-win-x64.zip">node-v21.99.0-win-x64.zip</a>
        """

        self.assertEqual(
            module.select_latest_node_archive(html, "x64"),
            "node-v22.19.0-win-x64.zip",
        )

    def test_select_latest_unix_node_archive_prefers_xz(self):
        module = _load_script_module()
        html = """
            <a href="node-v22.18.0-linux-x64.tar.gz">node-v22.18.0-linux-x64.tar.gz</a>
            <a href="node-v22.19.1-linux-arm64.tar.xz">node-v22.19.1-linux-arm64.tar.xz</a>
            <a href="node-v22.19.0-linux-x64.tar.gz">node-v22.19.0-linux-x64.tar.gz</a>
            <a href="node-v22.19.1-linux-x64.tar.xz">node-v22.19.1-linux-x64.tar.xz</a>
            <a href="node-v21.99.0-linux-x64.tar.xz">node-v21.99.0-linux-x64.tar.xz</a>
        """

        self.assertEqual(
            module.select_latest_unix_node_archive(html, "linux", "x64"),
            "node-v22.19.1-linux-x64.tar.xz",
        )

    def test_archive_specs_match_installer_runtime_assets(self):
        module = _load_script_module()

        x64_specs = module.archive_specs_for_arch("x64", "node-v22.19.0-win-x64.zip")
        x64_names = [spec.name for spec in x64_specs]

        self.assertEqual(
            x64_names,
            [
                "node-v22.19.0-win-x64.zip",
                "uv-x86_64-pc-windows-msvc.zip",
                "ripgrep-15.1.0-x86_64-pc-windows-msvc.zip",
                "PortableGit-2.54.0-64-bit.7z.exe",
            ],
        )
        self.assertTrue(x64_specs[0].url.endswith("/latest-v22.x/node-v22.19.0-win-x64.zip"))
        self.assertTrue(
            x64_specs[2].url.endswith("/15.1.0/ripgrep-15.1.0-x86_64-pc-windows-msvc.zip")
        )
        self.assertTrue(
            x64_specs[3].url.endswith("/v2.54.0.windows.1/PortableGit-2.54.0-64-bit.7z.exe")
        )

        arm64_specs = module.archive_specs_for_arch("arm64", "node-v22.19.0-win-arm64.zip")
        self.assertEqual(
            [spec.name for spec in arm64_specs],
            [
                "node-v22.19.0-win-arm64.zip",
                "uv-aarch64-pc-windows-msvc.zip",
                "ripgrep-15.1.0-aarch64-pc-windows-msvc.zip",
                "PortableGit-2.54.0-arm64.7z.exe",
            ],
        )

    def test_archive_specs_reject_unknown_architecture(self):
        module = _load_script_module()

        with self.assertRaisesRegex(ValueError, "unsupported Windows architecture"):
            module.archive_specs_for_arch("mips", "node-v22.19.0-win-mips.zip")

    def test_unix_archive_specs_match_uv_runtime_assets(self):
        module = _load_script_module()

        linux_specs = module.archive_specs_for_target(
            "linux",
            "x64",
            "node-v22.19.1-linux-x64.tar.xz",
        )
        self.assertEqual(
            [spec.name for spec in linux_specs],
            [
                "node-v22.19.1-linux-x64.tar.xz",
                "uv-x86_64-unknown-linux-gnu.tar.gz",
                "ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz",
            ],
        )
        self.assertTrue(
            linux_specs[1].url.endswith("/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz")
        )
        self.assertTrue(
            linux_specs[2].url.endswith("/15.1.0/ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz")
        )

        mac_specs = module.archive_specs_for_target(
            "macos",
            "arm64",
            "node-v22.19.1-darwin-arm64.tar.xz",
        )
        self.assertEqual(
            [spec.name for spec in mac_specs],
            [
                "node-v22.19.1-darwin-arm64.tar.xz",
                "uv-aarch64-apple-darwin.tar.gz",
                "ripgrep-15.1.0-aarch64-apple-darwin.tar.gz",
            ],
        )

        with self.assertRaisesRegex(ValueError, "unsupported Unix uv platform"):
            module.archive_specs_for_target("linux", "x86", "node-v22.19.1-linux-x86.tar.xz")

    def test_manifest_records_archive_size_and_sha256(self):
        module = _load_script_module()
        root = Path("tmp-bootstrap-tools-test")
        output_dir = root / "bootstrap-tools"
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "uv-x86_64-pc-windows-msvc.zip"
        archive.write_bytes(b"uv archive")

        try:
            spec = module.ArchiveSpec(
                name="uv-x86_64-pc-windows-msvc.zip",
                url="https://example.invalid/uv.zip",
            )
            prepared = module.prepared_archive_record("x64", spec, archive)
            manifest_path = module.write_manifest(output_dir, [prepared])
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["schemaVersion"], 1)
            self.assertEqual(payload["archives"][0]["arch"], "x64")
            self.assertEqual(payload["archives"][0]["name"], "uv-x86_64-pc-windows-msvc.zip")
            self.assertEqual(payload["archives"][0]["sizeBytes"], len(b"uv archive"))
            self.assertEqual(
                payload["archives"][0]["sha256"],
                "ba8cad66b72bd2f5aabb165b4b2c0a935637a8f629025bbfa1caf739f6706ed5",
            )
        finally:
            if archive.exists():
                archive.unlink()
            manifest = output_dir / "bootstrap-tools-manifest.json"
            if manifest.exists():
                manifest.unlink()
            output_dir.rmdir()
            root.rmdir()

    def test_validate_manifest_rejects_archive_checksum_mismatch(self):
        module = _load_script_module()
        root = Path("tmp-bootstrap-tools-validate-test")
        output_dir = root / "bootstrap-tools"
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "uv-x86_64-pc-windows-msvc.zip"
        archive.write_bytes(b"uv archive")

        try:
            spec = module.ArchiveSpec(
                name="uv-x86_64-pc-windows-msvc.zip",
                url="https://example.invalid/uv.zip",
            )
            prepared = module.prepared_archive_record("x64", spec, archive)
            module.write_manifest(output_dir, [prepared])

            self.assertEqual(module.validate_manifest(output_dir), 1)

            archive.write_bytes(b"badarchive")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                module.validate_manifest(output_dir)
        finally:
            if archive.exists():
                archive.unlink()
            manifest = output_dir / "bootstrap-tools-manifest.json"
            if manifest.exists():
                manifest.unlink()
            output_dir.rmdir()
            root.rmdir()

    def test_validate_manifest_rejects_archive_without_url(self):
        module = _load_script_module()
        root = Path("tmp-bootstrap-tools-url-test")
        output_dir = root / "bootstrap-tools"
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "uv-x86_64-pc-windows-msvc.zip"
        archive.write_bytes(b"uv archive")
        manifest = output_dir / "bootstrap-tools-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "archives": [
                        {
                            "arch": "x64",
                            "name": "uv-x86_64-pc-windows-msvc.zip",
                            "sizeBytes": len(b"uv archive"),
                            "sha256": module.sha256_file(archive),
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            with self.assertRaisesRegex(RuntimeError, "missing url"):
                module.validate_manifest(output_dir)
        finally:
            if archive.exists():
                archive.unlink()
            if manifest.exists():
                manifest.unlink()
            output_dir.rmdir()
            root.rmdir()

    def test_validate_manifest_rejects_archive_with_insecure_url(self):
        module = _load_script_module()
        root = Path("tmp-bootstrap-tools-url-scheme-test")
        output_dir = root / "bootstrap-tools"
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "uv-x86_64-pc-windows-msvc.zip"
        archive.write_bytes(b"uv archive")
        manifest = output_dir / "bootstrap-tools-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "archives": [
                        {
                            "arch": "x64",
                            "name": "uv-x86_64-pc-windows-msvc.zip",
                            "url": "http://example.invalid/uv.zip",
                            "sizeBytes": len(b"uv archive"),
                            "sha256": module.sha256_file(archive),
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            with self.assertRaisesRegex(RuntimeError, "invalid url"):
                module.validate_manifest(output_dir)
        finally:
            if archive.exists():
                archive.unlink()
            if manifest.exists():
                manifest.unlink()
            output_dir.rmdir()
            root.rmdir()

    def test_validate_manifest_rejects_archive_without_arch(self):
        module = _load_script_module()
        root = Path("tmp-bootstrap-tools-arch-test")
        output_dir = root / "bootstrap-tools"
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "uv-x86_64-pc-windows-msvc.zip"
        archive.write_bytes(b"uv archive")
        manifest = output_dir / "bootstrap-tools-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "archives": [
                        {
                            "name": "uv-x86_64-pc-windows-msvc.zip",
                            "url": "https://example.invalid/uv.zip",
                            "sizeBytes": len(b"uv archive"),
                            "sha256": module.sha256_file(archive),
                        }
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            with self.assertRaisesRegex(RuntimeError, "missing arch"):
                module.validate_manifest(output_dir)
        finally:
            if archive.exists():
                archive.unlink()
            if manifest.exists():
                manifest.unlink()
            output_dir.rmdir()
            root.rmdir()

    def test_installer_workflows_upload_bootstrap_tools_manifest(self):
        repo_root = Path(__file__).resolve().parents[2]
        windows_workflow_path = repo_root / ".github" / "workflows" / "build-windows-installer.yml"
        windows_workflow = windows_workflow_path.read_text(encoding="utf-8")
        unix_workflow = (
            repo_root / ".github" / "workflows" / "build-unix-installers.yml"
        ).read_text(encoding="utf-8")
        manifest_path = "apps/bootstrap-installer/src-tauri/bootstrap-tools/bootstrap-tools-manifest.json"

        self.assertIn(manifest_path, windows_workflow)
        self.assertIn(manifest_path, unix_workflow)
        self.assertIn("python scripts/prepare_bootstrap_tools.py --validate-only", windows_workflow)
        self.assertIn("python scripts/prepare_bootstrap_tools.py --validate-only", unix_workflow)
        upload_sections = [
            section
            for section in windows_workflow.split("\n      - name: ")
            if "actions/upload-artifact@" in section
        ]
        self.assertGreaterEqual(len(upload_sections), 3)
        for section in upload_sections:
            self.assertIn("if-no-files-found: error", section)


if __name__ == "__main__":
    unittest.main()
