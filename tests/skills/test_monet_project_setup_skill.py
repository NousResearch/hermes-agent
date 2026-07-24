"""Tests for optional-skills/software-development/monet-project-setup scripts."""

import base64
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "monet-project-setup"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import project_primer_runtime as runtime
import verify_package

TEMPLATE = SCRIPTS_DIR.parent / "templates" / "example-primer.json"

# Synthetic credential-shaped strings for injection tests only.
FAKE_GITHUB_TOKEN = "ghp_" + "Ab1" * 8
FAKE_BEARER_VALUE = "Bearer " + "A" * 24
FAKE_JWT = "eyJ" + "a" * 12 + ".eyJ" + "b" * 12 + "." + "c" * 16
FAKE_ENCRYPTED_PEM = "-----BEGIN ENCRYPTED PRIVATE KEY-----"
FAKE_PGP_BLOCK = "-----BEGIN PGP PRIVATE KEY BLOCK-----"

PREVIEW_VERSION_MEMBER = "versions/hermes-preview/version.json"
PREVIEW_PAGE_MEMBER = "versions/hermes-preview/pages/home.png"


def _build_package(tmp_path: Path) -> Path:
    primer = runtime.load_project_primer(TEMPLATE)
    output = tmp_path / f"{primer.project.slug}.monetproj"
    return runtime.write_project_primer_package(primer, output)


def _rewritten(package: Path, tmp_path: Path, name: str, members: dict) -> Path:
    """Rewrite or add members after the build, like an attacker would."""
    out = tmp_path / f"{name}.monetproj"
    with zipfile.ZipFile(package) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            if info.filename in members:
                continue
            dst.writestr(info, src.read(info.filename))
        for member, payload in members.items():
            data = payload if isinstance(payload, bytes) else json.dumps(payload)
            dst.writestr(member, data)
    return out


def _member_json(package: Path, member: str) -> dict:
    with zipfile.ZipFile(package) as archive:
        return json.loads(archive.read(member))


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    )


def _minimal_png(extra_chunks: bytes = b"", trailer: bytes = b"") -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + extra_chunks
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
        + trailer
    )


def _preview_version(meta_description: str | None = None) -> dict:
    return {
        "label": "Hermes Preview",
        "capturedAt": "2026-07-18T12:00:00+00:00",
        "notes": "Rendered by Hermes for immediate review.",
        "viewport": "desktop",
        "colorScheme": "light",
        "capturedWith": "chromium",
        "captureKind": "agent-preview",
        "parentVersionLabel": None,
        "pages": [
            {
                "url": "https://example.com/",
                "pageSlug": "home",
                "reviewOrder": 0,
                "pageTitle": "Home",
                "metaDescription": meta_description,
                "canonicalURL": "https://example.com/",
            }
        ],
    }


def _preview_package(
    tmp_path: Path,
    name: str = "preview",
    version: dict | None = None,
    png: bytes | None = None,
) -> Path:
    base = _build_package(tmp_path)
    manifest = _member_json(base, "manifest.json")
    manifest["versionCount"] = 1
    return _rewritten(
        base,
        tmp_path,
        name,
        {
            "manifest.json": manifest,
            PREVIEW_VERSION_MEMBER: version or _preview_version(),
            PREVIEW_PAGE_MEMBER: png if png is not None else _minimal_png(),
        },
    )


class TestBuildAndVerify:
    def test_round_trip_verifies_secret_free(self, tmp_path):
        package = _build_package(tmp_path)
        result = verify_package.verify(package)
        assert result["valid"] is True
        assert result["containsSecrets"] is False
        assert result["projectSlug"] == "example-site"
        assert result["previewPages"] == 0
        assert sorted(result["members"]) == ["manifest.json", "primer.json", "project.json"]

    def test_preview_round_trip_verifies(self, tmp_path):
        package = _preview_package(tmp_path)
        result = verify_package.verify(package)
        assert result["valid"] is True
        assert result["previewVersion"] == "Hermes Preview"
        assert result["previewPages"] == 1

    def test_builder_rejects_primer_with_credential_value(self, tmp_path):
        primer_data = json.loads(TEMPLATE.read_text())
        primer_data["project"]["description"] = f"Deploy with {FAKE_GITHUB_TOKEN}"
        with pytest.raises(ValueError, match="credential"):
            runtime.load_project_primer(primer_data)


class TestSecretScan:
    """Unit coverage for the shared detector the verifier relies on."""

    def test_camel_case_credential_key_rejected(self):
        with pytest.raises(ValueError, match="credential field"):
            runtime.reject_secret_material({"clientSecret": "opaque-value"}, label="t")

    def test_null_valued_credential_key_carries_nothing(self):
        runtime.reject_secret_material({"secretRef": None}, label="t")

    def test_slot_metadata_is_gated_by_flag(self):
        slots = {"secret_slots": [{"id": "github-access", "label": "GitHub access"}]}
        with pytest.raises(ValueError, match="credential field"):
            runtime.reject_secret_material(slots, label="t")
        runtime.reject_secret_material(slots, label="t", allow_slot_metadata=True)

    @pytest.mark.parametrize(
        "value", [FAKE_ENCRYPTED_PEM, FAKE_PGP_BLOCK, FAKE_JWT, FAKE_GITHUB_TOKEN]
    )
    def test_credential_shaped_values_rejected(self, value):
        with pytest.raises(ValueError, match="credential-like value"):
            runtime.reject_secret_material({"note": value}, label="t")


class TestTamperedPackages:
    """A package modified after build must never verify as secret-free."""

    def test_manifest_credential_value_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        manifest = _member_json(package, "manifest.json")
        manifest["buildNote"] = FAKE_GITHUB_TOKEN
        bad = _rewritten(package, tmp_path, "m-value", {"manifest.json": manifest})
        with pytest.raises(ValueError, match="credential-like value.*manifest.json"):
            verify_package.verify(bad)

    def test_manifest_encrypted_pem_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        manifest = _member_json(package, "manifest.json")
        manifest["buildNote"] = FAKE_ENCRYPTED_PEM
        bad = _rewritten(package, tmp_path, "m-pem", {"manifest.json": manifest})
        with pytest.raises(ValueError, match="credential-like value"):
            verify_package.verify(bad)

    def test_project_credential_field_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        project = _member_json(package, "project.json")
        project["github_token"] = "not-a-real-value"
        bad = _rewritten(package, tmp_path, "p-key", {"project.json": project})
        with pytest.raises(ValueError, match="credential field.*project.json"):
            verify_package.verify(bad)

    def test_project_camel_case_key_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        project = _member_json(package, "project.json")
        project["clientSecret"] = "opaque0123456789"
        bad = _rewritten(package, tmp_path, "p-camel", {"project.json": project})
        with pytest.raises(ValueError, match="credential field"):
            verify_package.verify(bad)

    def test_project_jwt_value_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        project = _member_json(package, "project.json")
        project["reviewNote"] = FAKE_JWT
        bad = _rewritten(package, tmp_path, "p-jwt", {"project.json": project})
        with pytest.raises(ValueError, match="credential-like value"):
            verify_package.verify(bad)

    def test_project_slot_exception_not_honored_at_verify(self, tmp_path):
        package = _build_package(tmp_path)
        project = _member_json(package, "project.json")
        project["secret_slots"] = ["prod-db-password-hunter2-live"]
        bad = _rewritten(package, tmp_path, "p-slot", {"project.json": project})
        with pytest.raises(ValueError, match="credential field"):
            verify_package.verify(bad)

    def test_project_non_null_secret_ref_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        project = _member_json(package, "project.json")
        project["connectors"][0]["secretRef"] = "not-a-real-value"
        bad = _rewritten(package, tmp_path, "p-ref", {"project.json": project})
        with pytest.raises(ValueError, match="credential field"):
            verify_package.verify(bad)

    def test_project_base64_config_smuggle_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        project = _member_json(package, "project.json")
        smuggled = base64.b64encode(
            json.dumps({"token": "not-a-real-value"}).encode()
        ).decode()
        project["connectors"][0]["configJSON"] = smuggled
        bad = _rewritten(package, tmp_path, "p-b64", {"project.json": project})
        with pytest.raises(ValueError, match="credential field.*configJSON"):
            verify_package.verify(bad)

    def test_preview_version_credential_value_rejected(self, tmp_path):
        bad = _preview_package(
            tmp_path, "v-cred", version=_preview_version(meta_description=FAKE_BEARER_VALUE)
        )
        with pytest.raises(ValueError, match="credential-like value"):
            verify_package.verify(bad)

    def test_png_trailing_bytes_rejected(self, tmp_path):
        bad = _preview_package(
            tmp_path, "png-trailer", png=_minimal_png(trailer=FAKE_PGP_BLOCK.encode())
        )
        with pytest.raises(ValueError, match="trailing or malformed"):
            verify_package.verify(bad)

    def test_png_text_chunk_rejected(self, tmp_path):
        text_chunk = _png_chunk(b"tEXt", b"comment\x00hidden payload")
        bad = _preview_package(
            tmp_path, "png-text", png=_minimal_png(extra_chunks=text_chunk)
        )
        with pytest.raises(ValueError, match="embedded text metadata"):
            verify_package.verify(bad)

    def test_archive_comment_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        commented = tmp_path / "commented.monetproj"
        commented.write_bytes(package.read_bytes())
        with zipfile.ZipFile(commented, "a") as archive:
            archive.comment = FAKE_GITHUB_TOKEN.encode()
        with pytest.raises(ValueError, match="archive comment"):
            verify_package.verify(commented)

    def test_member_comment_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        commented = tmp_path / "member-comment.monetproj"
        with zipfile.ZipFile(package) as src, zipfile.ZipFile(commented, "w") as dst:
            for info in src.infolist():
                info.comment = FAKE_GITHUB_TOKEN.encode()
                dst.writestr(info, src.read(info.filename))
        with pytest.raises(ValueError, match="unsafe member"):
            verify_package.verify(commented)

    def test_unexpected_member_rejected(self, tmp_path):
        package = _build_package(tmp_path)
        bad = _rewritten(
            package, tmp_path, "extra", {"extra.json": {"note": "surplus member"}}
        )
        with pytest.raises(ValueError, match="exactly three expected members"):
            verify_package.verify(bad)


class TestSkillMetadata:
    def test_description_is_one_short_sentence(self):
        import re

        skill_md = (SCRIPTS_DIR.parent / "SKILL.md").read_text()
        match = re.search(r"^description: (.*)$", skill_md, re.MULTILINE)
        assert match is not None
        description = match.group(1).strip()
        assert len(description) <= 60, len(description)
        assert description.endswith(".")
        assert description.count(".") == 1
