"""Read-only dashboard provenance contracts for doctor."""

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import doctor, doctor_web


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()


def _commit_fixture(repo: Path, content: str) -> str:
    (repo / "tracked.txt").write_text(content, encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.name=Hermes Test",
        "-c",
        "user.email=hermes-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        content.strip(),
    )
    return _git(repo, "rev-parse", "HEAD")


def _write_web_provenance(
    dist: Path,
    *,
    commit_sha: str,
    dirty: bool,
) -> None:
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<main>Hermes</main>\n", encoding="utf-8")
    (dist / "build-provenance.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "commitSha": commit_sha,
                "branch": "fixture-branch",
                "dirty": dirty,
                "builtAt": "2030-01-02T03:04:05.000Z",
                "invocation": "npm run build",
            }
        ),
        encoding="utf-8",
    )


class TestDoctorWebBundleProvenance:
    def _capture_report(self, monkeypatch):
        reported = {"ok": [], "warn": [], "info": [], "section": []}
        monkeypatch.setattr(
            doctor_web,
            "check_ok",
            lambda text, detail="": reported["ok"].append(f"{text} {detail}"),
        )
        monkeypatch.setattr(
            doctor_web,
            "check_warn",
            lambda text, detail="": reported["warn"].append(f"{text} {detail}"),
        )
        monkeypatch.setattr(doctor_web, "check_info", lambda text: reported["info"].append(text))
        monkeypatch.setattr(doctor_web, "_section", lambda text: reported["section"].append(text))
        finding = doctor_web._check_web_bundle_provenance(True)
        assert not finding.issues and not finding.manual_issues and finding.fixed == 0
        return reported

    def _repo_and_dist(self, monkeypatch, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "--quiet")
        dist = repo / "hermes_cli" / "web_dist"
        monkeypatch.setattr(doctor, "PROJECT_ROOT", repo)
        monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
        monkeypatch.delenv("HERMES_REVISION", raising=False)
        return repo, dist

    def test_reports_bundle_matching_current_head(self, monkeypatch, tmp_path):
        repo, dist = self._repo_and_dist(monkeypatch, tmp_path)
        current = _commit_fixture(repo, "first\n")
        _write_web_provenance(dist, commit_sha=current, dirty=False)

        reported = self._capture_report(monkeypatch)

        assert reported["section"] == ["Web Dashboard Bundle"]
        assert any("matches current source HEAD" in line for line in reported["ok"])
        assert not any("differs from current source HEAD" in line for line in reported["warn"])
        assert any("Build invocation: npm run build" in line for line in reported["info"])

    def test_reports_dirty_bundle_and_revision_drift(self, monkeypatch, tmp_path):
        repo, dist = self._repo_and_dist(monkeypatch, tmp_path)
        deployed = _commit_fixture(repo, "first\n")
        current = _commit_fixture(repo, "second\n")
        _write_web_provenance(dist, commit_sha=deployed, dirty=True)

        reported = self._capture_report(monkeypatch)
        warnings = "\n".join(reported["warn"])

        assert "built from a dirty tree" in warnings
        assert "differs from current source HEAD" in warnings
        assert deployed[:12] in warnings
        assert current[:12] in warnings

    def test_missing_record_is_advisory_and_does_not_repair(self, monkeypatch, tmp_path):
        _repo, dist = self._repo_and_dist(monkeypatch, tmp_path)
        dist.mkdir(parents=True)
        (dist / "index.html").write_text("<main>Legacy bundle</main>\n", encoding="utf-8")

        reported = self._capture_report(monkeypatch)

        assert any("provenance unavailable" in line for line in reported["warn"])
        assert not (dist / "build-provenance.json").exists()


@pytest.mark.parametrize("record", [None, "not json", json.dumps({
    "schemaVersion": 1, "commitSha": None, "branch": None, "dirty": None,
    "builtAt": "2030-01-02T03:04:05.000Z", "invocation": "vite build",
})])
def test_override_is_advisory_and_read_only_even_with_fix(tmp_path, monkeypatch, record):
    dist = tmp_path / "custom bundle"
    dist.mkdir()
    (dist / "index.html").write_text("<main>Custom</main>", encoding="utf-8")
    if record is not None:
        (dist / "build-provenance.json").write_text(record, encoding="utf-8")
    monkeypatch.setenv("HERMES_WEB_DIST", str(dist))
    monkeypatch.setattr(doctor_web, "_current_source_head", lambda: None)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode)
              for p in dist.iterdir()}

    assert doctor_web._active_web_dist() == dist
    finding = doctor_web._check_web_bundle_provenance(True)

    assert not finding.issues and not finding.manual_issues and finding.fixed == 0
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode)
            for p in dist.iterdir()} == before
