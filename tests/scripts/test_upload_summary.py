"""Every verified R2 upload lands in the GitHub step summary, with its public URL."""
import json

import pytest

from scripts.releases import r2, upload_summary
from scripts.releases.r2_scope import R2Scope


BASE = "https://hermes-assets.nousresearch.com"
KEY = "releases/commit/" + "a" * 40 + "/HermesBundled-1.2.3-win-x64.msix"


@pytest.fixture(autouse=True)
def clear_notes():
    upload_summary._KEYS.clear()
    upload_summary._REGISTERED = False
    yield
    upload_summary._KEYS.clear()


def test_render_uses_the_public_asset_host(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_R2_PUBLIC_URL", raising=False)
    monkeypatch.delenv("R2_DISPOSABLE_RUN", raising=False)
    text = upload_summary.render([KEY])
    assert f"[download]({BASE}/{KEY})" in text
    assert text.startswith("\n### R2 uploads\n")
    assert upload_summary.render([]) == ""


def test_render_names_a_key_whose_url_cannot_be_built(monkeypatch):
    monkeypatch.setattr(r2, "public_url_for",
                        lambda base, key: (_ for _ in ()).throw(ValueError("bad key")))
    text = upload_summary.render(["../escape"])
    assert "`../escape`" in text
    assert "| — |" in text


def test_note_dedupes_and_flush_appends_once(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    summary.write_text("existing\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("CLOUDFLARE_R2_PUBLIC_URL", raising=False)
    monkeypatch.delenv("R2_DISPOSABLE_RUN", raising=False)
    upload_summary.note(KEY)
    upload_summary.note(KEY)
    upload_summary.flush()
    upload_summary.flush()
    text = summary.read_text(encoding="utf-8")
    assert text.count("### R2 uploads") == 1
    assert text.startswith("existing\n")
    assert f"{BASE}/{KEY}" in text


def test_flush_without_a_summary_keeps_the_notes(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    upload_summary.note(KEY)
    upload_summary.flush()
    assert upload_summary._KEYS == [KEY]


def test_put_object_records_only_after_the_size_check(monkeypatch):
    recorded = []
    monkeypatch.setattr(upload_summary, "note", recorded.append)

    class Response:
        status = 200
        def header(self, name):
            return "11" if name == "content-length" else None
        def text(self):
            return ""

    monkeypatch.setattr(r2, "signed_request", lambda *args, **kwargs: Response())
    monkeypatch.setattr(r2, "R2Scope", type("Scope", (), {"configured": staticmethod(lambda: R2Scope())}))
    with pytest.raises(r2.R2RequestError):
        r2.put_object({"access_key_id": "k", "secret_key": "s"}, "https://example.r2.cloudflarestorage.com",
                       "bucket", KEY, b"payload-bytes", "20000101T000000Z", "application/octet-stream", fetcher=None)
    assert recorded == []
