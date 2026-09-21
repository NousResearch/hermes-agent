"""Security and provenance contracts for user-supplied image references."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from image_jobs import ReferenceRequest
from image_reference_staging import (
    FetchedReference,
    RawFetchResponse,
    ReferenceStager,
    ReferenceStagingError,
    SafeReferenceFetcher,
)


PUBLIC_IP = "93.184.216.34"


def _public_resolver(hostname: str) -> tuple[str, ...]:
    assert hostname in {"example.com", "images.example.com", "redirect.example.com"}
    return (PUBLIC_IP,)



def test_safe_fetcher_revalidates_each_redirect_before_transport_call() -> None:
    seen: list[str] = []

    def _raw_get(url: str, max_bytes: int) -> RawFetchResponse:
        seen.append(url)
        return RawFetchResponse(
            status=302,
            url=url,
            headers={"Location": "http://127.0.0.1:8188/system_stats"},
            body=b"",
        )

    fetcher = SafeReferenceFetcher(
        resolver=_public_resolver,
        raw_get=_raw_get,
        max_bytes=64,
    )

    with pytest.raises(ReferenceStagingError, match="public"):
        fetcher("https://example.com/start")

    assert seen == ["https://example.com/start"]


def test_safe_fetcher_rejects_oversized_transport_response() -> None:
    fetcher = SafeReferenceFetcher(
        resolver=_public_resolver,
        raw_get=lambda url, max_bytes: RawFetchResponse(
            status=200,
            url=url,
            headers={"Content-Type": "image/png"},
            body=b"x" * 65,
        ),
        max_bytes=64,
    )

    with pytest.raises(ReferenceStagingError, match="byte limit"):
        fetcher("https://example.com/source")


def test_stager_writes_hashed_image_inside_one_new_job_directory(tmp_path: Path) -> None:
    body = b"\x89PNG\r\n\x1a\nreference-bytes"
    stager = ReferenceStager(
        tmp_path / "staging",
        resolver=_public_resolver,
        fetcher=lambda url: FetchedReference(
            final_url=url,
            content_type="image/png; charset=binary",
            body=body,
            redirect_count=0,
        ),
    )

    staged = stager.stage((ReferenceRequest("https://images.example.com/ref.png"),), job_id="job-001")

    assert len(staged) == 1
    reference = staged[0]
    assert reference.sha256 == hashlib.sha256(body).hexdigest()
    assert reference.relative_path.startswith("job-001/")
    assert reference.realised_roles == frozenset({"visual_reference"})
    assert reference.requested_roles == frozenset({"written_inspiration", "visual_reference"})
    output = tmp_path / "staging" / reference.relative_path
    assert output.read_bytes() == body
    assert output.resolve().is_relative_to((tmp_path / "staging").resolve())
    assert (tmp_path / "staging" / "job-001" / "reference-manifest.json").is_file()


def test_stager_extracts_bounded_text_for_html_inspiration(tmp_path: Path) -> None:
    stager = ReferenceStager(
        tmp_path / "staging",
        resolver=_public_resolver,
        fetcher=lambda url: FetchedReference(
            final_url=url,
            content_type="text/html",
            body=b"<h1>Useful source</h1><script>ignore_me()</script><p>Grounded details.</p>",
            redirect_count=0,
        ),
    )

    staged = stager.stage((ReferenceRequest("https://example.com/source"),), job_id="job-html")

    assert staged[0].realised_roles == frozenset({"written_inspiration"})
    assert staged[0].text_excerpt == "Useful source Grounded details."


def test_stager_rejects_private_dns_resolution_before_fetch(tmp_path: Path) -> None:
    fetch_called = False

    def _fetcher(url: str) -> FetchedReference:
        nonlocal fetch_called
        fetch_called = True
        raise AssertionError("private DNS target must not be fetched")

    stager = ReferenceStager(
        tmp_path / "staging",
        resolver=lambda hostname: ("10.0.0.8",),
        fetcher=_fetcher,
    )

    with pytest.raises(ReferenceStagingError, match="public"):
        stager.stage((ReferenceRequest("https://example.com/private"),), job_id="job-private")

    assert fetch_called is False
    assert not (tmp_path / "staging" / "job-private").exists()


def test_stager_revalidates_redirect_destination_before_writing(tmp_path: Path) -> None:
    stager = ReferenceStager(
        tmp_path / "staging",
        resolver=_public_resolver,
        fetcher=lambda url: FetchedReference(
            final_url="http://127.0.0.1:8188/system_stats",
            content_type="image/png",
            body=b"not-used",
            redirect_count=1,
        ),
    )

    with pytest.raises(ReferenceStagingError, match="public"):
        stager.stage((ReferenceRequest("https://redirect.example.com/image"),), job_id="job-redirect")

    assert not (tmp_path / "staging" / "job-redirect").exists()


def test_stager_rejects_an_image_claim_with_nonmatching_bytes(tmp_path: Path) -> None:
    stager = ReferenceStager(
        tmp_path / "staging",
        resolver=_public_resolver,
        fetcher=lambda url: FetchedReference(
            final_url=url,
            content_type="image/png",
            body=b"<html>not a png</html>",
            redirect_count=0,
        ),
    )

    with pytest.raises(ReferenceStagingError, match="do not match"):
        stager.stage((ReferenceRequest("https://example.com/source"),), job_id="job-mime")

    assert not (tmp_path / "staging" / "job-mime").exists()


@pytest.mark.parametrize(
    "content_type,body",
    [
        ("application/octet-stream", b"opaque"),
        ("image/png", b"x" * 65),
    ],
)
def test_stager_rejects_disallowed_or_oversized_content(
    tmp_path: Path, content_type: str, body: bytes
) -> None:
    stager = ReferenceStager(
        tmp_path / "staging",
        resolver=_public_resolver,
        fetcher=lambda url: FetchedReference(
            final_url=url,
            content_type=content_type,
            body=body,
            redirect_count=0,
        ),
        max_bytes=64,
    )

    with pytest.raises(ReferenceStagingError):
        stager.stage((ReferenceRequest("https://example.com/source"),), job_id="job-reject")

    assert not (tmp_path / "staging" / "job-reject").exists()


def test_stager_rejects_existing_symlink_job_directory_without_following_it(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging_root / "job-link").symlink_to(outside, target_is_directory=True)

    stager = ReferenceStager(
        staging_root,
        resolver=_public_resolver,
        fetcher=lambda url: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    with pytest.raises(ReferenceStagingError, match="job directory"):
        stager.stage((ReferenceRequest("https://example.com/source"),), job_id="job-link")

    assert list(outside.iterdir()) == []
