"""Offline fixture tests for the YouTube/transcript acquisition hardening layer.

Covers the design artifact's required fixture families (no network, no media,
no cookies, no credentials):
  /home/filip/spearhead-execution/20260528-source-spikes/yt-dlp/followups/
  youtube-transcript-acquisition-hardening.md

Task: t_995513e8 (Mystra source acquisition hardening).
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "skills" / "media" / "youtube-content" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import acquisition as acq
from acquisition import (
    AcquisitionRequest,
    AcquisitionStatus,
    AuthMode,
    AuthPolicy,
    ProviderResult,
    RetryPolicy,
    SourceType,
    TranscriptKind,
    TranscriptProvenance,
    acquire,
    classify_source,
    detect_redactions,
    redact_acquisition_text,
)


# ---------------------------------------------------------------------------
# Test doubles: canned providers that never touch the network.
# ---------------------------------------------------------------------------


class FakeProvider:
    """A provider returning a fixed sequence of ProviderResults."""

    def __init__(self, label, results, *, phase="probe",
                 auth_mode=AuthMode.ANONYMOUS_PUBLIC, supports=True):
        self.label = label
        self.phase = phase
        self.auth_mode = auth_mode
        self._results = list(results)
        self._supports = supports
        self.fetch_calls = 0

    def supports(self, request):
        return self._supports

    def fetch(self, request):
        self.fetch_calls += 1
        if len(self._results) == 1:
            return self._results[0]
        return self._results.pop(0)


def _ok_result(**kw):
    return ProviderResult(
        status=AcquisitionStatus.OK,
        provenance=TranscriptProvenance(
            kind=TranscriptKind.MANUAL, original_language="en",
            requested_language="en", is_asr=False,
        ),
        artifacts={"transcript_text": "hello world", "language": "en"},
        duration_ms=10,
        **kw,
    )


# ===========================================================================
# 1. URL normalization / classification
# ===========================================================================


class TestClassifySource:
    def test_watch_url(self):
        ref = classify_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert ref.source_type is SourceType.VIDEO
        assert ref.video_id == "dQw4w9WgXcQ"

    def test_short_url(self):
        ref = classify_source("https://youtu.be/dQw4w9WgXcQ")
        assert ref.source_type is SourceType.VIDEO
        assert ref.video_id == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        ref = classify_source("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        assert ref.source_type is SourceType.VIDEO

    def test_bare_video_id(self):
        ref = classify_source("dQw4w9WgXcQ")
        assert ref.source_type is SourceType.VIDEO
        assert ref.video_id == "dQw4w9WgXcQ"

    def test_playlist_url(self):
        ref = classify_source("https://www.youtube.com/playlist?list=PL1234567890")
        assert ref.source_type is SourceType.PLAYLIST
        assert ref.playlist_id == "PL1234567890"

    def test_watch_with_list_keeps_both(self):
        ref = classify_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc1234567")
        assert ref.source_type is SourceType.VIDEO
        assert ref.video_id == "dQw4w9WgXcQ"
        assert ref.playlist_id == "PLabc1234567"

    def test_channel_id(self):
        ref = classify_source("https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv")
        assert ref.source_type is SourceType.CHANNEL
        assert ref.channel == "UCabcdefghijklmnopqrstuv"

    def test_channel_handle(self):
        ref = classify_source("https://www.youtube.com/@SomeCreator")
        assert ref.source_type is SourceType.CHANNEL
        assert ref.channel == "@SomeCreator"

    def test_non_youtube_host_unsupported(self):
        ref = classify_source("https://vimeo.com/12345")
        assert ref.source_type is SourceType.UNSUPPORTED
        assert "non-YouTube" in ref.reason

    def test_missing_id_unsupported(self):
        ref = classify_source("https://www.youtube.com/watch")
        assert ref.source_type is SourceType.UNSUPPORTED

    def test_empty_unsupported(self):
        assert classify_source("").source_type is SourceType.UNSUPPORTED
        assert classify_source(None).source_type is SourceType.UNSUPPORTED

    def test_url_hash_is_not_raw_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&auth=secrettoken"
        ref = classify_source(url)
        assert ref.url_hash.startswith("sha256:")
        assert "secrettoken" not in ref.url_hash
        assert url not in ref.url_hash


# ===========================================================================
# 2. Public transcript success
# ===========================================================================


class TestPublicTranscriptSuccess:
    def test_ok_returns_transcript_and_provenance(self):
        provider = FakeProvider("public_transcript_api", [_ok_result()])
        result = acquire("https://youtu.be/dQw4w9WgXcQ", [provider],
                         requested_languages=["en"])
        assert result.status is AcquisitionStatus.OK
        assert result.transcript_provenance.kind is TranscriptKind.MANUAL
        assert result.transcript_provenance.is_asr is False
        assert result.artifacts["transcript_text"] == "hello world"
        ev = result.to_evidence()
        assert ev["status"] == "OK"
        assert ev["video_id"] == "dQw4w9WgXcQ"
        # Evidence must be JSON-serializable.
        json.dumps(ev)

    def test_asr_vs_manual_distinguished(self):
        asr = ProviderResult(
            status=AcquisitionStatus.OK,
            provenance=TranscriptProvenance(
                kind=TranscriptKind.AUTOMATIC, original_language="en",
                requested_language="en", is_asr=True),
            artifacts={"transcript_text": "auto captions"},
            duration_ms=5,
        )
        result = acquire("dQw4w9WgXcQ", [FakeProvider("api", [asr])])
        assert result.transcript_provenance.kind is TranscriptKind.AUTOMATIC
        assert result.transcript_provenance.is_asr is True

    def test_translated_records_source_and_target(self):
        translated = ProviderResult(
            status=AcquisitionStatus.OK,
            provenance=TranscriptProvenance(
                kind=TranscriptKind.TRANSLATED, original_language="en",
                requested_language="cs", translated_from="en", is_asr=False),
            artifacts={"transcript_text": "ahoj"},
            duration_ms=5,
        )
        result = acquire("dQw4w9WgXcQ", [FakeProvider("api", [translated])],
                         requested_languages=["cs"])
        prov = result.transcript_provenance
        assert prov.kind is TranscriptKind.TRANSLATED
        assert prov.translated_from == "en"
        assert prov.requested_language == "cs"

    def test_stops_at_first_ok_no_extra_calls(self):
        first = FakeProvider("api1", [_ok_result()])
        second = FakeProvider("api2", [_ok_result()])
        acquire("dQw4w9WgXcQ", [first, second])
        assert first.fetch_calls == 1
        assert second.fetch_calls == 0


# ===========================================================================
# 3. Playlist / channel expansion metadata preserved
# ===========================================================================


class TestExpansionMetadata:
    def test_playlist_expansion_metadata_preserved(self):
        playlist_meta = ProviderResult(
            status=AcquisitionStatus.PARTIAL_METADATA_ONLY,
            artifacts={"playlist": {
                "id": "PLabc1234567", "title": "Series",
                "entries": [{"video_id": "aaaaaaaaaaa"}, {"video_id": "bbbbbbbbbbb"}],
            }},
            duration_ms=8,
        )
        result = acquire("https://www.youtube.com/playlist?list=PLabc1234567",
                         [FakeProvider("ytdlp_noauth_metadata", [playlist_meta])])
        assert result.source.source_type is SourceType.PLAYLIST
        assert result.status is AcquisitionStatus.PARTIAL_METADATA_ONLY
        entries = result.artifacts["playlist"]["entries"]
        assert len(entries) == 2
        # Provenance must NOT claim a transcript exists.
        assert result.transcript_provenance.kind is TranscriptKind.NONE

    def test_channel_expansion_metadata_preserved(self):
        channel_meta = ProviderResult(
            status=AcquisitionStatus.PARTIAL_METADATA_ONLY,
            artifacts={"channel": {"id": "UCabcdefghijklmnopqrstuv", "video_count": 3}},
            duration_ms=8,
        )
        result = acquire("https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv",
                         [FakeProvider("meta", [channel_meta])])
        assert result.source.source_type is SourceType.CHANNEL
        assert result.artifacts["channel"]["video_count"] == 3


# ===========================================================================
# 4. Disabled transcript / no transcript
# ===========================================================================


class TestDisabledTranscript:
    def test_no_transcript_available(self):
        result = acquire("dQw4w9WgXcQ", [FakeProvider("api", [
            ProviderResult(status=AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE, duration_ms=4)])])
        assert result.status is AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE
        assert result.transcript_provenance.kind is TranscriptKind.NONE
        assert "metadata-only" in result.next_safe_action or "language" in result.next_safe_action

    def test_partial_metadata_then_no_transcript_fallback(self):
        # Provider 1: metadata only. Provider 2: still no transcript.
        p1 = FakeProvider("api", [ProviderResult(
            status=AcquisitionStatus.PARTIAL_METADATA_ONLY,
            artifacts={"title": "vid"}, duration_ms=3)])
        p2 = FakeProvider("ytdlp", [ProviderResult(
            status=AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE, duration_ms=3)])
        result = acquire("dQw4w9WgXcQ", [p1, p2])
        # PARTIAL outranks NO_TRANSCRIPT in precedence; metadata preserved.
        assert result.status is AcquisitionStatus.PARTIAL_METADATA_ONLY
        assert result.artifacts.get("title") == "vid"
        assert p2.fetch_calls == 1  # fallback was attempted


# ===========================================================================
# 5. Login-required marker (hard stop, no auto-auth)
# ===========================================================================


class TestLoginRequired:
    def test_login_required_hard_stops(self):
        gated = FakeProvider("api", [ProviderResult(
            status=AcquisitionStatus.LOGIN_REQUIRED, duration_ms=2)])
        never = FakeProvider("api2", [_ok_result()])
        result = acquire("dQw4w9WgXcQ", [gated, never])
        assert result.status is AcquisitionStatus.LOGIN_REQUIRED
        # No automatic fallback after a login gate.
        assert never.fetch_calls == 0
        assert "approval" in result.next_safe_action.lower()

    def test_captcha_and_geo_hard_stop(self):
        for st in (AcquisitionStatus.CAPTCHA_REQUIRED, AcquisitionStatus.GEO_RESTRICTED):
            nxt = FakeProvider("next", [_ok_result()])
            result = acquire("dQw4w9WgXcQ", [
                FakeProvider("api", [ProviderResult(status=st, duration_ms=1)]), nxt])
            assert result.status is st
            assert nxt.fetch_calls == 0


# ===========================================================================
# 6. Auth / browser provider NOT selected without approval
# ===========================================================================


class TestAuthPolicyDefaultDeny:
    def test_cookie_provider_skipped_by_default(self):
        cookie_provider = FakeProvider(
            "approved_cookie_file", [_ok_result()],
            auth_mode=AuthMode.APPROVED_COOKIE_FILE)
        anon = FakeProvider("anon", [ProviderResult(
            status=AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE, duration_ms=1)])
        result = acquire("dQw4w9WgXcQ", [cookie_provider, anon])
        # Cookie provider never called; anonymous one wins.
        assert cookie_provider.fetch_calls == 0
        assert result.status is AcquisitionStatus.NO_TRANSCRIPT_AVAILABLE

    def test_browser_import_skipped_by_default(self):
        browser = FakeProvider(
            "browser_session_import", [_ok_result()],
            auth_mode=AuthMode.BROWSER_SESSION_IMPORT)
        result = acquire("dQw4w9WgXcQ", [browser])
        assert browser.fetch_calls == 0
        # No provider produced anything → falls through to transient default.
        assert result.status is AcquisitionStatus.NETWORK_TRANSIENT

    def test_cookie_provider_runs_only_with_matching_approval(self):
        cookie_provider = FakeProvider(
            "approved_cookie_file", [_ok_result()],
            auth_mode=AuthMode.APPROVED_COOKIE_FILE)
        policy = AuthPolicy(mode=AuthMode.APPROVED_COOKIE_FILE,
                            approved=True, approval_id="t_63cf2237")
        result = acquire("dQw4w9WgXcQ", [cookie_provider], auth_policy=policy)
        assert cookie_provider.fetch_calls == 1
        assert result.status is AcquisitionStatus.OK

    def test_approval_without_id_does_not_permit(self):
        policy = AuthPolicy(mode=AuthMode.APPROVED_COOKIE_FILE, approved=True, approval_id=None)
        assert policy.permits(AuthMode.APPROVED_COOKIE_FILE) is False
        assert policy.permits(AuthMode.ANONYMOUS_PUBLIC) is True

    def test_approval_for_wrong_mode_does_not_permit(self):
        policy = AuthPolicy(mode=AuthMode.APPROVED_COOKIE_FILE, approved=True, approval_id="x")
        assert policy.permits(AuthMode.BROWSER_SESSION_IMPORT) is False


# ===========================================================================
# 7. Retry / backoff
# ===========================================================================


class TestRetryBackoff:
    def test_transient_succeeds_after_retry(self):
        provider = FakeProvider("api", [
            ProviderResult(status=AcquisitionStatus.NETWORK_TRANSIENT, duration_ms=1),
            _ok_result(),
        ])
        slept = []
        result = acquire("dQw4w9WgXcQ", [provider],
                         sleeper=slept.append, rng=lambda a, b: a)
        assert result.status is AcquisitionStatus.OK
        assert provider.fetch_calls == 2
        assert len(slept) == 1  # one backoff between the two attempts

    def test_transient_exhausted_after_max_retries(self):
        provider = FakeProvider("api", [ProviderResult(
            status=AcquisitionStatus.NETWORK_TRANSIENT, duration_ms=1)])
        slept = []
        result = acquire("dQw4w9WgXcQ", [provider],
                         retry_policy=RetryPolicy(max_transient_retries=2),
                         sleeper=slept.append, rng=lambda a, b: a)
        assert result.status is AcquisitionStatus.NETWORK_TRANSIENT
        # initial + 2 retries = 3 fetches, 2 sleeps
        assert provider.fetch_calls == 3
        assert len(slept) == 2

    def test_non_retryable_stops_immediately(self):
        provider = FakeProvider("api", [ProviderResult(
            status=AcquisitionStatus.TOKEN_REQUIRED, duration_ms=1)])
        slept = []
        result = acquire("dQw4w9WgXcQ", [provider], sleeper=slept.append)
        assert result.status is AcquisitionStatus.TOKEN_REQUIRED
        assert provider.fetch_calls == 1
        assert slept == []

    def test_rate_limited_honors_retry_after(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=0.0)
        # retry_after larger than computed backoff should win.
        delay = policy.backoff_delay(1, retry_after=12.0, rng=lambda a, b: a)
        assert delay == 12.0

    def test_backoff_clamped_to_max(self):
        policy = RetryPolicy(base_delay=10.0, max_delay=15.0, jitter=0.0)
        assert policy.backoff_delay(5, rng=lambda a, b: a) == 15.0

    def test_backoff_exponential_with_zero_jitter(self):
        policy = RetryPolicy(base_delay=1.0, jitter=0.0)
        assert policy.backoff_delay(1, rng=lambda a, b: a) == 1.0
        assert policy.backoff_delay(2, rng=lambda a, b: a) == 2.0
        assert policy.backoff_delay(3, rng=lambda a, b: a) == 4.0


# ===========================================================================
# 8. Unsupported URL short-circuit
# ===========================================================================


class TestUnsupportedShortCircuit:
    def test_unsupported_url_no_provider_called(self):
        provider = FakeProvider("api", [_ok_result()])
        result = acquire("https://vimeo.com/123", [provider])
        assert result.status is AcquisitionStatus.UNSUPPORTED_URL
        assert provider.fetch_calls == 0


# ===========================================================================
# 9. Redaction
# ===========================================================================


class TestRedaction:
    def test_cookie_value_redacted(self):
        text = "Cookie: SAPISID=AbCdEf123456; HSID=secrethsid; other=keepme"
        out = redact_acquisition_text(text)
        assert "AbCdEf123456" not in out
        assert "secrethsid" not in out
        assert "***" in out

    def test_named_cookie_inline_redacted(self):
        out = redact_acquisition_text("debug SAPISID=topsecretvalue end")
        assert "topsecretvalue" not in out
        assert "SAPISID=***" in out

    def test_po_token_and_visitor_data_redacted(self):
        text = '{"poToken": "POABCDEFGH", "visitorData": "VDxyz123"}'
        out = redact_acquisition_text(text)
        assert "POABCDEFGH" not in out
        assert "VDxyz123" not in out

    def test_datasync_id_redacted(self):
        out = redact_acquisition_text("data_sync_id=DS123456789")
        assert "DS123456789" not in out

    def test_authorization_header_redacted(self):
        out = redact_acquisition_text("Authorization: Bearer abc.def.ghi-very-long-token-value")
        assert "very-long-token-value" not in out

    def test_signed_googlevideo_url_hashed(self):
        url = "https://r1---sn-abc.googlevideo.com/videoplayback?sig=SECRETSIG&pot=SECRETPOT"
        out = redact_acquisition_text(f"media url {url}")
        assert "SECRETSIG" not in out
        assert "SECRETPOT" not in out
        assert "sha256:" in out

    def test_detect_redactions_categories(self):
        text = "Cookie: SAPISID=x; poToken=y; visitorData=z https://x.googlevideo.com/p?sig=q"
        cats = detect_redactions(text)
        assert "cookie" in cats
        assert "po_token" in cats
        assert "visitor_data" in cats
        assert "signed_url" in cats

    def test_non_secret_text_passes_through(self):
        text = "This is a normal transcript line about cats."
        assert redact_acquisition_text(text) == text

    def test_none_passthrough(self):
        assert redact_acquisition_text(None) is None

    def test_evidence_artifacts_are_redacted(self):
        leaky = ProviderResult(
            status=AcquisitionStatus.PARTIAL_METADATA_ONLY,
            artifacts={"debug_headers": "Cookie: SAPISID=leakyvalue123"},
            duration_ms=2,
        )
        result = acquire("dQw4w9WgXcQ", [FakeProvider("api", [leaky])])
        ev = result.to_evidence()
        blob = json.dumps(ev)
        assert "leakyvalue123" not in blob


# ===========================================================================
# 10. No-hallucination partial-source handling
# ===========================================================================


class TestNoHallucination:
    def test_partial_does_not_invent_transcript(self):
        # A buggy provider returns PARTIAL but mistakenly sets a transcript kind.
        bad = ProviderResult(
            status=AcquisitionStatus.PARTIAL_METADATA_ONLY,
            provenance=TranscriptProvenance(kind=TranscriptKind.MANUAL, is_asr=False),
            artifacts={"title": "only metadata"},
            duration_ms=2,
        )
        result = acquire("dQw4w9WgXcQ", [FakeProvider("api", [bad])])
        # Hardening layer forces provenance back to NONE when status != OK.
        assert result.transcript_provenance.kind is TranscriptKind.NONE
        assert "transcript_text" not in result.artifacts

    def test_evidence_shape_has_required_keys(self):
        result = acquire("dQw4w9WgXcQ", [FakeProvider("api", [_ok_result()])],
                         requested_languages=["en"])
        ev = result.to_evidence()
        for key in ("request_id", "url_hash", "source_type", "status",
                    "providers", "transcript_provenance", "next_safe_action"):
            assert key in ev
        assert ev["request_id"].startswith("yt-")


# ===========================================================================
# 11. Taxonomy completeness
# ===========================================================================


class TestTaxonomyCompleteness:
    def test_all_statuses_have_next_safe_action(self):
        for status in AcquisitionStatus:
            assert acq.next_safe_action(status)

    def test_retryable_set_only_transient_classes(self):
        assert acq.RETRYABLE_STATUSES == frozenset({
            AcquisitionStatus.NETWORK_TRANSIENT,
            AcquisitionStatus.RATE_LIMITED,
        })

    def test_request_id_deterministic_from_url(self):
        r1 = acquire("dQw4w9WgXcQ", [FakeProvider("a", [_ok_result()])])
        r2 = acquire("dQw4w9WgXcQ", [FakeProvider("a", [_ok_result()])])
        assert r1.request_id == r2.request_id
