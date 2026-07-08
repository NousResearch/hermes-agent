"""Browser-assisted Teams meeting recording download helpers."""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from plugins.teams_context.recording import RecordingIngestError, ingest_recording
from plugins.teams_context.store import TeamsContextStore


DEFAULT_CDP_URL = "http://127.0.0.1:9222"
REPORT_FILENAME = "teams-context-download-report.json"
RECORDING_SUFFIXES = {".mp4", ".m4v", ".mov", ".webm", ".mkv"}
TRANSCRIPT_SUFFIXES = {".vtt", ".srt", ".txt"}


class MeetingDownloadError(RuntimeError):
    """Raised when browser-assisted meeting download cannot proceed."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class DownloadedMeetingFiles:
    recording_path: Path | None = None
    transcript_path: Path | None = None
    skipped: bool = False


@dataclass
class MeetingDownloadResult:
    url: str
    sanitized_url: str
    status: str
    retryable: bool = False
    meeting_label: str | None = None
    recording_path: str | None = None
    transcript_path: str | None = None
    report_path: str | None = None
    ingested: dict[str, Any] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_report_item(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("url", None)
        return payload


def classify_sharepoint_stream_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if not parsed.scheme or parsed.scheme not in {"http", "https"}:
        return "unsupported"
    if "sharepoint.com" in host:
        return "sharepoint"
    if "microsoftstream.com" in host or "stream.microsoft.com" in host:
        return "stream"
    if "office.com" in host and "stream" in path:
        return "stream"
    if "teams.microsoft.com" in host:
        return "teams"
    return "unsupported"


def sanitize_url_for_metadata(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    path = urllib.parse.quote(urllib.parse.unquote(parsed.path or ""), safe="/:@")
    return urllib.parse.urlunparse((parsed.scheme, netloc, path, "", "", ""))


def safe_filename(value: str, *, fallback: str = "teams-meeting") -> str:
    text = urllib.parse.unquote(str(value or "")).strip()
    text = re.sub(r"[^\w\s().-]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        text = fallback
    text = text.replace(" ", "-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:120].strip(".-") or fallback


def parse_url_file(path: str | Path) -> list[str]:
    urls: list[str] = []
    for raw_line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def pair_downloaded_files(paths: Iterable[str | Path]) -> DownloadedMeetingFiles:
    recordings: list[Path] = []
    transcripts: list[Path] = []
    for item in paths:
        path = Path(item)
        suffix = path.suffix.lower()
        if suffix in RECORDING_SUFFIXES:
            recordings.append(path)
        elif suffix in TRANSCRIPT_SUFFIXES:
            transcripts.append(path)
    recordings.sort(key=lambda p: (p.stem.lower(), p.name.lower()))
    transcripts.sort(key=lambda p: (p.stem.lower(), p.name.lower()))
    recording = recordings[0] if recordings else None
    transcript = _best_transcript_for_recording(recording, transcripts) if recording else (transcripts[0] if transcripts else None)
    return DownloadedMeetingFiles(recording_path=recording, transcript_path=transcript)


def existing_meeting_files(output_dir: str | Path, base_name: str) -> DownloadedMeetingFiles:
    directory = Path(output_dir).expanduser()
    candidates = []
    if directory.exists():
        candidates = [path for path in directory.iterdir() if path.is_file() and path.stem == base_name]
    paired = pair_downloaded_files(candidates)
    paired.skipped = paired.recording_path is not None
    return paired


def write_download_report(output_dir: str | Path, results: list[MeetingDownloadResult]) -> Path:
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / REPORT_FILENAME
    for result in results:
        result.report_path = str(report_path)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [result.to_report_item() for result in results],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def download_meetings(
    urls: list[str],
    *,
    output_dir: str | Path,
    cdp_url: str = DEFAULT_CDP_URL,
    store: TeamsContextStore | None = None,
    artifact_cache: str | Path | None = None,
    force: bool = False,
    ingest: bool = True,
    browser_downloader: Callable[..., DownloadedMeetingFiles] | None = None,
) -> dict[str, Any]:
    results = [
        download_meeting(
            url,
            output_dir=output_dir,
            cdp_url=cdp_url,
            store=store,
            artifact_cache=artifact_cache,
            force=force,
            ingest=ingest,
            browser_downloader=browser_downloader,
        )
        for url in urls
    ]
    report_path = write_download_report(output_dir, results)
    return {
        "report_path": str(report_path),
        "results": [result.to_report_item() for result in results],
        "succeeded": sum(1 for result in results if result.status in {"downloaded", "skipped", "ingested"}),
        "failed": sum(1 for result in results if result.status == "failed"),
    }


def download_meeting(
    url: str,
    *,
    output_dir: str | Path,
    cdp_url: str = DEFAULT_CDP_URL,
    store: TeamsContextStore | None = None,
    artifact_cache: str | Path | None = None,
    force: bool = False,
    ingest: bool = True,
    browser_downloader: Callable[..., DownloadedMeetingFiles] | None = None,
) -> MeetingDownloadResult:
    sanitized_url = sanitize_url_for_metadata(url)
    if classify_sharepoint_stream_url(url) == "unsupported":
        return MeetingDownloadResult(
            url=url,
            sanitized_url=sanitized_url,
            status="failed",
            retryable=False,
            error="URL is not a supported Teams, SharePoint, or Stream meeting link.",
        )
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    fallback_base = _base_name_from_url(sanitized_url)
    preexisting = existing_meeting_files(directory, fallback_base)
    if preexisting.recording_path and not force:
        return _finish_download_result(
            url=url,
            sanitized_url=sanitized_url,
            status="skipped",
            meeting_label=fallback_base,
            files=preexisting,
            store=store,
            artifact_cache=artifact_cache,
            ingest_requested=ingest,
        )

    try:
        downloader = browser_downloader or download_meeting_with_browser
        files = downloader(
            url=url,
            output_dir=directory,
            cdp_url=cdp_url,
            base_name=fallback_base,
            force=force,
        )
        if files.recording_path is None:
            raise MeetingDownloadError(
                "Browser did not produce a recording file. Confirm the meeting page exposes a download action.",
                retryable=True,
            )
        label = files.recording_path.stem
        return _finish_download_result(
            url=url,
            sanitized_url=sanitized_url,
            status="downloaded",
            meeting_label=label,
            files=files,
            store=store,
            artifact_cache=artifact_cache,
            ingest_requested=ingest,
        )
    except MeetingDownloadError as exc:
        return MeetingDownloadResult(
            url=url,
            sanitized_url=sanitized_url,
            status="failed",
            retryable=exc.retryable,
            meeting_label=fallback_base,
            error=str(exc),
        )


def download_meeting_with_browser(
    *,
    url: str,
    output_dir: str | Path,
    cdp_url: str = DEFAULT_CDP_URL,
    base_name: str | None = None,
    force: bool = False,
) -> DownloadedMeetingFiles:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise MeetingDownloadError(
            "Meeting download requires Playwright and a Chrome DevTools endpoint. "
            "Install Playwright or run in a Hermes environment that includes it.",
            retryable=False,
        ) from exc

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    before = _snapshot_files(directory)
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise MeetingDownloadError(
                f"Could not connect to Chrome DevTools at {cdp_url}. "
                "Start Chrome with remote debugging enabled and sign into Microsoft 365.",
                retryable=True,
            ) from exc
        context = browser.contexts[0] if browser.contexts else browser.new_context(accept_downloads=True)
        try:
            page = context.new_page()
        except Exception:
            page = context.pages[0] if context.pages else browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1500)
        if _looks_logged_out(page):
            raise MeetingDownloadError(
                "The browser appears to be signed out. Sign into Microsoft 365 in Chrome and retry.",
                retryable=True,
            )
        title = _detect_meeting_title(page) or base_name or _base_name_from_url(sanitize_url_for_metadata(url))
        stable_base = safe_filename(title)
        existing = existing_meeting_files(directory, stable_base)
        if existing.recording_path and not force:
            return existing
        download_paths = _click_available_downloads(
            page,
            output_dir=directory,
            base_name=stable_base,
            timeout_error=PlaywrightTimeoutError,
        )
    after = _snapshot_files(directory)
    discovered = [path for path in after - before if path.exists()]
    return pair_downloaded_files([*download_paths, *discovered])


def _finish_download_result(
    *,
    url: str,
    sanitized_url: str,
    status: str,
    meeting_label: str,
    files: DownloadedMeetingFiles,
    store: TeamsContextStore | None,
    artifact_cache: str | Path | None,
    ingest_requested: bool,
) -> MeetingDownloadResult:
    result = MeetingDownloadResult(
        url=url,
        sanitized_url=sanitized_url,
        status=status,
        retryable=False,
        meeting_label=meeting_label,
        recording_path=str(files.recording_path) if files.recording_path else None,
        transcript_path=str(files.transcript_path) if files.transcript_path else None,
    )
    if not ingest_requested:
        return result
    if not files.recording_path:
        result.status = "failed"
        result.retryable = True
        result.error = "No recording file was available for ingestion."
        return result
    try:
        result.ingested = ingest_recording(
            str(files.recording_path),
            meeting_label=meeting_label,
            transcript_path=str(files.transcript_path) if files.transcript_path else None,
            store=store or TeamsContextStore(None),
            artifact_cache=artifact_cache,
            metadata={
                "source_url": sanitized_url,
                "downloaded_recording_path": str(files.recording_path),
                "downloaded_transcript_path": str(files.transcript_path) if files.transcript_path else None,
            },
        )
        result.status = "ingested"
    except RecordingIngestError as exc:
        result.status = "failed"
        result.retryable = False
        result.error = f"Downloaded recording, but ingestion failed: {exc}"
    return result


def _click_available_downloads(page: Any, *, output_dir: Path, base_name: str, timeout_error: type[Exception]) -> list[Path]:
    downloaded: list[Path] = []
    download_names = [
        re.compile(r"^download$", re.I),
        re.compile(r"download video", re.I),
        re.compile(r"download recording", re.I),
        re.compile(r"download transcript", re.I),
        re.compile(r"download captions", re.I),
    ]
    for name in download_names:
        try:
            locator = page.get_by_role("button", name=name).first
            if locator.count() == 0:
                locator = page.get_by_role("link", name=name).first
            if locator.count() == 0:
                continue
            with page.expect_download(timeout=15000) as download_info:
                locator.click()
            downloaded.append(_save_playwright_download(download_info.value, output_dir=output_dir, base_name=base_name))
            page.wait_for_timeout(500)
        except timeout_error:
            continue
        except Exception:
            continue
    if downloaded:
        return downloaded
    try:
        more = page.get_by_role("button", name=re.compile(r"more|settings and more|actions|options", re.I)).first
        if more.count() > 0:
            more.click()
            page.wait_for_timeout(300)
            for name in download_names:
                item = page.get_by_role("menuitem", name=name).first
                if item.count() == 0:
                    continue
                with page.expect_download(timeout=15000) as download_info:
                    item.click()
                downloaded.append(_save_playwright_download(download_info.value, output_dir=output_dir, base_name=base_name))
    except timeout_error:
        pass
    except Exception:
        pass
    if not downloaded:
        raise MeetingDownloadError(
            "No recording or transcript download action was visible. "
            "Open the Stream/SharePoint meeting page in the signed-in Chrome session and retry.",
            retryable=True,
        )
    return downloaded


def _save_playwright_download(download: Any, *, output_dir: Path, base_name: str) -> Path:
    suggested = safe_filename(getattr(download, "suggested_filename", "") or base_name)
    suffix = Path(suggested).suffix.lower()
    if not suffix:
        suffix = ".bin"
    stem = Path(suggested).stem
    if stem.lower() in {"download", "videoplayback", "recording"}:
        stem = base_name
    destination = output_dir / f"{safe_filename(stem)}{suffix}"
    download.save_as(str(destination))
    return destination


def _best_transcript_for_recording(recording: Path | None, transcripts: list[Path]) -> Path | None:
    if not transcripts:
        return None
    if recording is None:
        return transcripts[0]
    recording_stem = recording.stem.lower()
    for transcript in transcripts:
        if transcript.stem.lower() == recording_stem:
            return transcript
    for transcript in transcripts:
        if transcript.stem.lower() in recording_stem or recording_stem in transcript.stem.lower():
            return transcript
    return transcripts[0]


def _base_name_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).stem
    if not name:
        name = parsed.hostname or "teams-meeting"
    return safe_filename(name)


def _snapshot_files(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {path for path in directory.iterdir() if path.is_file()}


def _looks_logged_out(page: Any) -> bool:
    try:
        text = page.locator("body").inner_text(timeout=3000).lower()
    except Exception:
        return False
    return any(marker in text for marker in ("sign in", "signin", "pick an account", "use another account"))


def _detect_meeting_title(page: Any) -> str | None:
    for selector in ("h1", "[data-automationid='TitleTextId']", "[role='heading']"):
        try:
            text = page.locator(selector).first.inner_text(timeout=3000).strip()
        except Exception:
            continue
        if text:
            return text
    try:
        title = page.title().strip()
    except Exception:
        title = ""
    return title or None
