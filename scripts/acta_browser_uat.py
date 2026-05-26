#!/usr/bin/env python3
"""Real-browser UAT/regression harness for the Acta dashboard.

This intentionally uses an actual Chromium/Chrome binary instead of DOM-only
unit tests. It catches regressions where the generated HTML exists but the
browser-rendered document no longer preserves Acta's operator feed contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Mapping, cast
from urllib.parse import urljoin, urlparse


DEV_ROW_RE = re.compile(r"<(?:section|article)\b(?=[^>]*\bclass=\"[^\"]*(?:brief-row|lead)\b)(?=[^>]*\bdata-feed-lane=\"dev\")[\s\S]*?</(?:section|article)>", re.I)
DAILY_ROW_RE = re.compile(r"<(?:section|article)\b(?=[^>]*\bclass=\"[^\"]*(?:brief-row|lead)\b)(?=[^>]*\bdata-feed-lane=\"daily\")[\s\S]*?</(?:section|article)>", re.I)
DEV_JOB_RE = re.compile(
    r"startup sprint|sprint ceo|self-healing sentinel|user[- ]testing sweep|qa pipeline|qa canary|operator sprint|security scan|app security|vesta import|vesta startup|acta startup|minerva startup|praetor startup",
    re.I,
)
CONFIDENCE_CHIP_RE = re.compile(r"\bCONF\s+(?:HIGH|MED|LOW[-/]GAP)\b", re.I)
OUTPUT_CONFIDENCE_RE = re.compile(r"\b(?:CONF\s+(?:HIGH|MED|LOW[-/]GAP)|CATALOG)\b", re.I)
OUTPUT_LEAK_RE = re.compile(
    r"##\s*(?:Prompt|Tool)\b|tool\s+output|Traceback|/Users/|C:\\Users\\|HERMES_HOME|api_key=",
    re.I,
)
HTML_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class _ClassTextExtractor(HTMLParser):
    """Extract browser-DOM-ish text from elements carrying a CSS class."""

    def __init__(self, class_name: str):
        super().__init__(convert_charrefs=True)
        self.class_name = class_name
        self.depth = 0
        self.current: list[str] = []
        self.matches: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        classes = ""
        for key, value in attrs:
            if key.lower() == "class" and value:
                classes = value
                break
        if self.depth == 0 and self.class_name in classes.split():
            self.depth = 1
            self.current = []
        elif self.depth and tag not in HTML_VOID_TAGS:
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if not self.depth:
            return
        self.depth -= 1
        if self.depth == 0:
            text = " ".join(" ".join(self.current).split())
            self.matches.append(text)
            self.current = []

    def handle_data(self, data: str) -> None:
        if self.depth and data.strip():
            self.current.append(data.strip())


class _ClassHtmlExtractor(HTMLParser):
    """Extract raw HTML from elements carrying a CSS class."""

    VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, dom: str, class_name: str):
        super().__init__(convert_charrefs=False)
        self.dom = dom
        self.class_name = class_name
        self.line_offsets = [0]
        for match in re.finditer(r"\n", dom):
            self.line_offsets.append(match.end())
        self.depth = 0
        self.start_offset = 0
        self.matches: list[str] = []

    def _absolute_offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        classes = ""
        for key, value in attrs:
            if key.lower() == "class" and value:
                classes = value
                break
        if self.depth == 0 and self.class_name in classes.split():
            self.depth = 1
            self.start_offset = self._absolute_offset()
        elif self.depth and tag not in self.VOID_TAGS:
            self.depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing nested tags do not affect the enclosing element depth.
        return

    def handle_endtag(self, tag: str) -> None:
        if not self.depth:
            return
        self.depth -= 1
        if self.depth == 0:
            end_offset = self._absolute_offset() + len(f"</{tag}>")
            self.matches.append(self.dom[self.start_offset:end_offset])


@dataclass
class BrowserResult:
    url: str
    dom: str
    screenshot: Path
    browser_path: Path
    console_output: str = ""
    errors_output: str = ""
    viewport_width: int = 390
    viewport_height: int = 844
    layout_metrics: dict[str, object] | None = None
    horizontal_overflow: bool = False
    action_state_probe: dict[str, object] | None = None


def _extract_text_by_class(dom: str, class_name: str) -> list[str]:
    parser = _ClassTextExtractor(class_name)
    parser.feed(dom)
    parser.close()
    return parser.matches


def _extract_html_by_class(dom: str, class_name: str) -> list[str]:
    parser = _ClassHtmlExtractor(dom, class_name)
    parser.feed(dom)
    parser.close()
    return parser.matches


def _is_usable_open_url(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    if not stripped or stripped == "#":
        return False
    if stripped.lower().startswith("javascript:"):
        return False
    return True


_ARTIFACT_OPEN_ANCHOR_CLASSES = {
    "artifact-open",
    "artifact-open-overlay",
    "open-action",
    "output-open",
    "output-open-overlay",
}
_ARTIFACT_OPEN_EXCLUDED_ANCHOR_CLASSES = {"ask", "ask-label", "followup", "follow-up", "followup-meta", "telegram"}


class _ClickableOpenAffordanceParser(HTMLParser):
    """Detect usable row-level or artifact-open anchor affordances in row HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.has_affordance = False
        self._seen_root = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.has_affordance:
            return
        tag = tag.lower()
        attr_map = {key.lower(): value for key, value in attrs}
        if not self._seen_root:
            self._seen_root = True
            if _is_usable_open_url(attr_map.get("data-open-url")):
                self.has_affordance = True
                return
        if tag != "a" or not _is_usable_open_url(attr_map.get("href")):
            return
        classes = set((attr_map.get("class") or "").lower().split())
        if classes & _ARTIFACT_OPEN_ANCHOR_CLASSES and not classes & _ARTIFACT_OPEN_EXCLUDED_ANCHOR_CLASSES:
            self.has_affordance = True
            return


def _has_clickable_open_affordance(row_html: str) -> bool:
    parser = _ClickableOpenAffordanceParser()
    parser.feed(row_html)
    parser.close()
    return parser.has_affordance


class _FirstOutputArtifactTargetParser(HTMLParser):
    """Find the first real artifact-open target in rendered Outputs row HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.target: str | None = None
        self._seen_root = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.target is not None:
            return
        tag = tag.lower()
        attr_map = {key.lower(): value for key, value in attrs}
        if not self._seen_root:
            self._seen_root = True
            data_open_url = attr_map.get("data-open-url")
            if _is_usable_open_url(data_open_url):
                self.target = data_open_url.strip() if data_open_url is not None else None
                return
        if tag != "a" or not _is_usable_open_url(attr_map.get("href")):
            return
        classes = set((attr_map.get("class") or "").lower().split())
        if classes & _ARTIFACT_OPEN_ANCHOR_CLASSES and not classes & _ARTIFACT_OPEN_EXCLUDED_ANCHOR_CLASSES:
            href = attr_map.get("href")
            self.target = href.strip() if href is not None else None


def _safe_output_artifact_url(target: str | None, base_url: str) -> str | None:
    if not _is_usable_open_url(target):
        return None
    assert target is not None
    target = target.strip()
    if target.startswith("//") or "\\" in target:
        return None
    parsed_target = urlparse(target)
    parsed_base = urlparse(base_url)
    target_path_lower = parsed_target.path.lower()
    if not parsed_target.path:
        return None
    if ".." in parsed_target.path or "%2e" in target_path_lower or "%2f" in target_path_lower:
        return None
    if parsed_target.username is not None or parsed_target.password is not None:
        return None
    if parsed_target.scheme and parsed_target.scheme not in {"http", "https"}:
        return None
    if parsed_base.scheme == "file":
        if parsed_target.scheme or parsed_target.netloc or target.startswith("/"):
            return None
        target_path = parsed_target.path
        if "/" in target_path or ".." in target_path or "%2e" in target_path.lower():
            return None
        return urljoin(base_url, target)
    if parsed_base.scheme not in {"http", "https"}:
        return None
    resolved = urljoin(base_url, target)
    parsed_resolved = urlparse(resolved)
    if parsed_resolved.scheme not in {"http", "https"}:
        return None
    if parsed_resolved.username is not None or parsed_resolved.password is not None:
        return None
    if (parsed_resolved.scheme, parsed_resolved.hostname, parsed_resolved.port) != (
        parsed_base.scheme,
        parsed_base.hostname,
        parsed_base.port,
    ):
        return None
    if parsed_resolved.hostname == "t.me" or (parsed_resolved.hostname or "").endswith(".t.me"):
        return None
    return resolved


def _first_output_artifact_url(dom: str, base_url: str) -> str | None:
    for row_html in _extract_html_by_class(dom, "output-row"):
        parser = _FirstOutputArtifactTargetParser()
        parser.feed(row_html)
        parser.close()
        safe_url = _safe_output_artifact_url(parser.target, base_url)
        if safe_url:
            return safe_url
    return None


class _ArchiveHrefParser(HTMLParser):
    """Extract anchor hrefs from an archive-card fragment."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href":
                self.hrefs.append(value)
                return


def _archive_card_hrefs(card_html: str) -> list[str | None]:
    parser = _ArchiveHrefParser()
    parser.feed(card_html)
    parser.close()
    return parser.hrefs


def _is_safe_archive_href(value: str | None) -> bool:
    if value is None or value != value.strip():
        return False
    match = re.fullmatch(r"/archive/(\d{4}-\d{2}-\d{2})", value)
    if not match:
        return False
    try:
        date.fromisoformat(match.group(1))
    except ValueError:
        return False
    return True


def _has_numeric_source_signal_counts(card_text: str) -> bool:
    return bool(re.search(r"\bVisible\s+\d+\b.*\bSilent\s+\d+\b.*\bMissing\s+\d+\b", card_text, re.I | re.S))


def _agent_browser_command() -> list[str]:
    env = os.environ.get("ACTA_UAT_AGENT_BROWSER")
    if env:
        return [env]
    direct = shutil.which("agent-browser")
    if direct:
        return [direct]
    npx = shutil.which("npx")
    if npx:
        return [npx, "agent-browser"]
    raise RuntimeError("agent-browser CLI not found. Run `npm install` or install browser tools, then retry.")


def _browser_identity(command: list[str]) -> Path:
    # agent-browser owns browser discovery and launches the installed Chrome for Testing.
    return Path(" ".join(command))


def _target_url(args: argparse.Namespace) -> str:
    if bool(args.html) == bool(args.url):
        raise SystemExit("Pass exactly one of --html or --url")
    if args.url:
        parsed = urlparse(args.url)
        if parsed.scheme not in {"http", "https"}:
            raise SystemExit("--url must be http(s)")
        if parsed.username or parsed.password:
            raise SystemExit("--url must not include userinfo")
        try:
            parsed.port
        except ValueError as exc:
            raise SystemExit(f"Invalid --url port: {exc}") from exc
        if not parsed.hostname:
            raise SystemExit("--url must include a host")
        return args.url
    html_path = Path(args.html).expanduser().resolve()
    if not html_path.exists():
        raise SystemExit(f"HTML file not found: {html_path}")
    return html_path.as_uri()


def _report_url(url: str) -> str:
    """Return a report-safe URL without credentials, query strings, or fragments."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return f"file://{parsed.path}"
    if parsed.scheme not in {"http", "https"}:
        return f"{parsed.scheme}:" if parsed.scheme else parsed.path
    if parsed.username or parsed.password:
        raise ValueError("Report URL must not include userinfo")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid report URL port: {exc}") from exc
    if not parsed.hostname:
        raise ValueError("Report URL must include a host")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{netloc}{parsed.path or '/'}"


def _run_agent_browser(command: list[str], args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [*command, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        cmd = " ".join(str(part) for part in [*command, *args])
        output = exc.output or exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        raise RuntimeError(
            "agent-browser command timed out after "
            f"{timeout}s: {_sanitize_diagnostic_output(cmd)}\n{_sanitize_diagnostic_output(str(output))}"
        ) from exc


def _checked_agent_browser(command: list[str], args: list[str], timeout: int, label: str) -> subprocess.CompletedProcess[str]:
    result = _run_agent_browser(command, args, timeout)
    if result.returncode != 0:
        raise RuntimeError(f"agent-browser {label} failed:\n{_sanitize_diagnostic_output(result.stdout)}")
    return result


def _parse_eval_json(raw_output: str) -> object:
    raw = raw_output.strip()
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, str):
        stripped = parsed.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return parsed
    return parsed


def _sanitize_diagnostic_output(output: str, limit: int = 4000) -> str:
    """Keep browser diagnostics useful without archiving obvious secrets or huge logs."""
    sanitized = re.sub(r"([?&](?:token|api_key|key|secret|sig|signature|password|auth|code)=)[^\s&#]+", r"\1[REDACTED]", output, flags=re.I)
    sanitized = re.sub(
        r"\b((?:Bearer|Token|Api-Key|X-Api-Key)(?:\s*:\s*|\s+))[A-Za-z0-9._~+/=-]{12,}",
        lambda match: match.group(1) + "[REDACTED]",
        sanitized,
        flags=re.I,
    )
    if len(sanitized) > limit:
        return f"{sanitized[:limit]}\n...[truncated {len(sanitized) - limit} chars]"
    return sanitized


def _run_chrome(url: str, artifact_dir: Path, timeout: int, viewport_width: int = 390, viewport_height: int = 844) -> BrowserResult:
    command = _agent_browser_command()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshot = artifact_dir / "acta-uat.png"
    try:
        _checked_agent_browser(command, ["set", "viewport", str(viewport_width), str(viewport_height)], timeout, "set viewport")
        _checked_agent_browser(command, ["console", "--clear"], timeout, "console clear")
        _checked_agent_browser(command, ["errors", "--clear"], timeout, "errors clear")
        _checked_agent_browser(command, ["open", url], timeout, "open")
        # Give any inline read-state script a browser turn before collecting DOM.
        _run_agent_browser(command, ["wait", "250"], timeout)
        dom_result = _checked_agent_browser(command, ["eval", "document.documentElement.outerHTML"], timeout, "DOM eval")
        action_probe_expr = r"""
JSON.stringify((function(){
  var row=document.querySelector('.readable[data-read-key]');
  if(!row) return {skipped:true, reason:'no-readable-row'};
  var save=row.querySelector('.state-toggle[data-state-action="save"]');
  var dismiss=row.querySelector('.state-toggle[data-state-action="dismiss"]');
  var later=row.querySelector('.state-toggle[data-state-action="later"]');
  var overlay=row.querySelector('.row-open-overlay');
  var before=location.href;
  if(!save || !dismiss || !later || !overlay) return {skipped:false, ok:false, reason:'missing-action-or-overlay'};
  function ensurePressed(button){
    var wasPressed=button.getAttribute('aria-pressed')==='true';
    if(wasPressed) button.click();
    button.click();
    return wasPressed;
  }
  function restorePressed(button, wasPressed){
    var isPressed=button.getAttribute('aria-pressed')==='true';
    if(isPressed !== wasPressed) button.click();
  }
  var saveWasPressed=ensurePressed(save);
  var saveOk=row.classList.contains('saved') && save.textContent.trim()==='Saved' && save.getAttribute('aria-pressed')==='true' && location.href===before;
  var dismissWasPressed=ensurePressed(dismiss);
  var dismissOk=row.classList.contains('dismissed') && dismiss.textContent.trim()==='Dismissed' && dismiss.getAttribute('aria-pressed')==='true' && location.href===before;
  var laterWasPressed=ensurePressed(later);
  var laterOk=row.classList.contains('read-later') && later.textContent.trim()==='Later' && later.getAttribute('aria-pressed')==='true' && location.href===before;
  var wasRead=row.classList.contains('read');
  var readToggle=row.querySelector('.read-toggle');
  var overlayHref='';
  overlay.addEventListener('click', function(ev){ ev.preventDefault(); overlayHref=overlay.href; }, {once:true});
  overlay.click();
  var overlayOk=!!overlayHref && row.classList.contains('read') && location.href===before;
  if(row.classList.contains('read') !== wasRead && readToggle) readToggle.click();
  restorePressed(later, laterWasPressed);
  restorePressed(dismiss, dismissWasPressed);
  restorePressed(save, saveWasPressed);
  var unsafeHasActions=Array.prototype.some.call(document.querySelectorAll('.brief-row:not(.readable), .lead:not(.readable)'), function(el){ return !!el.querySelector('.state-toggle,[data-state-action]'); });
  return {skipped:false, ok:saveOk && dismissOk && laterOk && overlayOk && !unsafeHasActions, saveOk:saveOk, dismissOk:dismissOk, laterOk:laterOk, overlayOk:overlayOk, unsafeHasActions:unsafeHasActions, hrefStayed:location.href===before};
})())
""".strip()
        action_probe_result = _checked_agent_browser(command, ["eval", action_probe_expr], timeout, "action state probe eval")
        metrics_expr = (
            "JSON.stringify({"
            "innerWidth: window.innerWidth,"
            "innerHeight: window.innerHeight,"
            "scrollWidth: Math.max(document.documentElement.scrollWidth || 0, document.body ? document.body.scrollWidth || 0 : 0),"
            "bodyScrollWidth: document.body ? document.body.scrollWidth : 0,"
            "mobilebarVisible: !!document.querySelector('[data-mobilebar], .mobilebar, .mobile-bar, .mobile-nav, [data-testid=mobilebar]')"
            "})"
        )
        metrics_result = _checked_agent_browser(command, ["eval", metrics_expr], timeout, "layout metrics eval")
        _checked_agent_browser(command, ["screenshot", str(screenshot)], timeout, "screenshot")
        console_result = _checked_agent_browser(command, ["console"], timeout, "console")
        errors_result = _checked_agent_browser(command, ["errors"], timeout, "errors")
    finally:
        _run_agent_browser(command, ["close", "--all"], max(5, min(timeout, 10)))

    parsed_dom = _parse_eval_json(dom_result.stdout)
    dom = parsed_dom if isinstance(parsed_dom, str) else dom_result.stdout.strip()
    parsed_metrics = _parse_eval_json(metrics_result.stdout)
    parsed_action_probe = _parse_eval_json(action_probe_result.stdout)
    layout_metrics = cast(dict[str, object], parsed_metrics) if isinstance(parsed_metrics, dict) else {}
    action_state_probe = cast(dict[str, object], parsed_action_probe) if isinstance(parsed_action_probe, dict) else {}
    inner_width_value = layout_metrics.get("innerWidth", viewport_width)
    scroll_width_value = layout_metrics.get("scrollWidth", inner_width_value)
    try:
        horizontal_overflow = float(str(scroll_width_value)) > float(str(inner_width_value)) + 1
    except (TypeError, ValueError):
        horizontal_overflow = False
    if not screenshot.exists():
        raise RuntimeError("Browser rendered DOM but did not create the expected screenshot artifact")
    return BrowserResult(
        url=url,
        dom=dom,
        screenshot=screenshot,
        browser_path=_browser_identity(command),
        console_output=_sanitize_diagnostic_output(console_result.stdout.strip()),
        errors_output=_sanitize_diagnostic_output(errors_result.stdout.strip()),
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        layout_metrics=layout_metrics,
        horizontal_overflow=horizontal_overflow,
        action_state_probe=action_state_probe,
    )


def _section_index(dom: str, text: str) -> int:
    return dom.casefold().find(text.casefold())


def _output_has_messages(output: str) -> bool:
    stripped = output.strip()
    if not stripped:
        return False
    return not re.fullmatch(r"(?is)\s*(?:no\s+(?:console\s+)?messages|no\s+(?:page\s+)?errors|0\s+(?:messages|errors))\s*", stripped)


def _console_has_meaningful_error(output: str) -> bool:
    return _output_has_messages(output) and bool(re.search(r"\b(error|exception)\b", output, re.I))


def _common_browser_failures(
    dom: str,
    *,
    horizontal_overflow: bool = False,
    console_output: str = "",
    errors_output: str = "",
) -> tuple[list[str], bool]:
    failures: list[str] = []
    if horizontal_overflow:
        failures.append("Horizontal overflow detected at mobile viewport")
    if _console_has_meaningful_error(console_output):
        failures.append("Browser console contains error/exception output")
    if _output_has_messages(errors_output):
        failures.append("Browser page errors were reported")
    if "Sign in to Acta" in dom or "Acta access token" in dom:
        failures.append("Acta sign-in wall rendered; pass a local --html artifact or validate with authenticated browser storage")
        return failures, True
    return failures, False


def _validate_feed_contract(
    dom: str,
    *,
    horizontal_overflow: bool = False,
    console_output: str = "",
    errors_output: str = "",
    action_state_probe: Mapping[str, object] | None = None,
) -> list[str]:
    failures, auth_wall = _common_browser_failures(
        dom,
        horizontal_overflow=horizontal_overflow,
        console_output=console_output,
        errors_output=errors_output,
    )
    if auth_wall:
        return failures
    output_streams = _section_index(dom, "Output Streams")
    daily = _section_index(dom, "Daily life feed")
    dev = _section_index(dom, "Development sprint cycles")
    if output_streams < 0:
        failures.append("Output Streams heading is missing")
    if daily < 0:
        failures.append("Daily life feed section is missing")
    if dev < 0:
        failures.append("Development sprint cycles section is missing")
    if daily >= 0 and dev >= 0 and daily > dev:
        failures.append("Daily life feed must render before Development sprint cycles")
    daily_rows = DAILY_ROW_RE.findall(dom)
    dev_rows = DEV_ROW_RE.findall(dom)
    if not daily_rows:
        failures.append("No browser-rendered daily feed rows found")
    if not dev_rows:
        failures.append("No browser-rendered development feed rows found")
    daily_text = "\n".join(daily_rows)
    dev_text = "\n".join(dev_rows)
    if DEV_JOB_RE.search(daily_text):
        failures.append("Development sprint output is commingled into Daily life feed")
    if not DEV_JOB_RE.search(dev_text):
        failures.append("Development sprint feed has no recognized sprint/QA/security rows")
    if "lane-chip" not in dom:
        failures.append("Lane chips are missing from rendered rows")
    if action_state_probe and not bool(action_state_probe.get("skipped")) and not bool(action_state_probe.get("ok")):
        reason = str(action_state_probe.get("reason") or action_state_probe)
        failures.append(f"Signed row action-state browser probe failed: {reason}")
    return failures


def _validate_jobs_contract(
    dom: str,
    *,
    horizontal_overflow: bool = False,
    console_output: str = "",
    errors_output: str = "",
) -> list[str]:
    failures, auth_wall = _common_browser_failures(
        dom,
        horizontal_overflow=horizontal_overflow,
        console_output=console_output,
        errors_output=errors_output,
    )
    if auth_wall:
        return failures

    if not re.search(r"\bJobs\b|Source\s+Runs", dom, re.I):
        failures.append("Jobs/source-runs identity is missing")

    job_rows = _extract_text_by_class(dom, "job-row")
    if not job_rows:
        failures.append("No browser-rendered job rows found")
        return failures

    for index, row_text in enumerate(job_rows, start=1):
        if not CONFIDENCE_CHIP_RE.search(row_text):
            failures.append(f"Job row {index} is missing visible confidence chips (CONF HIGH/MED/LOW-GAP)")
        if not re.search(r"\bLAST\s+RUN\b", row_text, re.I):
            failures.append(f"Job row {index} is missing LAST RUN freshness copy")
        if not re.search(r"\bSCHEDULE\b", row_text, re.I):
            failures.append(f"Job row {index} is missing SCHEDULE copy")
        if not re.search(r"\b(?:OPEN|SIGNED|NO\s+PAGE)\b", row_text, re.I):
            failures.append(f"Job row {index} is missing action/status copy (OPEN/SIGNED or NO PAGE)")
        if not re.search(r"\bSOURCE\b|\bjob_id\b", row_text, re.I):
            failures.append(f"Job row {index} is missing source/provenance copy (SOURCE/job_id)")
    return failures


def _validate_outputs_contract(
    dom: str,
    *,
    horizontal_overflow: bool = False,
    console_output: str = "",
    errors_output: str = "",
) -> list[str]:
    failures, auth_wall = _common_browser_failures(
        dom,
        horizontal_overflow=horizontal_overflow,
        console_output=console_output,
        errors_output=errors_output,
    )
    if auth_wall:
        return failures

    if OUTPUT_LEAK_RE.search(dom):
        failures.append("Outputs DOM contains raw prompt/tool/path leakage")

    if not re.search(r"\bOutputs\b|Signed\s+source\s+objects|Persistent\s+catalog", dom, re.I):
        failures.append("Outputs shelf identity is missing")

    output_rows = _extract_text_by_class(dom, "output-row")
    output_row_html = _extract_html_by_class(dom, "output-row")
    if not output_rows:
        failures.append("No browser-rendered output rows found")
        return failures

    for index, row_text in enumerate(output_rows, start=1):
        row_html = output_row_html[index - 1] if index <= len(output_row_html) else ""
        if not OUTPUT_CONFIDENCE_RE.search(row_text):
            failures.append(f"Output row {index} is missing visible confidence/status copy (CONF HIGH/MED/LOW-GAP or CATALOG)")
        has_source = bool(re.search(r"\bSOURCE\b", row_text, re.I))
        has_job_id = bool(re.search(r"\bJOB\b", row_text, re.I) and re.search(r"\bID\b", row_text, re.I))
        if not (has_source or has_job_id):
            failures.append(f"Output row {index} is missing source/provenance copy (SOURCE or JOB/ID)")
        if not re.search(
            r"\bSCHEDULE\b|\bbrief\b|\bsystem\b|\bOUTPUT\b|\bPINNED\b|\b\d+\s*(?:m|min|h|hr|d|day)s?\s+ago\b|\bjust\s+now\b|\bage\b|\bcatalog\b",
            row_text,
            re.I,
        ):
            failures.append(f"Output row {index} is missing freshness/category metadata")
        if not re.search(r"\b(?:SIGNED|OPEN|No\s+signed\s+link|No\s+public\s+link)\b", row_text, re.I):
            failures.append(f"Output row {index} is missing action/status copy (SIGNED/OPEN or no-link status)")

        actionable = re.search(r"\b(?:SIGNED|OPEN)\b", row_text, re.I) and not re.search(
            r"\bNo\s+(?:signed|public)\s+link\b", row_text, re.I
        )
        if actionable:
            if not _has_clickable_open_affordance(row_html):
                failures.append(f"Output row {index} is missing clickable artifact-open affordance (artifact-open href/data-open-url)")
            row_without_toggle_copy = re.sub(r"\bMark\s+(?:read|unread)\b", " ", row_text, flags=re.I)
            if not re.search(r"\b(?:READ|UNREAD)\b", row_without_toggle_copy, re.I):
                failures.append(f"Output row {index} is missing read/unread state")
            if not re.search(r"\bMark\s+(?:read|unread)\b", row_text, re.I):
                failures.append(f"Output row {index} is missing Mark read/Mark unread toggle")
    return failures


def _validate_output_artifact_contract(
    dom: str,
    *,
    horizontal_overflow: bool = False,
    console_output: str = "",
    errors_output: str = "",
) -> list[str]:
    failures, auth_wall = _common_browser_failures(
        dom,
        horizontal_overflow=horizontal_overflow,
        console_output=console_output,
        errors_output=errors_output,
    )
    if auth_wall:
        return failures
    if OUTPUT_LEAK_RE.search(dom):
        failures.append("Opened output artifact contains raw prompt/tool/path leakage")
    return failures



def _validate_archive_contract(
    dom: str,
    *,
    horizontal_overflow: bool = False,
    console_output: str = "",
    errors_output: str = "",
) -> list[str]:
    failures, auth_wall = _common_browser_failures(
        dom,
        horizontal_overflow=horizontal_overflow,
        console_output=console_output,
        errors_output=errors_output,
    )
    if auth_wall:
        return failures

    if not re.search(r"\bArchive\b|Previous\s+days", dom, re.I):
        failures.append("Archive identity is missing")

    archive_cards = _extract_text_by_class(dom, "archive-card")
    archive_card_html = _extract_html_by_class(dom, "archive-card")
    if not archive_cards:
        failures.append("No browser-rendered archive cards found")
        return failures

    has_source_signal_card = False
    for index, card_text in enumerate(archive_cards, start=1):
        card_html = archive_card_html[index - 1] if index <= len(archive_card_html) else ""
        hrefs = _archive_card_hrefs(card_html)
        safe_href = any(_is_safe_archive_href(href) for href in hrefs)
        for href in hrefs:
            if not _is_safe_archive_href(href):
                display_href = "<missing>" if href is None else (href if href else "<empty>")
                failures.append(f"Archive card {index} has unsafe href: {display_href}")
        if not safe_href:
            failures.append(f"Archive card {index} is missing safe archive href")
        has_counts = _has_numeric_source_signal_counts(card_text)
        if safe_href and has_counts:
            has_source_signal_card = True
    if not has_source_signal_card:
        failures.append("No archive card includes both safe archive href and numeric Visible/Silent/Missing counts")
    return failures


def _scenario_metadata(scenario: str) -> dict[str, str]:
    if scenario == "archive":
        return {
            "persona": "mobile Acta operator reviewing previous days",
            "scenario": "Validate Acta Archive cards at mobile width before tapping into prior-day snapshots",
        }
    if scenario == "outputs":
        return {
            "persona": "mobile Acta operator inspecting Outputs shelf artifacts",
            "scenario": "Validate Acta Outputs shelf artifact rows in a narrow mobile browser viewport",
        }
    if scenario == "jobs":
        return {
            "persona": "mobile Acta operator inspecting Jobs/source-runs freshness and confidence",
            "scenario": "Validate Acta Jobs/source-runs rows in a narrow mobile browser viewport",
        }
    return {
        "persona": "mobile Acta operator checking dashboard feed lanes",
        "scenario": "Validate Acta dashboard feed lanes in a narrow mobile browser viewport",
    }


def run(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    url = _target_url(args)
    scenario = getattr(args, "scenario", "feed")
    output_artifact_result: BrowserResult | None = None
    output_artifact_url: str | None = None
    output_artifact_failures: list[str] = []
    try:
        result = _run_chrome(url, artifact_dir, args.timeout, args.viewport_width, args.viewport_height)
        if scenario == "archive":
            validate = _validate_archive_contract
        elif scenario == "outputs":
            validate = _validate_outputs_contract
        elif scenario == "jobs":
            validate = _validate_jobs_contract
        else:
            validate = _validate_feed_contract
        failures = validate(
            result.dom,
            horizontal_overflow=result.horizontal_overflow,
            console_output=result.console_output,
            errors_output=result.errors_output,
            **({"action_state_probe": result.action_state_probe or {}} if scenario == "feed" else {}),
        )
        if scenario == "outputs":
            output_artifact_url = _first_output_artifact_url(result.dom, result.url)
            if output_artifact_url:
                output_artifact_result = _run_chrome(
                    output_artifact_url,
                    artifact_dir / "output-artifact",
                    args.timeout,
                    args.viewport_width,
                    args.viewport_height,
                )
                output_artifact_failures = _validate_output_artifact_contract(
                    output_artifact_result.dom,
                    horizontal_overflow=output_artifact_result.horizontal_overflow,
                    console_output=output_artifact_result.console_output,
                    errors_output=output_artifact_result.errors_output,
                )
                failures.extend(output_artifact_failures)
            elif _extract_text_by_class(result.dom, "output-row"):
                failures.append("No actionable Outputs artifact target found/opened")
    except Exception as exc:  # noqa: BLE001 - CLI harness should print actionable failure text.
        print("FAIL Acta browser UAT")
        print(str(exc))
        return 1

    report = {
        **_scenario_metadata(scenario),
        "scenario_key": scenario,
        "url": _report_url(result.url),
        "browser": str(result.browser_path),
        "screenshot": str(result.screenshot),
        "viewport": {"width": result.viewport_width, "height": result.viewport_height},
        "console_output": result.console_output,
        "errors_output": result.errors_output,
        "layout_metrics": result.layout_metrics or {},
        "horizontal_overflow": result.horizontal_overflow,
        "action_state_probe": (result.action_state_probe or {}) if scenario == "feed" else {},
        "failures": failures,
    }
    if scenario == "archive":
        report["archive_cards"] = len(_extract_text_by_class(result.dom, "archive-card"))
    elif scenario == "outputs":
        report["output_rows"] = len(_extract_text_by_class(result.dom, "output-row"))
        report["opened_output_artifact_url"] = _report_url(output_artifact_url) if output_artifact_url else ""
        report["output_artifact_screenshot"] = str(output_artifact_result.screenshot) if output_artifact_result else ""
        report["output_artifact_horizontal_overflow"] = (
            output_artifact_result.horizontal_overflow if output_artifact_result else False
        )
        report["output_artifact_failures"] = output_artifact_failures
    elif scenario == "jobs":
        report["job_rows"] = len(_extract_text_by_class(result.dom, "job-row"))
    else:
        report["daily_rows"] = len(DAILY_ROW_RE.findall(result.dom))
        report["dev_rows"] = len(DEV_ROW_RE.findall(result.dom))
    report_path = artifact_dir / "acta-uat-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        print("FAIL Acta browser UAT")
        for failure in failures:
            print(f"- {failure}")
        print(f"Screenshot: {result.screenshot}")
        print(f"Report: {report_path}")
        return 1

    print("PASS Acta browser UAT")
    if scenario == "archive":
        print(f"Archive cards: {report['archive_cards']}")
    elif scenario == "outputs":
        print(f"Output rows: {report['output_rows']}")
        if report.get("opened_output_artifact_url"):
            print(f"Opened output artifact: {report['opened_output_artifact_url']}")
    elif scenario == "jobs":
        print(f"Job rows: {report['job_rows']}")
    else:
        print(f"Daily rows: {report['daily_rows']}")
        print(f"Development rows: {report['dev_rows']}")
    print(f"Screenshot: {result.screenshot}")
    print(f"Report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-browser UAT harness for Acta browser scenarios")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--html", help="Path to a generated Acta dashboard HTML file")
    target.add_argument("--url", help="Published Acta URL to validate")
    parser.add_argument("--artifact-dir", default=".hermes/uat/acta", help="Directory for screenshot and JSON report")
    parser.add_argument("--timeout", type=int, default=30, help="Chrome render timeout in seconds")
    parser.add_argument("--viewport-width", type=int, default=390, help="Browser viewport width for mobile UAT")
    parser.add_argument("--viewport-height", type=int, default=844, help="Browser viewport height for mobile UAT")
    parser.add_argument("--scenario", choices=("feed", "jobs", "outputs", "archive"), default="feed", help="Acta UAT scenario to validate")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
