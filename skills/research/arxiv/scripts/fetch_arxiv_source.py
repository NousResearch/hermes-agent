#!/usr/bin/env python3
"""Fetch an arXiv paper as LaTeX, readable HTML text, or PDF."""

from __future__ import annotations

import argparse
import gzip
import re
import tarfile
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2_000
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_TEXT_BYTES = 10 * 1024 * 1024
MAX_INCLUDE_DEPTH = 30
MAX_BIB_ENTRIES = 200
USER_AGENT = "Hermes-Agent-arXiv/1.0 (+https://github.com/NousResearch/hermes-agent)"

_MODERN_ID_RE = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
_LEGACY_ID_RE = re.compile(r"[a-z][a-z0-9.-]*/\d{7}(?:v\d+)?", re.IGNORECASE)
_INCLUDE_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]+)\}")
_BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\s*\{([^{}]+)\}")
_ADD_BIB_RE = re.compile(r"\\addbibresource(?:\[[^]]*\])?\s*\{([^{}]+)\}")
_TEXT_EXTENSIONS = {".tex", ".bib", ".bbl"}
_OUTPUT_EXTENSIONS = {".tex", ".txt", ".pdf"}


class FetchError(RuntimeError):
    """Raised when a remote arXiv representation cannot be fetched."""


class SourceFormatError(RuntimeError):
    """Raised when an arXiv source response cannot be safely rendered."""


@dataclass(frozen=True)
class Download:
    data: bytes
    content_type: str
    final_url: str


@dataclass(frozen=True)
class FetchResult:
    kind: str
    path: Path
    source_url: str


Fetcher = Callable[[str], Download]


def normalize_arxiv_id(value: str) -> str:
    """Return a validated modern or legacy arXiv identifier."""
    candidate = value.strip()
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate[6:]

    parsed = urlsplit(candidate)
    if parsed.scheme or parsed.netloc:
        host = (parsed.hostname or "").lower()
        if host != "arxiv.org" and not host.endswith(".arxiv.org"):
            raise ValueError("expected an arxiv.org URL")
        path = parsed.path.lstrip("/")
        for prefix in ("abs/", "pdf/", "src/", "html/"):
            if path.startswith(prefix):
                candidate = path[len(prefix) :]
                break
        else:
            raise ValueError("arXiv URL must use /abs, /pdf, /src, or /html")

    candidate = candidate.removesuffix(".pdf").strip("/")
    if not (_MODERN_ID_RE.fullmatch(candidate) or _LEGACY_ID_RE.fullmatch(candidate)):
        raise ValueError(f"invalid arXiv identifier: {value!r}")
    return candidate


def _download(url: str, timeout: float = 30.0) -> Download:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
            final_url = response.geturl()
            host = (urlsplit(final_url).hostname or "").lower()
            if host != "arxiv.org" and not host.endswith(".arxiv.org"):
                raise FetchError(f"refused redirect outside arxiv.org: {final_url}")

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_DOWNLOAD_BYTES:
                        raise FetchError("download exceeds the 50 MiB limit")
                except ValueError:
                    pass

            data = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise FetchError("download exceeds the 50 MiB limit")
            content_type = response.headers.get_content_type()
    except FetchError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise FetchError(str(exc)) from exc

    return Download(data=data, content_type=content_type, final_url=final_url)


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if (
        not parts
        or path.is_absolute()
        or ".." in parts
        or parts[0].endswith(":")
        or "\x00" in normalized
    ):
        raise SourceFormatError(f"unsafe archive member path: {name!r}")
    return PurePosixPath(*parts)


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _read_source_files(data: bytes) -> dict[PurePosixPath, str]:
    """Read selected text members without extracting the archive to disk."""
    try:
        archive = tarfile.open(fileobj=BytesIO(data), mode="r:*")
    except (tarfile.ReadError, EOFError):
        archive = None

    if archive is not None:
        files: dict[PurePosixPath, str] = {}
        seen_paths: set[PurePosixPath] = set()
        with archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise SourceFormatError("source archive has too many members")

            total_size = 0
            for member in members:
                path = _safe_member_path(member.name)
                if path in seen_paths:
                    raise SourceFormatError(f"duplicate archive member: {path}")
                seen_paths.add(path)
                if (
                    member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise SourceFormatError(
                        f"source archive contains a special or linked member: {path}"
                    )
                if member.isdir():
                    continue
                if not member.isfile() or member.size < 0:
                    raise SourceFormatError(f"unsupported archive member: {path}")
                total_size += member.size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise SourceFormatError("expanded source archive exceeds 200 MiB")
                if path.suffix.lower() not in _TEXT_EXTENSIONS:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SourceFormatError(f"could not read archive member: {path}")
                member_data = extracted.read(MAX_TEXT_BYTES + 1)
                if len(member_data) > MAX_TEXT_BYTES:
                    raise SourceFormatError(f"text member exceeds 10 MiB: {path}")
                files[path] = _decode_text(member_data)

        if not any(path.suffix.lower() == ".tex" for path in files):
            raise SourceFormatError("source archive contains no TeX files")
        return files

    if data.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=BytesIO(data)) as compressed:
                data = compressed.read(MAX_TEXT_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise SourceFormatError("invalid gzip source response") from exc
        if len(data) > MAX_TEXT_BYTES:
            raise SourceFormatError("expanded source exceeds 10 MiB")

    text = _decode_text(data)
    if "\\documentclass" not in text and "\\begin{document}" not in text:
        raise SourceFormatError("source response is not a TeX document")
    return {PurePosixPath("main.tex"): text}


def _main_document(files: dict[PurePosixPath, str]) -> PurePosixPath:
    candidates = [path for path in files if path.suffix.lower() == ".tex"]
    if not candidates:
        raise SourceFormatError("source contains no TeX document")

    preferred_names = {"main", "paper", "manuscript", "article", "ms"}

    def score(path: PurePosixPath) -> tuple[int, int, str]:
        text = files[path]
        points = 0
        points += 100 if "\\documentclass" in text else 0
        points += 40 if "\\begin{document}" in text else 0
        points += 20 if path.stem.lower() in preferred_names else 0
        points -= 50 if "supp" in path.stem.lower() else 0
        return points, -len(path.parts), str(path)

    return max(candidates, key=score)


def _split_tex_comment(line: str) -> tuple[str, str]:
    for index, char in enumerate(line):
        if char != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return line[:index], line[index:]
    return line, ""


def _resolve_reference(
    raw: str,
    suffix: str,
    current_dir: PurePosixPath,
    root_dir: PurePosixPath,
    files: dict[PurePosixPath, str],
) -> PurePosixPath | None:
    value = raw.strip().replace("\\", "/")
    if not value or value.startswith("/") or "#" in value or "\\" in value:
        return None
    if not PurePosixPath(value).suffix:
        value += suffix

    for base in (current_dir, root_dir, PurePosixPath(".")):
        parts: list[str] = []
        escaped = False
        for part in (*base.parts, *PurePosixPath(value).parts):
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    escaped = True
                    break
                parts.pop()
            else:
                parts.append(part)
        if escaped or not parts:
            continue
        candidate = PurePosixPath(*parts)
        if candidate in files:
            return candidate
    return None


def _expand_includes(main: PurePosixPath, files: dict[PurePosixPath, str]) -> str:
    root_dir = main.parent

    def expand(path: PurePosixPath, stack: tuple[PurePosixPath, ...]) -> str:
        if len(stack) >= MAX_INCLUDE_DEPTH:
            return f"\n% [include depth limit reached at {path}]\n"
        if path in stack:
            cycle = " -> ".join(str(item) for item in (*stack, path))
            return f"\n% [cyclic include omitted: {cycle}]\n"

        rendered: list[str] = []
        for line in files[path].splitlines(keepends=True):
            code, comment = _split_tex_comment(line)

            def replace(match: re.Match[str]) -> str:
                target = _resolve_reference(
                    match.group(1), ".tex", path.parent, root_dir, files
                )
                if target is None:
                    return match.group(0)
                included = expand(target, (*stack, path))
                return (
                    f"\n% --- begin included: {target} ---\n"
                    f"{included}"
                    f"\n% --- end included: {target} ---\n"
                )

            rendered.append(_INCLUDE_RE.sub(replace, code) + comment)

        result = "".join(rendered)
        if len(result.encode("utf-8")) > MAX_TEXT_BYTES:
            raise SourceFormatError("expanded TeX document exceeds 10 MiB")
        return result

    return expand(main, ())


def _bib_value(fields: str, name: str) -> str:
    match = re.search(rf"(?i)(?:^|,)\s*{re.escape(name)}\s*=\s*", fields)
    if not match:
        return ""
    index = match.end()
    if index >= len(fields):
        return ""

    opener = fields[index]
    if opener in '{"':
        closer = "}" if opener == "{" else '"'
        depth = 0
        cursor = index + 1
        while cursor < len(fields):
            char = fields[cursor]
            if opener == "{" and char == "{":
                depth += 1
            elif char == closer and (opener != "{" or depth == 0):
                value = fields[index + 1 : cursor]
                break
            elif opener == "{" and char == "}":
                depth -= 1
            cursor += 1
        else:
            value = fields[index + 1 :]
    else:
        value = fields[index:].split(",", 1)[0]

    value = re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()
    return value[:300]


def _abbreviate_bibliography(contents: list[str]) -> str:
    entries: list[str] = []
    entry_start = re.compile(r"@(\w+)\s*([({])")
    for content in contents:
        cursor = 0
        while len(entries) < MAX_BIB_ENTRIES:
            match = entry_start.search(content, cursor)
            if not match:
                break
            opener = match.group(2)
            closer = "}" if opener == "{" else ")"
            depth = 1
            index = match.end()
            while index < len(content) and depth:
                if content[index] == opener:
                    depth += 1
                elif content[index] == closer:
                    depth -= 1
                index += 1
            if depth:
                break
            body = content[match.end() : index - 1]
            key, separator, fields = body.partition(",")
            if separator and key.strip():
                details = [
                    value
                    for value in (
                        _bib_value(fields, "title"),
                        _bib_value(fields, "author"),
                        _bib_value(fields, "year"),
                    )
                    if value
                ]
                suffix = " | ".join(details)
                entries.append(f"% - {key.strip()}: {suffix}".rstrip(": "))
            cursor = index

    if not entries:
        return ""
    if len(entries) == MAX_BIB_ENTRIES:
        entries.append("% - [bibliography truncated]")
    return "\n\n% --- abbreviated bibliography ---\n" + "\n".join(entries) + "\n"


def render_latex_source(data: bytes) -> str:
    files = _read_source_files(data)
    main = _main_document(files)
    rendered = _expand_includes(main, files)

    bibliography_paths: list[PurePosixPath] = []
    references: list[str] = []
    for match in _BIBLIOGRAPHY_RE.finditer(rendered):
        references.extend(part.strip() for part in match.group(1).split(","))
    references.extend(
        match.group(1).strip() for match in _ADD_BIB_RE.finditer(rendered)
    )
    for reference in references:
        path = _resolve_reference(reference, ".bib", main.parent, main.parent, files)
        if path is not None and path not in bibliography_paths:
            bibliography_paths.append(path)
    if not bibliography_paths:
        available = [path for path in files if path.suffix.lower() == ".bib"]
        if len(available) == 1:
            bibliography_paths = available

    bibliography = _abbreviate_bibliography([
        files[path] for path in bibliography_paths
    ])
    return f"% arXiv main document: {main}\n{rendered}{bibliography}"


class _ReadableHTMLParser(HTMLParser):
    _IGNORED = {
        "aside",
        "footer",
        "form",
        "head",
        "header",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
    }
    _VOID = {
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
    _BLOCKS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.body_depth = 0
        self.primary_depth = 0
        self.ignored_depth = 0
        self.all_text: list[str] = []
        self.primary_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self.ignored_depth:
            if tag not in self._VOID:
                self.ignored_depth += 1
            return
        if tag in self._IGNORED:
            self.ignored_depth = 1
            return
        if tag == "body":
            self.body_depth += 1
        if tag in {"article", "main"}:
            self.primary_depth += 1
        if tag in self._BLOCKS:
            self._append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.ignored_depth and tag in self._BLOCKS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_depth:
            self.ignored_depth -= 1
            return
        if tag in self._BLOCKS:
            self._append("\n")
        if tag in {"article", "main"} and self.primary_depth:
            self.primary_depth -= 1
        if tag == "body" and self.body_depth:
            self.body_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and self.body_depth:
            self._append(data)

    def _append(self, value: str) -> None:
        if not self.body_depth:
            return
        self.all_text.append(value)
        if self.primary_depth:
            self.primary_text.append(value)

    def text(self) -> str:
        chunks = self.primary_text or self.all_text
        lines = [
            re.sub(r"\s+", " ", line).strip() for line in "".join(chunks).splitlines()
        ]
        return "\n".join(line for line in lines if line)


def render_html_text(data: bytes) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(_decode_text(data))
    text = parser.text()
    if not text:
        raise SourceFormatError("HTML response contains no readable paper text")
    return text + "\n"


def _output_path(output: str | Path | None, arxiv_id: str, suffix: str) -> Path:
    if output is None:
        safe_id = arxiv_id.replace("/", "-")
        return Path(f"arxiv-{safe_id}{suffix}")
    path = Path(output)
    if path.suffix.lower() in _OUTPUT_EXTENSIONS:
        return path.with_suffix(suffix)
    return Path(f"{path}{suffix}")


def fetch_paper(
    identifier: str,
    output: str | Path | None = None,
    fetcher: Fetcher = _download,
) -> FetchResult:
    arxiv_id = normalize_arxiv_id(identifier)
    attempts: list[str] = []

    source_url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        source = fetcher(source_url)
        text = render_latex_source(source.data)
        path = _output_path(output, arxiv_id, ".tex")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return FetchResult("latex", path, source.final_url)
    except (FetchError, SourceFormatError) as exc:
        attempts.append(f"source: {exc}")

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        html = fetcher(html_url)
        text = render_html_text(html.data)
        path = _output_path(output, arxiv_id, ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return FetchResult("html", path, html.final_url)
    except (FetchError, SourceFormatError) as exc:
        attempts.append(f"html: {exc}")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        pdf = fetcher(pdf_url)
        if not pdf.data.startswith(b"%PDF-"):
            raise SourceFormatError("PDF response does not have a PDF signature")
        path = _output_path(output, arxiv_id, ".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf.data)
        return FetchResult("pdf", path, pdf.final_url)
    except (FetchError, SourceFormatError) as exc:
        attempts.append(f"pdf: {exc}")

    raise FetchError("all arXiv representations failed (" + "; ".join(attempts) + ")")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv LaTeX source, falling back to HTML and then PDF."
    )
    parser.add_argument("arxiv_id", help="arXiv ID or arxiv.org paper URL")
    parser.add_argument(
        "--output",
        help="output path stem (the selected .tex, .txt, or .pdf suffix is added)",
    )
    args = parser.parse_args(argv)

    try:
        result = fetch_paper(args.arxiv_id, args.output)
    except (FetchError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Fetched {result.kind} from {result.source_url} -> {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
