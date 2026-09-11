#!/usr/bin/env python3
"""Render the release download tables into <!-- HERMES_BUILDS_TABLE -->, and
the same rows as standalone pages in the bucket.

Runs as the LAST job of desktop-bundled-release.yml, after every matrix
leg has uploaded, and edits the GitHub release body in place. The tables
are built from the bucket's ACTUAL object names (scripts/releases/r2.py
list --prefix releases/tag/<tag>/), filtered to the tag's exact version —
a missing artifact shows up as a missing row, never a dead link. The
GitHub release carries the notes only; the binaries live in the R2 bucket
under releases/tag/<tag>/, and the download links point at the R2 public
URL (CLOUDFLARE_R2_PUBLIC_URL / --r2-base-url).

Every run also publishes the same rows as a tiny HTML page in the bucket,
so a build can be read straight from the download origin:

  releases/<channel>/index.html      replaced by each release; the channel
                                    page holds the latest stable or canary
                                    builds for every variant
  releases/commit/<sha>/index.html  commit mode: every expected binary of
                                    one commit build, built or not

Tables: Hermes Desktop (bundled) and Hermes Light, one row per (OS,
arch). Feed manifests (latest*/light*/canary*.yml), blockmaps and mac .zip
(an electron-updater delta target, not a user download) stay out of the
tables on purpose; they still live in the bucket for the updater to
consume.

With --pending-run-url, renders a "builds in progress" link to the
workflow run instead of the tables. The builds-pending job runs this
mode as the first job of the run, so the draft body points at the live
run while the matrix builds. The link block keeps the marker wrapper,
so the final render replaces it.

Usage: render-builds-table.py --tag vX.Y.Z [--repo owner/repo] [--r2-base-url URL] [--dry-run]
Idempotent: re-running replaces the previously rendered block (the
marker is kept as an HTML comment wrapper around the tables).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Direct-script invocation starts with scripts/, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.releases import handoff, r2, semver  # noqa: E402

MARKER = "<!-- HERMES_BUILDS_TABLE -->"
END_MARKER = "<!-- /HERMES_BUILDS_TABLE -->"

# Asset name shapes (electron-builder artifactName in
# apps/desktop/electron-builder.config.cjs):
#   Hermes-0.28.0-mac-arm64.dmg        (bundled)
#   HermesBundled-0.28.0-win-x64.msix  (bundled)
_ASSET_RE = re.compile(
    r"^(?P<app>HermesBundled|HermesLight)-(?P<version>[^-]+(?:-canary\.\d{8}(?:\d{6})?)?)"
    r"-(?P<os>mac|win|linux)-(?P<arch>x64|arm64)\.(?P<ext>dmg|msix|AppImage)$"
)

_OS_LABEL = {"mac": "macOS", "win": "Windows", "linux": "Linux (AppImage)"}
_ARCH_LABEL = {
    ("mac", "arm64"): "Apple Silicon (M-series)",
    ("mac", "x64"): "Intel",
    ("win", "x64"): "x86 (64-bit)",
    ("win", "arm64"): "ARM (arm64 / aarch64)",
    ("linux", "x64"): "x86 (64-bit)",
    ("linux", "arm64"): "arm64",
}
_KIND_LABEL = {"dmg": "DMG", "msix": "MSIX", "AppImage": "AppImage"}
_ROW_ORDER = [("mac", "arm64"), ("mac", "x64"), ("win", "x64"), ("win", "arm64"),
              ("linux", "x64"), ("linux", "arm64")]


def parse_assets(names: list[str]) -> dict[str, dict[tuple[str, str], tuple[str, str]]]:
    """{app: {(os, arch): (full_key, ext)}} for table-shaped assets only.

    Object keys are releases/tag/<tag>/<filename>; the shape match runs on
    the basename, but the stored name keeps the full key so the download
    link points at the object's real location.
    """
    out: dict[str, dict[tuple[str, str], tuple[str, str]]] = {"HermesBundled": {}, "HermesLight": {}}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        m = _ASSET_RE.match(base)
        if m:
            out[m.group("app")][(m.group("os"), m.group("arch"))] = (name, m.group("ext"))
    return out


def table_rows(assets_by_app: dict) -> list[tuple[str, list[tuple[str, str, str, str]]]]:
    """[(section title, [(os label, arch label, kind label, object key), ...]), ...].

    The ONE row set every sink renders (release-body markdown, R2 page);
    a row exists only for an object that is actually in the bucket.
    """
    sections: list[tuple[str, list[tuple[str, str, str, str]]]] = []
    for app, title in (("HermesBundled", "Hermes Desktop"), ("HermesLight", "Hermes Light (remote-only client)")):
        rows = []
        for key in _ROW_ORDER:
            entry = assets_by_app.get(app, {}).get(key)
            if not entry:
                continue
            name, ext = entry
            rows.append((_OS_LABEL[key[0]], _ARCH_LABEL[key], _KIND_LABEL[ext], name))
        if rows:
            sections.append((title, rows))
    return sections


def render_tables(assets_by_app: dict, base_url: str) -> str:
    """The replacement block: marker + tables + end marker."""
    sections = []
    for title, rows in table_rows(assets_by_app):
        lines = [
            f"| {os_name} | {arch} | [{kind}]({r2.public_url_for(base_url, name)}) |"
            for os_name, arch, kind, name in rows
        ]
        sections.append(
            f"### {title}\n\n| OS | Architecture | Download |\n|---|---|---|\n" + "\n".join(lines)
        )
    if not sections:
        return ""
    return MARKER + "\n## Downloads\n\n" + "\n\n".join(sections) + "\n" + END_MARKER


def render_pending(run_url: str) -> str:
    """The placeholder block: a link to the run, in the same marker wrapper."""
    return (
        MARKER
        + f"\n> 🚧 [Builds in progress]({run_url}) — the download links"
        + " appear here when the build matrix finishes.\n"
        + END_MARKER
    )


def filter_names_for_version(names: list[str], version: str) -> list[str]:
    """Table-shaped names whose embedded version equals `version` (exact, not prefix).

    Matches on the basename (keys carry the releases/tag/<tag>/ prefix) and
    returns the full keys.
    """
    out = []
    for name in names:
        base = name.rsplit("/", 1)[-1]
        m = _ASSET_RE.match(base)
        if m and m.group("version") == version:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Commit-build summary: every expected binary of a commit run, built or not.
# ---------------------------------------------------------------------------

# A row needs a unique receipt-listed artifact and its uploaded object.
_COMMIT_EXPECTED = [
    ("Windows x64 (MSIX)", "win32-x64",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-win-x64\.msix$"),
    ("Windows ARM64 (MSIX)", "win32-arm64",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-win-arm64\.msix$"),
    ("Windows universal bundle (MSIXBUNDLE)", "windows-universal",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-win\.msixbundle$"),
    ("macOS Apple Silicon (DMG)", "darwin-arm64",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-mac-arm64\.dmg$"),
    ("macOS Intel (DMG)", "darwin-x64",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-mac-x64\.dmg$"),
    ("macOS Apple Silicon (ZIP)", "darwin-arm64",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-mac-arm64\.zip$"),
    ("macOS Intel (ZIP)", "darwin-x64",
     r"^HermesBundled-[^-]+(?:-canary\.\d+)?-mac-x64\.zip$"),
    ("Termux aarch64 (.deb)", "termux", r"^.*\.deb$"),
]

# Linux release legs are disabled; they still get a row so a reader can see
# they were never expected to publish.
_COMMIT_DISABLED = ["Linux x64 (AppImage)", "Linux ARM64 (AppImage)"]

COMMIT_RECEIPT_NAMES = sorted({leg for _label, leg, _pattern in _COMMIT_EXPECTED})
_COMMIT_JOBS = {
    "win32-x64": "build-win32", "win32-arm64": "build-win32",
    "darwin-x64": "build-darwin", "darwin-arm64": "build-darwin",
    "windows-universal": "publish-win32-updater", "termux": "termux-deb",
}


def commit_expected_rows(names: list[str],
                         receipts: dict[str, dict | None]) -> list[dict]:
    """Classify artifacts from validated receipts without trusting orphan objects."""
    objects = set(names)
    rows: list[dict] = []
    for label, leg, pattern in _COMMIT_EXPECTED:
        receipt = receipts.get(leg)
        listed = [r2.commit_key_for(receipt["commit"], row["path"])
                  for row in receipt["files"]
                  if re.fullmatch(pattern, row["path"].rsplit("/", 1)[-1])] if receipt else []
        if receipt is None:
            state, key = "receipt-missing", None
        elif not listed:
            state, key = "receipt-omits", None
        elif len(listed) > 1:
            state, key = "ambiguous", None
        elif listed[0] not in objects:
            state, key = "object-missing", None
        else:
            state, key = "built", listed[0]
        rows.append({"label": label, "leg": leg, "key": key, "state": state})
    return rows


def _validated_commit_inputs(commit: str, receipts: dict[str, dict | None]) -> None:
    """The summary and the commit page must never disagree about what was
    built, so both validate the SHA and every receipt the same way."""
    r2.commit_prefix_for(commit)
    for leg, receipt in receipts.items():
        if receipt is not None:
            handoff.validate_commit_receipt(receipt, commit, leg)


def commit_entries(commit: str, names: list[str], base_url: str,
                   receipts: dict[str, dict | None],
                   failed_legs: list[str] | None = None,
                   ) -> list[tuple[str, str, str | None, str | None]]:
    """[(label, status, download URL or None, object basename or None)] — the
    ONE row set the step summary and the commit page render. A link exists
    only for a receipt-listed object that is really in the bucket."""
    failed = set(failed_legs or [])
    entries: list[tuple[str, str, str | None, str | None]] = []
    for row in commit_expected_rows(names, receipts):
        label, leg, key, state = row["label"], row["leg"], row["key"], row["state"]
        if state == "built":
            entries.append((label, "✅ Built", r2.public_url_for(base_url, key), key.rsplit("/", 1)[-1]))
        elif state == "receipt-missing":
            related = failed.intersection({leg, _COMMIT_JOBS[leg]})
            blame = f"failed: {', '.join(sorted(related))}" if related \
                else "leg incomplete or upload interrupted"
            entries.append((label, f"❌ Not built ({blame})", None, None))
        elif state == "object-missing":
            entries.append((label, "❌ Not built (receipt present but object missing)", None, None))
        elif state == "receipt-omits":
            entries.append((label, "❌ Not built (artifact absent from receipt)", None, None))
        else:
            entries.append((label, "❌ Not built (ambiguous: multiple objects match)", None, None))
    for label in _COMMIT_DISABLED:
        entries.append((label, "❌ Not built (release leg disabled)", None, None))
    return entries


def render_commit_summary(names: list[str], base_url: str, commit: str,
                          receipts: dict[str, dict | None],
                          failed_legs: list[str] | None = None) -> str:
    """Render every expected product without reading or changing a release."""
    _validated_commit_inputs(commit, receipts)
    lines = [
        f"## Commit build `{commit[:12]}`",
        "",
        "| Binary | Status | Download |",
        "|---|---|---|",
    ]
    for label, status, url, basename in commit_entries(commit, names, base_url, receipts, failed_legs):
        cell = f"[{basename}]({url})" if url and basename else "—"
        lines.append(f"| {label} | {status} | {cell} |")
    return "\n".join([*lines, ""])


def read_commit_receipts(commit: str,
                         names: list[str] | None = None) -> dict[str, dict | None]:
    """Missing receipts describe incomplete legs; corrupt receipts raise."""
    out: dict[str, dict | None] = {}
    for name in (names or COMMIT_RECEIPT_NAMES):
        try:
            out[name] = handoff.read_commit_receipt(commit, name)
        except handoff.MissingReceipt:
            out[name] = None
    return out


def failed_legs_from_release_needs(release_needs_json: str | None) -> list[str]:
    """Read failure labels from the optional workflow result summary."""
    if not release_needs_json:
        return []
    try:
        needs = json.loads(release_needs_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(needs, dict):
        return []
    return sorted(name for name, info in needs.items()
                  if isinstance(info, dict) and info.get("result") not in ("success", "skipped"))


# ---------------------------------------------------------------------------
# Bucket pages: the same rows as the tables, served from the download origin
# ---------------------------------------------------------------------------

_PAGE_STYLE = (
    "body{font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:2rem auto;"
    "max-width:54rem;padding:0 1rem;color:#1a1a1a;background:#fff}"
    "h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:1.75rem}"
    "p{color:#444}table{border-collapse:collapse;width:100%}"
    "th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid #dcdcdc}"
    "th{font-weight:600}a{color:#0a58ca}code{font-size:.95em}"
)

# The record a channel page keeps of the release it describes; the write
# guard reads it back so an older tag re-run never regresses the page.
_BUILD_META_RE = re.compile(r'<meta name="hermes-build" content="([^"]*)"')


def _page(title: str, build: str, body: list[str]) -> str:
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="hermes-build" content="{html.escape(build, quote=True)}">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> list[str]:
    head = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    body = ["<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows]
    return [f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>", *body, "</tbody>\n</table>"]


def _link(url: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}">'


def render_page(tag: str, assets_by_app: dict, base_url: str) -> str:
    """The channel page: the release-body download table as HTML."""
    channel = r2.channel_for_tag(tag)
    body = [
        f"<h1>Hermes Desktop {channel} builds</h1>",
        f"<p>Release <code>{html.escape(tag)}</code>. Only objects this release "
        "actually staged in the bucket are listed.</p>",
    ]
    for title, rows in table_rows(assets_by_app):
        body.append(f"<h2>{html.escape(title)}</h2>")
        body.extend(_table(
            ("OS", "Architecture", "Download"),
            [[html.escape(os_name), html.escape(arch),
              f"{_link(r2.public_url_for(base_url, name))}{html.escape(kind)}</a>"]
             for os_name, arch, kind, name in rows],
        ))
    return _page(f"Hermes Desktop {channel} builds", tag, body)


def render_commit_page(commit: str, names: list[str], base_url: str,
                       receipts: dict[str, dict | None],
                       failed_legs: list[str] | None = None) -> str:
    """The commit-build page: every expected binary, built or not."""
    _validated_commit_inputs(commit, receipts)
    rows = []
    for label, status, url, basename in commit_entries(commit, names, base_url, receipts, failed_legs):
        cell = (f"{_link(url)}{html.escape(basename)}</a>" if url and basename else "—")
        rows.append([html.escape(label), html.escape(status), cell])
    body = [
        f"<h1>Hermes commit build <code>{html.escape(commit[:12])}</code></h1>",
        f"<p>Commit <code>{html.escape(commit)}</code>. Every expected binary is listed; "
        "a row without a link was not built.</p>",
        *_table(("Binary", "Status", "Download"), rows),
    ]
    return _page(f"Hermes commit build {commit[:12]}", commit, body)


def recorded_build(page: str | None) -> str | None:
    """The release tag or commit a page in the bucket describes."""
    match = _BUILD_META_RE.search(page or "")
    return (match.group(1) or None) if match else None


def supersedes(existing_page: str | None, tag: str) -> bool:
    """Whether `tag` may replace the channel page.

    Channel pages are mutable pointers: a re-run of an OLDER tag must not
    regress the page a newer release already published (the release body
    cannot regress — each tag owns its own release). Ordering uses the same
    semver authority the feeds do; an absent or unreadable record is written,
    because the new page is then the best available information.
    """
    recorded = recorded_build(existing_page)
    if not recorded:
        return True
    try:
        return semver.compare(recorded.lstrip("v"), tag.lstrip("v")) <= 0
    except ValueError:
        return True


def write_page(key: str, page: str, base_url: str) -> str:
    """PUT one page object at `key` (a FULL key, no tag archive) and return
    its public URL. Mutable: the page is replaced by each build of its own
    channel/commit."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                     suffix=".html", delete=False) as handle:
        handle.write(page)
        path = handle.name
    try:
        r2.put(tag="", key=key, file=path, key_is_full=True)
    finally:
        os.unlink(path)
    url = r2.public_url_for(base_url, key)
    print(f"✓ Page {url} ({len(page)} bytes)")
    return url


def existing_page(key: str) -> str | None:
    """The page currently at `key`, or None when nothing is published there."""
    try:
        creds, base, bucket = r2.credentials()
        return r2.get_object(creds, base, bucket, key, r2.amz_timestamp())
    except r2.R2RequestError as err:
        if err.status == 404:
            return None
        raise


def write_channel_page(tag: str, assets_by_app: dict, base_url: str) -> str | None:
    """Publish releases/<channel>/index.html for the tag's own channel."""
    key = r2.channel_page_key_for(r2.channel_for_tag(tag))
    if not supersedes(existing_page(key), tag):
        print(f"::warning::{key} already describes a newer release; leaving it unchanged")
        return None
    return write_page(key, render_page(tag, assets_by_app, base_url), base_url)


def r2_object_names_under(prefix: str) -> list[str]:
    return r2.list_objects(prefix=prefix)["keys"]


def r2_object_names(tag: str) -> list[str]:
    """Object keys in the R2 staging dir for `tag`, under releases/tag/<tag>/.

    A tag prefix and exact version match exclude neighboring releases.
    """
    keys = r2_object_names_under(f"releases/tag/{tag}/")
    return filter_names_for_version(keys, tag.lstrip("v"))


def splice(body: str, block: str) -> str:
    """Replace the marker (or a previously rendered block) with `block`."""
    if END_MARKER in body:
        pattern = re.compile(re.escape(MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
        return pattern.sub(lambda _m: block, body, count=1)
    return body.replace(MARKER, block, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=False, help="Release tag to render the release-body table for")
    parser.add_argument("--summary-commit", default=None,
                        help="Commit-only mode: render the FULL expected-binary matrix for "
                             "releases/commit/<sha>/ into --summary-out. Never touches a "
                             "GitHub release; every expected binary gets a row, built or not")
    parser.add_argument("--summary-out", default=None,
                        help="With --summary-commit: file the summary block is written to "
                             "(the workflow passes $GITHUB_STEP_SUMMARY)")
    parser.add_argument("--summary-failed-legs", default="",
                        help="With --summary-commit: comma-separated failed job names, "
                             "blamed on the Not built rows")
    parser.add_argument("--repo", default="NousResearch/hermes-agent")
    parser.add_argument("--r2-base-url", default=os.environ.get("CLOUDFLARE_R2_PUBLIC_URL"),
                        help="Public base URL of the R2 bucket (default: $CLOUDFLARE_R2_PUBLIC_URL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the spliced body instead of editing the release "
                             "(writes nothing, including the channel page)")
    parser.add_argument("--pending-run-url", default=None,
                        help="Render a 'builds in progress' link to this workflow run "
                             "instead of the tables")
    args = parser.parse_args()

    if args.summary_commit and (args.tag or args.pending_run_url):
        parser.error("--summary-commit cannot be combined with release-body arguments")

    if args.summary_commit:

        if not args.summary_out:
            print("::error::--summary-out is required with --summary-commit")
            return 1
        if not args.r2_base_url:
            print("::error::--r2-base-url (or CLOUDFLARE_R2_PUBLIC_URL) is required to render the summary")
            return 1
        commit = args.summary_commit
        try:
            prefix = r2.commit_prefix_for(commit)
        except ValueError as err:
            print(f"::error::{err}")
            return 1
        names = r2_object_names_under(prefix)
        receipts = read_commit_receipts(commit)
        failed_legs = (failed_legs_from_release_needs(os.environ.get("RELEASE_NEEDS"))
                       or [leg.strip() for leg in args.summary_failed_legs.split(",") if leg.strip()])
        block = render_commit_summary(names, args.r2_base_url, commit, receipts, failed_legs)
        with open(args.summary_out, "a", encoding="utf-8") as out:
            out.write(block)
        write_page(r2.commit_page_key_for(commit),
                   render_commit_page(commit, names, args.r2_base_url, receipts, failed_legs),
                   args.r2_base_url)
        built = sum(1 for row in commit_expected_rows(names, receipts) if row["state"] == "built")
        print(f"✓ Commit summary appended to {args.summary_out} ({built}/{len(_COMMIT_EXPECTED)} binaries built)")
        return 0

    if not args.tag:
        parser.error("--tag is required (or use --summary-commit for a commit build summary)")

    view = subprocess.run(
        ["gh", "release", "view", args.tag, "--repo", args.repo,
         "--json", "body"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if view.returncode != 0:
        print(f"::error::gh release view failed: {view.stderr.strip()}")
        return 1
    release = json.loads(view.stdout)
    body = release.get("body") or ""

    if args.pending_run_url:
        block = render_pending(args.pending_run_url)
        names: list[str] = []
    else:
        if not args.r2_base_url:
            print("::error::--r2-base-url (or CLOUDFLARE_R2_PUBLIC_URL) is required to render the tables")
            return 1
        names = r2_object_names(args.tag)
        assets = parse_assets(names)
        block = render_tables(assets, args.r2_base_url)
        if not block:
            print("::warning::no table-shaped assets for this tag in the bucket; leaving the body unchanged")
            return 0
        # The channel page is independent of the release body (and of the
        # marker), so it is published even if the body cannot be edited.
        if not args.dry_run:
            write_channel_page(args.tag, assets, args.r2_base_url)
    if MARKER not in body:
        print("::warning::release body has no HERMES_BUILDS_TABLE marker; leaving it unchanged")
        return 0

    new_body = splice(body, block)
    if args.dry_run:
        print(new_body)
        return 0

    edit = subprocess.run(
        ["gh", "release", "edit", args.tag, "--repo", args.repo,
         "--notes-file", "-"],
        input=new_body, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    if edit.returncode != 0:
        print(f"::error::gh release edit failed: {edit.stderr.strip()}")
        return 1
    what = "Builds-in-progress link" if args.pending_run_url else "Builds table"
    print(f"✓ {what} rendered into {args.tag} ({len(names)} assets scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
