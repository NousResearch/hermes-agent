#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Render the release download tables into <!-- HERMES_BUILDS_TABLE -->, and
the same rows as standalone pages in the bucket.

Runs after the build jobs of desktop-bundled-release.yml finish, including
failed legs, and edits the GitHub release body in place. The tables
are built from the bucket's ACTUAL object names (scripts/releases/r2.py
list --prefix releases/tag/<tag>/), filtered to the tag's exact version —
a missing artifact shows up as a missing row, never a dead link. The
GitHub release carries the notes only; the binaries live in the R2 bucket
under releases/tag/<tag>/, and the download links point at the R2 public
URL (CLOUDFLARE_R2_PUBLIC_URL / --r2-base-url).

Every run also publishes the same rows as a tiny HTML page in the bucket,
so a build can be read straight from the download origin:

  releases/tag/<tag>/index.html     every admitted tag, including failed builds
  releases/<channel>/index.html     latest successful stable or canary build;
                                    failed prerequisites leave it unchanged
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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

# Direct-script invocation starts with scripts/, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.releases import handoff, job_groups, r2, stable
from scripts.releases.desktop_targets import TARGETS as DESKTOP_TARGETS
from scripts.releases.downloads_page import (
    DEFAULT_REPO,
    _bundle_env_html,
    _link,
    _page,
    _table,
    commit_claim_rows,
    filter_names_for_version,
    parse_assets,
    render_page,
    smoke_html,
    smoke_markdown,
    supersedes,
    table_rows,
)

MARKER = "<!-- HERMES_BUILDS_TABLE -->"
END_MARKER = "<!-- /HERMES_BUILDS_TABLE -->"


def render_tables(assets_by_app: dict, base_url: str,
                  incomplete_jobs: list[str] | None = None,
                  run_url: str | None = None, *, smoke_results: dict | None = None) -> str:
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
        sections.append("No downloadable artifacts were staged for this build.")
    if incomplete_jobs:
        sections.insert(0, "> **Build incomplete.** Jobs not successful: "
                        + ", ".join(incomplete_jobs) + ". The channel page was not advanced.")
        if run_url:
            sections.append("### Build diagnostics\n\n| Job | Diagnostics |\n|---|---|\n"
                            + "\n".join(f"| {job} | [View build run]({run_url}) |"
                                        for job in incomplete_jobs))
    sections.append(smoke_markdown(smoke_results))
    return MARKER + "\n## Downloads\n\n" + "\n\n".join(sections) + "\n" + END_MARKER


def render_pending(run_url: str) -> str:
    """The placeholder block: a link to the run, in the same marker wrapper."""
    return (
        MARKER
        + f"\n> 🚧 [Builds in progress]({run_url}) — the download links"
        + " appear here when the build matrix finishes.\n"
        + END_MARKER
    )


# ---------------------------------------------------------------------------
# Commit-build summary: every expected binary of a commit run, built or not.
# ---------------------------------------------------------------------------

# A row needs a unique receipt-listed artifact and its uploaded object.
_COMMIT_EXPECTED = [
    ("Windows x64 (MSIX)", "win32-x64",
     r"^HermesBundled-[^-]+-win-x64\.msix$"),
    ("Windows ARM64 (MSIX)", "win32-arm64",
     r"^HermesBundled-[^-]+-win-arm64\.msix$"),
    ("Windows universal bundle (MSIXBUNDLE)", "windows-universal",
     r"^HermesBundled-[^-]+-win\.msixbundle$"),
    ("macOS Apple Silicon (DMG)", "darwin-arm64",
     r"^HermesBundled-[^-]+-mac-arm64\.dmg$"),
    ("macOS Apple Silicon (ZIP)", "darwin-arm64",
     r"^HermesBundled-[^-]+-mac-arm64\.zip$"),
    ("Termux aarch64 (.deb)", "termux", r"^.*\.deb$"),
]

# Commit builds skip the Linux legs; they still get a row so a reader can see
# they were never expected to publish.
_COMMIT_DISABLED = ["Linux x64 (AppImage)", "Linux ARM64 (AppImage)"]

COMMIT_RECEIPT_NAMES = sorted({leg for _label, leg, _pattern in _COMMIT_EXPECTED})
_COMMIT_JOBS = {
    "win32-x64": "build-win32-x64", "win32-arm64": "build-win32-arm64",
    "darwin-arm64": "build-darwin-arm64",
    "windows-universal": "assemble-win32-bundle", "termux": "termux-deb",
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
                   run_url: str | None = None,
                   ) -> list[tuple[str, str, str | None, str | None]]:
    """[(label, status, URL or None, link text or None)] for both summary sinks.

    Downloads require a receipt-listed object in the bucket. Missing artifacts
    link to the actual workflow run, never a guessed download or job URL.
    """
    failed = set(failed_legs or [])
    entries: list[tuple[str, str, str | None, str | None]] = []
    for row in commit_expected_rows(names, receipts):
        label, leg, key, state = row["label"], row["leg"], row["key"], row["state"]
        if state == "built":
            entries.append((label, "✅ Built", r2.public_url_for(base_url, key), key.rsplit("/", 1)[-1]))
            continue
        elif state == "receipt-missing":
            related = failed.intersection({leg, _COMMIT_JOBS[leg]})
            blame = f"failed: {', '.join(sorted(related))}" if related \
                else "leg incomplete or upload interrupted"
            status = f"❌ Not built ({blame})"
        elif state == "object-missing":
            status = "❌ Not built (receipt present but object missing)"
        elif state == "receipt-omits":
            status = "❌ Not built (artifact absent from receipt)"
        else:
            status = "❌ Not built (ambiguous: multiple objects match)"
        entries.append((label, status, run_url, "View build run" if run_url else None))
    for label in _COMMIT_DISABLED:
        entries.append((label, "Skipped (tag builds only)", None, None))
    return entries


def render_commit_summary(names: list[str], base_url: str, commit: str,
                          receipts: dict[str, dict | None],
                          failed_legs: list[str] | None = None,
                          run_url: str | None = None, *, smoke_results: dict | None = None) -> str:
    """Render every expected product without reading or changing a release."""
    _validated_commit_inputs(commit, receipts)
    lines = [
        f"## Commit build `{commit[:12]}`",
        "",
        "| Binary | Status | Download / diagnostics |",
        "|---|---|---|",
    ]
    for label, status, url, link_text in commit_entries(commit, names, base_url, receipts, failed_legs, run_url):
        cell = f"[{link_text}]({url})" if url and link_text else "—"
        lines.append(f"| {label} | {status} | {cell} |")
    return "\n".join([*lines, "", smoke_markdown(smoke_results), ""])


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


def incomplete_release_jobs(release_needs_json: str | None) -> list[str]:
    """Tag pages may describe failures; channel pointers require successful jobs.

    Unlike a commit summary, skipped release prerequisites are incomplete too.
    Diagnostic builds never are. Standalone invocations without workflow
    results retain their existing behavior.
    """
    if release_needs_json is None:
        return []
    try:
        needs = json.loads(release_needs_json)
        if not isinstance(needs, dict) or not needs:
            raise ValueError("missing workflow results")
        return [f"{name} ({needs[name].get('result', 'unknown')})"
                for name in sorted(job_groups.gating_jobs(needs)) if needs[name].get("result") != "success"]
    except (ValueError, TypeError, AttributeError):
        return ["workflow results unavailable"]


def render_commit_claim_tables(names: list[str], base_url: str, *,
                               smoke_results: dict | None = None) -> str:
    """Markdown download table for a commit claim's step summary (every staged downloadable asset)."""
    rows = commit_claim_rows(names)
    if rows:
        lines = [
            f"| {platform} | {arch} | [{kind} ({key.rsplit('/', 1)[-1]})]({r2.public_url_for(base_url, key)}) |"
            for platform, arch, kind, key in rows
        ]
        table = "### Downloads\n\n| Platform | Architecture | Download |\n|---|---|---|\n" + "\n".join(lines)
    else:
        table = "No downloadable artifacts were staged for this build."
    return MARKER + "\n" + "\n\n".join([table, smoke_markdown(smoke_results)]) + "\n" + END_MARKER


def render_commit_page(commit: str, names: list[str], base_url: str,
                       receipts: dict[str, dict | None],
                       failed_legs: list[str] | None = None,
                       run_url: str | None = None, *, repo: str = DEFAULT_REPO,
                       bundle_env: dict[str, str | None] | None = None,
                       smoke_results: dict | None = None) -> str:
    """The commit-build page: every expected binary, built or not."""
    _validated_commit_inputs(commit, receipts)
    commit_url = f"https://github.com/{quote(repo, safe='/')}/commit/{commit}"
    rows = []
    for label, status, url, link_text in commit_entries(commit, names, base_url, receipts, failed_legs, run_url):
        cell = (f"{_link(url)}{html.escape(link_text)}</a>" if url and link_text else "—")
        rows.append([html.escape(label), html.escape(status), cell])
    body = [
        f"<h1>Hermes commit build <code>{html.escape(commit[:12])}</code></h1>",
        f"<p>Commit {_link(commit_url)}<code>{html.escape(commit)}</code></a>. Every expected binary is listed; "
        "built rows link to downloads; incomplete rows link to the build run when available.</p>",
        *_table(("Binary", "Status", "Download / diagnostics"), rows),
        *_bundle_env_html(bundle_env),
    ]
    body.extend(smoke_html(smoke_results))
    return _page(f"Hermes commit build {commit[:12]}", commit, body)


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


def write_channel_page(tag: str, assets_by_app: dict, base_url: str,
                       *, repo: str = DEFAULT_REPO, smoke_results: dict | None = None) -> str | None:
    """Publish releases/<channel>/index.html for the tag's own channel."""
    key = r2.channel_page_key_for(r2.channel_for_tag(tag))
    if not supersedes(existing_page(key), tag):
        print(f"::warning::{key} already describes a newer release; leaving it unchanged")
        return None
    return write_page(key, render_page(tag, assets_by_app, base_url, repo=repo, smoke_results=smoke_results), base_url)


def r2_object_names_under(prefix: str) -> list[str]:
    return r2.list_objects(prefix=prefix)["keys"]


def r2_object_names(tag: str) -> list[str]:
    """Object keys in the R2 staging dir for `tag`, under releases/tag/<tag>/.

    A tag prefix and exact version match exclude neighboring releases. An
    attempt ref filters by its plain version, which is what file names carry.
    """
    from scripts.releases.versioning import parse_attempt_ref

    keys = r2_object_names_under(f"releases/tag/{tag}/")
    parsed = parse_attempt_ref(tag)
    from hermes_cli.update_channel import release_version

    return filter_names_for_version(keys, parsed[0] if parsed else (release_version(tag) or tag.lstrip("v")))


def splice(body: str, block: str) -> str:
    """Replace the marker (or a previously rendered block) with `block`."""
    if END_MARKER in body:
        pattern = re.compile(re.escape(MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
        return pattern.sub(lambda _m: block, body, count=1)
    return body.replace(MARKER, block, 1)


def render_channel_summary(manifest: dict) -> str:
    request = manifest["request"]
    rows = []
    for platform, arch in DESKTOP_TARGETS:
        selected = [row for row in manifest["packages"] if row["platform"] == platform
                    and row["arch"] == arch and row["variant"] == "bundled"]
        label = f"{platform} {arch}"
        if len(selected) != 1:
            rows.append(f"| {label} | Unavailable | — |")
            continue
        item = selected[0]
        url = r2.public_url_for(request["publicBase"], item["artifact"]["key"])
        rows.append(f"| {label} | {item['version']} | [Download]({url}) |")
    return (f"## Channel `{request['channel']}` build `{request['buildId']}`\n\n"
            f"Source `{request['commit']}` ({request['sourceVersion']}); sequence {request['sequence']}.\n\n"
            "| Native package | Version | Download |\n|---|---|---|\n" + "\n".join(rows) + "\n")


def _render_commit_summary(args, smoke_results: dict) -> int:
    """--summary-commit: the legacy tagless commit build's summary and releases/commit/<sha>/ page."""
    if not args.summary_out:
        print("::error::--summary-out is required with --summary-commit")
        return 1
    if not args.r2_base_url:
        print("::error::--r2-base-url (or CLOUDFLARE_R2_PUBLIC_URL) is required to render the summary")
        return 1
    commit = args.summary_commit
    try:
        from scripts.releases.bundle_env import decode

        bundle_env = decode(args.bundle_env_json)
        prefix = r2.commit_prefix_for(commit)
    except ValueError as err:
        print(f"::error::{err}")
        return 1
    names = r2_object_names_under(prefix)
    receipts = read_commit_receipts(commit)
    failed_legs = (failed_legs_from_release_needs(os.environ.get("RELEASE_NEEDS"))
                   or [leg.strip() for leg in args.summary_failed_legs.split(",") if leg.strip()])
    block = render_commit_summary(names, args.r2_base_url, commit, receipts, failed_legs, args.run_url,
                                  smoke_results=smoke_results)
    with open(args.summary_out, "a", encoding="utf-8") as out:
        out.write(block)
    write_page(r2.commit_page_key_for(commit),
               render_commit_page(commit, names, args.r2_base_url, receipts, failed_legs, args.run_url,
                                  repo=args.repo, bundle_env=bundle_env, smoke_results=smoke_results),
               args.r2_base_url)
    built = sum(1 for row in commit_expected_rows(names, receipts) if row["state"] == "built")
    print(f"✓ Commit summary appended to {args.summary_out} ({built}/{len(_COMMIT_EXPECTED)} binaries built)")
    return 0


def _write_tag_page(args, archive: str, assets: dict, incomplete: list[str],
                    smoke_results: dict, *, commit_files: list[str] | None = None) -> None:
    """The archive's own page. A commit build's page links its commit, lists every staged
    downloadable artifact and lists its bundle defaults."""
    from hermes_cli.update_channel import is_commit_tag
    from scripts.releases.bundle_env import decode

    commit_page = is_commit_tag(archive)
    write_page(r2.staging_key_for(archive, "index.html"),
               render_page(archive, assets, args.r2_base_url, incomplete, args.run_url,
                           repo=args.repo, smoke_results=smoke_results,
                           commit=args.candidate_commit if commit_page else None,
                           bundle_env=decode(args.bundle_env_json) if commit_page else None,
                           commit_files=commit_files if commit_page else None),
               args.r2_base_url)


def _finish_commit_build(args, parser, archive: str, block: str, names: list[str]) -> int:
    """A commit build owns no GitHub release and no channel page: its claim archive's page
    (already written) and the run summary are its whole publication."""
    if args.pending_run_url:
        parser.error("A commit build has no release body to hold a builds-in-progress link")
    if args.summary_out:
        with open(args.summary_out, "a", encoding="utf-8") as out:
            out.write(f"## Commit build {args.tag}\n\n"
                      f"Builds page: {r2.public_url_for(args.r2_base_url, r2.staging_key_for(archive, 'index.html'))}\n\n"
                      + block)
    print(f"✓ Commit build page rendered for {args.tag} ({len(names)} assets scanned)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-build")
    parser.add_argument("--channel-request-sha256")
    parser.add_argument("--tag", required=False, help="Release tag to render the release-body table for")
    parser.add_argument("--archive", default=None,
                        help="Attempt ref of the release archive when --tag is the plain payload tag")
    parser.add_argument("--candidate-manifest-sha256", default=None,
                        help="Stable promotion: render smoke admission from this pinned candidate, not RELEASE_NEEDS")
    parser.add_argument("--candidate-commit", default=None,
                        help="Exact admitted commit for --candidate-manifest-sha256")
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
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--bundle-env-json", default=os.environ.get("HERMES_BUNDLE_ENV_JSON", ""),
                        help="Explicit non-secret commit desktop bundle overrides, not the CI environment")
    parser.add_argument("--run-url", default=None,
                        help="Actual workflow run URL for incomplete-build diagnostics")
    parser.add_argument("--r2-base-url", default=os.environ.get("CLOUDFLARE_R2_PUBLIC_URL"),
                        help="Public base URL of the R2 bucket (default: $CLOUDFLARE_R2_PUBLIC_URL)")
    parser.add_argument("--no-channel-page", action="store_true",
                        help="Render the attempt page and release table but leave the shared channel "
                             "downloads page alone (a release that did not build every desktop package "
                             "the channel serves)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the spliced body instead of editing the release "
                             "(writes nothing, including the channel page)")
    parser.add_argument("--pending-run-url", default=None,
                        help="Render a 'builds in progress' link to this workflow run "
                             "instead of the tables")
    args = parser.parse_args()
    try:
        smoke_results = json.loads(os.environ.get("RELEASE_NEEDS", "{}"))
        if not isinstance(smoke_results, dict):
            smoke_results = {}
    except ValueError:
        smoke_results = {}

    if args.summary_commit and (args.tag or args.pending_run_url):
        parser.error("--summary-commit cannot be combined with release-body arguments")
    # The archive ref (attempt ref for stable attempts) keys every object the
    # page lists and writes; the payload tag names the GitHub release body.
    archive = args.archive or args.tag

    if args.channel_build:
        from hermes_cli.release_channels import ChannelReader
        from scripts.releases.channel_publish import read_request
        if args.tag or args.summary_commit or not args.summary_out or not args.r2_base_url:
            parser.error("Channel summary needs --summary-out and --r2-base-url; no tag/commit mode")
        request = read_request(args.channel_build, args.channel_request_sha256, args.r2_base_url, args.repo)
        resolved = ChannelReader(args.r2_base_url, repository=args.repo).resolve(request["channel"])
        if resolved.manifest is None or resolved.manifest["request"] != request:
            raise ValueError("Channel head no longer names this build; refusing stale download summary")
        with open(args.summary_out, "a", encoding="utf-8") as stream:
            stream.write(render_channel_summary(resolved.manifest))
        return 0

    candidate = None
    if args.candidate_manifest_sha256 is not None or args.candidate_commit is not None:
        if (args.candidate_manifest_sha256 is None or args.candidate_commit is None
                or not args.tag or not args.r2_base_url or args.summary_commit or args.pending_run_url):
            parser.error("Candidate rendering requires tag, base URL, manifest SHA256 and commit; no summary or pending mode")
        candidate = stable.read_admitted_candidate(args.tag, args.candidate_commit, args.r2_base_url,
                                                   args.candidate_manifest_sha256, archive=archive)
        smoke_results = candidate["smoke_results"]

    if args.summary_commit:
        return _render_commit_summary(args, smoke_results)

    if not args.tag:
        parser.error("--tag is required (or use --summary-commit for a commit build summary)")

    from hermes_cli.update_channel import is_commit_tag

    commit_claim = is_commit_tag(args.tag)
    incomplete: list[str] = []
    assets: dict = {}
    commit_files: list[str] | None = None
    if args.pending_run_url:
        block = render_pending(args.pending_run_url)
        names: list[str] = []
    else:
        if not args.r2_base_url:
            print("::error::--r2-base-url (or CLOUDFLARE_R2_PUBLIC_URL) is required to render the tables")
            return 1
        if commit_claim and candidate is not None:
            # Render directly from the admitted candidate's receipt-bound files so universal
            # .msixbundle, macOS .zip, Termux .deb and legacy pre-claim filenames all appear.
            names = [r2.staging_key_for(archive, item["path"]) for item in candidate["files"]]
            commit_files = names
            block = render_commit_claim_tables(names, args.r2_base_url, smoke_results=smoke_results)
        else:
            names = r2_object_names(archive)
            if candidate is not None:
                admitted = {r2.staging_key_for(archive, item["path"]) for item in candidate["files"]}
                names = [name for name in names if name in admitted]
            assets = parse_assets(names)
            incomplete = [] if candidate is not None else incomplete_release_jobs(os.environ.get("RELEASE_NEEDS"))
            block = render_tables(assets, args.r2_base_url, incomplete, args.run_url, smoke_results=smoke_results)
        # A failed run still owns its tag page, never the channel pointer
        # consumed by source updates. Missing artifacts never become downloads.
        if not args.dry_run:
            _write_tag_page(args, archive, assets, incomplete, smoke_results, commit_files=commit_files)

    if commit_claim:
        return _finish_commit_build(args, parser, archive, block, names)

    # Keep the per-tag diagnostic page even when GitHub cannot supply a draft.
    # Resolved through PATH and PATHEXT: Windows' CreateProcess would only ever look for gh.exe.
    gh = shutil.which("gh") or "gh"
    view = subprocess.run(
        [gh, "release", "view", args.tag, "--repo", args.repo,
         "--json", "body"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    if view.returncode != 0:
        print(f"::error::gh release view failed: {view.stderr.strip()}")
        return 1
    release = json.loads(view.stdout)
    body = release.get("body") or ""
    if not args.pending_run_url and not args.dry_run and not incomplete and names and not args.no_channel_page:
        write_channel_page(args.tag, assets, args.r2_base_url, repo=args.repo, smoke_results=smoke_results)
    if MARKER not in body:
        print("::warning::release body has no HERMES_BUILDS_TABLE marker; leaving it unchanged")
        return 0

    new_body = splice(body, block)
    if args.dry_run:
        print(new_body)
        return 0

    edit = subprocess.run(
        [gh, "release", "edit", args.tag, "--repo", args.repo,
         "--notes-file", "-"],
        input=new_body, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
        check=False,
    )
    if edit.returncode != 0:
        print(f"::error::gh release edit failed: {edit.stderr.strip()}")
        return 1
    what = "Builds-in-progress link" if args.pending_run_url else "Builds table"
    print(f"✓ {what} rendered into {args.tag} ({len(names)} assets scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
