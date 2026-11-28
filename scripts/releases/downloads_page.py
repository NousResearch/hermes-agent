"""Shared pure rendering and ordering helpers for R2 release download pages."""
from __future__ import annotations

import html
import json
import re
from urllib.parse import quote

from scripts.releases import r2, semver, stable, versioning

DEFAULT_REPO = "NousResearch/hermes-agent"

_ASSET_RE = re.compile(
    r"^(?P<app>HermesBundled|HermesLight)-(?P<version>[^-]+)"
    r"-(?P<os>mac|win|linux)-(?P<arch>x64|arm64)\.(?P<ext>dmg|msix|AppImage)$"
)

_OS_LABEL = {"mac": "macOS", "win": "Windows", "linux": "Linux (AppImage)"}
_ARCH_LABEL = {
    ("mac", "arm64"): "Apple Silicon (M-series)",
    ("win", "x64"): "x86 (64-bit)",
    ("win", "arm64"): "ARM (arm64 / aarch64)",
    ("linux", "x64"): "x86 (64-bit)",
    ("linux", "arm64"): "arm64",
}
_KIND_LABEL = {"dmg": "DMG", "msix": "MSIX", "AppImage": "AppImage"}
_ROW_ORDER = [("mac", "arm64"), ("win", "x64"), ("win", "arm64"),
              ("linux", "x64"), ("linux", "arm64")]
_UNLISTED_OS = frozenset({"linux"})

_SMOKE_SCOPE = (
    "Download availability is independent of smoke status. Each result covers all native legs "
    "in that format group. Linux AppImages and unsigned Store envelopes are not install-smoked. "
    "No-upload dry tags have no download handoff and are not smoke-qualified."
)

_PAGE_STYLE = (
    "body{font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:2rem auto;"
    "max-width:54rem;padding:0 1rem;color:#1a1a1a;background:#fff}"
    "h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:1.75rem}"
    "p{color:#444}table{border-collapse:collapse;width:100%}"
    "th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid #dcdcdc}"
    "th{font-weight:600}a{color:#0a58ca}code{font-size:.95em;white-space:pre-wrap;overflow-wrap:anywhere}"
)

_BUILD_META_RE = re.compile(r'<meta name="hermes-build" content="([^"]*)"')

_COMMIT_CLAIM_ASSETS = (
    ("Windows", "Universal (x64 + arm64)", "MSIXBUNDLE", r"^HermesBundled-[^-]+-win\.msixbundle$"),
    ("Windows", "x86 (64-bit)", "MSIX", r"^HermesBundled-[^-]+-win-x64\.msix$"),
    ("Windows", "ARM (arm64 / aarch64)", "MSIX", r"^HermesBundled-[^-]+-win-arm64\.msix$"),
    ("macOS", "Apple Silicon (M-series)", "DMG", r"^HermesBundled-[^-]+-mac-arm64\.dmg$"),
    ("macOS", "Apple Silicon (M-series)", "ZIP", r"^HermesBundled-[^-]+-mac-arm64\.zip$"),
    ("Termux (Android)", "ARM64 (aarch64)", ".deb", r"^.*\.deb$"),
)


def parse_assets(names: list[str]) -> dict[str, dict[tuple[str, str], tuple[str, str]]]:
    """{app: {(os, arch): (full_key, ext)}} for table-shaped assets only."""
    out: dict[str, dict[tuple[str, str], tuple[str, str]]] = {"HermesBundled": {}, "HermesLight": {}}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        match = _ASSET_RE.match(base)
        if match:
            out[match.group("app")][(match.group("os"), match.group("arch"))] = (name, match.group("ext"))
    return out


def table_rows(assets_by_app: dict) -> list[tuple[str, list[tuple[str, str, str, str]]]]:
    """[(section title, [(os label, arch label, kind label, object key), ...]), ...]."""
    sections: list[tuple[str, list[tuple[str, str, str, str]]]] = []
    for app, title in (("HermesBundled", "Hermes Desktop"), ("HermesLight", "Hermes Light (remote-only client)")):
        rows = []
        for key in _ROW_ORDER:
            entry = assets_by_app.get(app, {}).get(key)
            if not entry or key[0] in _UNLISTED_OS:
                continue
            name, ext = entry
            rows.append((_OS_LABEL[key[0]], _ARCH_LABEL[key], _KIND_LABEL[ext], name))
        if rows:
            sections.append((title, rows))
    return sections


def filter_names_for_version(names: list[str], version: str) -> list[str]:
    """Table-shaped names whose embedded version equals `version` (exact, not prefix)."""
    out = []
    for name in names:
        base = name.rsplit("/", 1)[-1]
        match = _ASSET_RE.match(base)
        if match and match.group("version") == version:
            out.append(name)
    return out


def smoke_rows(results: dict | None) -> list[tuple[str, str]]:
    statuses = {"success": "Passed", "failure": "Failed", "cancelled": "Cancelled", "skipped": "Not run"}
    return [(label, statuses.get((results or {}).get(job, {}).get("result"), "Not run (no result)"))
            for job, label in stable.SMOKE_JOBS.items()]


def smoke_markdown(results: dict | None) -> str:
    return ("### Native install/chat smoke\n\n" + _SMOKE_SCOPE
            + "\n\n| Packages | Smoke status |\n|---|---|\n"
            + "\n".join(f"| {label} | {status} |" for label, status in smoke_rows(results)))


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> list[str]:
    head = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    body = ["<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows]
    return [f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>", *body, "</tbody>\n</table>"]


def _link(url: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}">'


def _page(title: str, build: str, body: list[str]) -> str:
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="hermes-build" content="{html.escape(build, quote=True)}">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def smoke_html(results: dict | None) -> list[str]:
    return ["<h2>Native install/chat smoke</h2>", f"<p>{html.escape(_SMOKE_SCOPE)}</p>",
            *_table(("Packages", "Smoke status"),
                    [[html.escape(label), html.escape(status)] for label, status in smoke_rows(results)])]


def _bundle_env_html(bundle_env: dict[str, str | None] | None) -> list[str]:
    if not bundle_env:
        return []
    from scripts.releases.bundle_env import validate

    return [
        "<h2>Bundle environment</h2>",
        "<p>Explicit non-secret desktop bundle overrides only. Runtime values override defaults; "
        "Unset always removes the variable. String values use JSON notation.</p>",
        *_table(("Variable", "Value / action"), [
            [f"<code>{html.escape(key)}</code>", "Unset" if value is None else
             f"<code>{html.escape(json.dumps(value, ensure_ascii=False))}</code>"]
            for key, value in sorted(validate(bundle_env).items())
        ]),
    ]


def commit_claim_rows(names: list[str]) -> list[tuple[str, str, str, str]]:
    """[(platform label, arch label, kind label, full object key)] for staged commit-claim files."""
    rows: list[tuple[str, str, str, str]] = []
    for platform, arch, kind, pattern in _COMMIT_CLAIM_ASSETS:
        matched = sorted(key for key in names if re.fullmatch(pattern, key.rsplit("/", 1)[-1]))
        for key in matched:
            rows.append((platform, arch, kind, key))
    return rows


def render_page(tag: str, assets_by_app: dict, base_url: str,
                incomplete_jobs: list[str] | None = None,
                run_url: str | None = None, *, repo: str = DEFAULT_REPO,
                smoke_results: dict | None = None, commit: str | None = None,
                bundle_env: dict[str, str | None] | None = None,
                commit_files: list[str] | None = None) -> str:
    """The tag/channel page: the release-body download table as HTML."""
    from hermes_cli.update_channel import is_commit_tag

    commit_page = is_commit_tag(tag)
    if commit_page:
        if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("A commit build page needs its exact source commit")
        commit_url = f"https://github.com/{quote(repo, safe='/')}/commit/{commit}"
        body = [
            "<h1>Hermes Desktop commit build</h1>",
            f"<p>Commit build <code>{html.escape(tag)}</code> of {_link(commit_url)}<code>{commit}</code></a>. "
            "It updates nothing: no release, channel, feed or Store listing. Only objects this build "
            "actually staged in the bucket are listed.</p>",
        ]
    else:
        if bundle_env:
            raise ValueError("Only a commit build page lists bundle defaults")
        tag_url = f"https://github.com/{quote(repo, safe='/')}/releases/tag/{quote(tag, safe='')}"
        body = [
            f"<h1>Hermes Desktop {r2.channel_for_tag(tag)} builds</h1>",
            f"<p>Release {_link(tag_url)}<code>{html.escape(tag)}</code></a>. Only objects this release "
            "actually staged in the bucket are listed.</p>",
        ]
    attempt = versioning.parse_attempt_ref(tag)
    if attempt is not None:
        version = attempt[0]
        body.append("<p><strong>"
                    + html.escape(f"Attempt builds are not upgrade-safe: every attempt of {version} has the same "
                                  f"package version, so an installed attempt is not replaced by the published {version}.")
                    + "</strong></p>")
    if incomplete_jobs:
        body.append("<p><strong>Build incomplete.</strong> Jobs not successful: "
                    + html.escape(", ".join(incomplete_jobs))
                    + ". The channel page was not advanced.</p>")
    if commit_page and commit_files is not None:
        rows = commit_claim_rows(commit_files)
        if not rows:
            body.append("<p>No downloadable artifacts were staged for this build.</p>")
        else:
            body.append("<h2>Downloads</h2>")
            body.extend(_table(
                ("Platform", "Architecture", "Download"),
                [[html.escape(platform), html.escape(arch),
                  f"{_link(r2.public_url_for(base_url, key))}{html.escape(kind)} "
                  f"<code>{html.escape(key.rsplit('/', 1)[-1])}</code></a>"]
                 for platform, arch, kind, key in rows],
            ))
    else:
        sections = table_rows(assets_by_app)
        if not sections:
            body.append("<p>No downloadable artifacts were staged for this build.</p>")
        for title, rows in sections:
            body.append(f"<h2>{html.escape(title)}</h2>")
            body.extend(_table(
                ("OS", "Architecture", "Download"),
                [[html.escape(os_name), html.escape(arch),
                  f"{_link(r2.public_url_for(base_url, name))}{html.escape(kind)}</a>"]
                 for os_name, arch, kind, name in rows],
            ))
    if incomplete_jobs and run_url:
        body.append("<h2>Build diagnostics</h2>")
        body.extend(_table(
            ("Job", "Diagnostics"),
            [[html.escape(job), f"{_link(run_url)}View build run</a>"] for job in incomplete_jobs],
        ))
    body.extend(smoke_html(smoke_results))
    body.extend(_bundle_env_html(bundle_env))
    title = "Hermes Desktop commit build" if commit_page else f"Hermes Desktop {r2.channel_for_tag(tag)} builds"
    return _page(title, tag, body)


def render_channel_page_from_manifest(
    manifest: dict,
    base_url: str,
    *,
    repo: str = DEFAULT_REPO,
) -> str:
    """Render a channel downloads page purely from receipt-backed candidate manifest files."""
    from hermes_cli.update_channel import release_version

    tag = manifest["tag"]
    archive = manifest["archive"]
    version = release_version(tag)
    if version is None:
        raise ValueError(f"Not a release tag: {tag!r}")
    names = filter_names_for_version(
        [r2.staging_key_for(archive, item["path"]) for item in manifest.get("files", [])],
        version,
    )
    assets = parse_assets(names)
    if not table_rows(assets):
        raise ValueError(f"No downloadable desktop artifacts in candidate manifest for {tag}")
    return render_page(tag, assets, base_url, repo=repo, smoke_results=manifest.get("smoke_results"))


def recorded_build(page: str | None) -> str | None:
    """The release tag or commit a page in the bucket describes."""
    match = _BUILD_META_RE.search(page or "")
    return (match.group(1) or None) if match else None


def supersedes(existing_page: str | None, tag: str) -> bool:
    """Whether `tag` may replace the channel page without regressing a newer release."""
    recorded = recorded_build(existing_page)
    if not recorded:
        return True
    from hermes_cli.update_channel import canary_timestamp, published_canary_timestamp

    recorded_stamp = published_canary_timestamp(recorded)
    tag_stamp = canary_timestamp(tag)
    if (recorded_stamp is not None) != (tag_stamp is not None):
        return False
    if tag_stamp is not None:
        return recorded_stamp <= tag_stamp
    try:
        return semver.compare(recorded.lstrip("v"), tag.lstrip("v")) <= 0
    except ValueError:
        return True
