"""Stable release admission, signed-package transitions and final release receipt."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from hermes_cli.update_channel import STABLE_TAG_RE
from scripts.releases.draft_warning import strip_draft_warning
SHA = re.compile(r"[a-f0-9]{40}")
DIGEST = re.compile(r"[a-f0-9]{64}")
DESKTOP_TARGETS = ("windows/x64", "windows/arm64", "macos/x64", "macos/arm64")
SMOKE_JOBS = {
    "smoke-darwin-arm64": "macOS DMG + ZIP (arm64)",
    "smoke-darwin-x64": "macOS DMG + ZIP (x64)",
    "smoke-win32-arm64": "Windows MSIX (arm64)",
    "smoke-win32-x64": "Windows MSIX (x64)",
}


def admit_claim(tag: str, commit: str, *, on_main) -> dict:
    """Admit a release from its attempt ref. The checkout version is not read.

    ``main`` carries ``0.0.0`` on purpose, so the version and the attempt come
    from ``rc.<N>-vX.Y.Z`` and the only question about the commit is whether
    it is on ``main``.
    """
    from scripts.releases.versioning import parse_attempt_ref

    parsed = parse_attempt_ref(tag)
    if parsed is None:
        raise ValueError(f"{tag} is not a claim tag")
    version, attempt = parsed
    if not on_main(commit):
        raise ValueError(f"{commit} is not on main")
    return {"claim_tag": tag, "tag": f"v{version}", "version": version, "attempt": attempt,
            "commit": commit}


def require_stable_identity(tag: str, commit: str) -> None:
    """Validate the final payload identity without requiring its future ref."""
    if not isinstance(tag, str) or not STABLE_TAG_RE.fullmatch(tag) or not SHA.fullmatch(commit or ""):
        raise ValueError("Invalid stable payload identity")


def require_success(needs: dict, required: list[str]) -> None:
    if not required or len(set(required)) != len(required):
        raise ValueError("Invalid required-job list")
    failures = [f"{name}={needs.get(name, {}).get('result', 'missing')}"
                for name in required if needs.get(name, {}).get("result") != "success"]
    if failures:
        raise ValueError("Release blocked: " + ", ".join(failures))


# The claim flags that remove stable-release.yml jobs, per job. A job one of
# the claim's active flags removes must report `skipped`. Every other gated job
# must report `success`. Jobs not listed here never skip.
CLAIM_FLAGS = ("autopublish", "skipBundles", "skipTests")
_TESTS, _BUNDLES = frozenset({"skipTests"}), frozenset({"skipBundles"})
SKIPPED_BY = {
    **{job: _TESTS for job in ("ci", "nix", "termux-checks", "windows-live", "install-e2e",
                               "bootstrap-version")},
    **{job: _BUNDLES for job in ("candidates-darwin-arm64", "candidates-darwin-x64",
                                 "candidates-win32-arm64", "candidates-win32-x64",
                                 "candidates-win32-bundle", "candidates-termux",
                                 "candidate-manifest", "publish-bundles")},
    # Bundle acceptance is a test of bundles, so either flag removes it.
    **{job: _TESTS | _BUNDLES for job in ("pm-bundle", "transitions-darwin-arm64",
                                          "transitions-darwin-x64", "transitions-win32",
                                          "windows-packaged", "macos-packaged-arm64",
                                          "macos-packaged-x64")},
}


def gate_expectations(required: list[str], *, skip_bundles: bool, skip_tests: bool) -> dict:
    """Each gated job's required result under the claim's flags."""
    active = {flag for flag, on in (("skipBundles", skip_bundles), ("skipTests", skip_tests)) if on}
    return {name: "skipped" if SKIPPED_BY.get(name, frozenset()) & active else "success"
            for name in required}


def require_gate(needs: dict, required: list[str], *, skip_bundles: bool, skip_tests: bool) -> None:
    """``require_success`` that also demands a flag-removed job really was skipped."""
    if not required or len(set(required)) != len(required):
        raise ValueError("Invalid required-job list")
    expected = gate_expectations(required, skip_bundles=skip_bundles, skip_tests=skip_tests)
    failures = [f"{name}={needs.get(name, {}).get('result', 'missing')} (expected {want})"
                for name, want in expected.items() if needs.get(name, {}).get("result") != want]
    if failures:
        raise ValueError("Release blocked: " + ", ".join(failures))


def accepted_smoke_results(needs: object) -> dict:
    """Persist only observed native groups, never infer them from artifacts.

    Every group passed, or every group was skipped. Only a claim that skipped
    tests may produce the second shape. ``smokes_skipped`` lets claim-aware
    readers check that.
    """
    if not isinstance(needs, dict):
        raise ValueError("Candidate smoke results must be a job-result object")
    results = {}
    for job in SMOKE_JOBS:
        row = needs.get(job)
        results[job] = {"result": row.get("result") if isinstance(row, dict) else None}
    if {row["result"] for row in results.values()} != {"skipped"}:
        require_success(results, list(SMOKE_JOBS))
    return results


def smokes_skipped(manifest: dict) -> bool:
    results = manifest.get("smoke_results") or {}
    return all((results.get(job) or {}).get("result") == "skipped" for job in SMOKE_JOBS)


def require_smokes_match_claim(manifest: dict, *, skip_tests: bool) -> None:
    if smokes_skipped(manifest) != skip_tests:
        raise ValueError("Candidate smoke results differ from the claim's test policy")


def stable_windows_version(epoch: object) -> str:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Stable release epoch must be a non-negative integer")
    instant = datetime.fromtimestamp(epoch, tz=timezone.utc)
    start = datetime(instant.year, 1, 1, tzinfo=timezone.utc)
    hour_of_year = (instant - start).days * 24 + instant.hour
    second_of_hour = instant.minute * 60 + instant.second
    return f"{instant.year}.{hour_of_year}.{second_of_hour}.0"


RECEIPT_TARGETS = {
    "darwin-arm64": ("macos/arm64",),
    "darwin-x64": ("macos/x64",),
    "win32-bundle": ("windows/x64", "windows/arm64"),
}

# The staged handoffs each receipt is assembled from. The Windows bundle
# receipt reads the per-arch metadata handoffs plus the universal bundle
# handoff; fetch() re-verifies every staged byte against its receipt digest,
# and validate_receipt enforces the signing facts (teamId, publisher).
RECEIPT_HANDOFFS = {
    "darwin-arm64": ("darwin-arm64",),
    "darwin-x64": ("darwin-x64",),
    "win32-bundle": ("win32-x64", "win32-arm64", "windows-universal"),
}
RECEIPT_INCLUDES = ("metadata-*.json", "*.zip", "*.msixbundle")


def _validated_rows(manifest: dict, tag: str, commit: str, public_base: str,
                    release_epoch: int | None, *, archive: str) -> dict:
    """The per-row checks shared by the full-manifest and receipt validators."""
    require_stable_identity(tag, commit)
    if manifest.get("schema") != 2 or manifest.get("tag") != tag or manifest.get("commit") != commit or not isinstance(manifest.get("packages"), list):
        raise ValueError("Candidate manifest does not match release identity")
    if manifest.get("archive") != archive:
        raise ValueError("Candidate manifest names a different release archive")
    admitted_epoch = manifest.get("releaseEpoch")
    expected_windows_version = stable_windows_version(admitted_epoch)
    if release_epoch is not None and admitted_epoch != release_epoch:
        raise ValueError("Candidate release epoch differs from the admitted claim")
    prefix = urlsplit(f"{public_base.rstrip('/')}/releases/tag/{archive}/")
    if prefix.scheme != "https" or prefix.username or prefix.password or not prefix.netloc:
        raise ValueError("Public release origin must use HTTPS")
    rows = {}
    for item in manifest["packages"]:
        target = f"{item.get('platform')}/{item.get('arch')}"
        if target not in (*DESKTOP_TARGETS, "termux/aarch64") or target in rows or item.get("tag") != tag or item.get("commit") != commit:
            raise ValueError(f"Invalid or duplicate candidate target: {target}")
        artifact = item.get("artifact", {})
        url = urlsplit(artifact.get("url", ""))
        decoded = unquote(url.path)
        if any(part in (".", "..") for part in decoded.split("/")) or "\\" in decoded or "%" in decoded:
            raise ValueError("Invalid artifact path encoding")
        if (url.scheme, url.netloc) != (prefix.scheme, prefix.netloc) or not url.path.startswith(prefix.path) or url.query or url.fragment or url.username or url.password:
            raise ValueError(f"Candidate package is outside its immutable tag archive: {target}")
        if not DIGEST.fullmatch(artifact.get("sha256", "")) or not item.get("identity"):
            raise ValueError(f"Invalid candidate digest or identity: {target}")
        if item["platform"] == "windows":
            if (item.get("version") != expected_windows_version
                    or item.get("executableVersion") != expected_windows_version):
                raise ValueError("Windows candidate version differs from the admitted release epoch")
            windows_version(item["version"])
            if not item.get("publisher") or not item.get("applicationId") or not url.path.endswith(".msixbundle"):
                raise ValueError("Windows candidate needs publisher, applicationId and MSIX bundle")
        elif item["platform"] == "macos":
            if item.get("version") != tag[1:] or not re.fullmatch(r"[A-Z0-9]{10}", item.get("teamId", "")) or not url.path.endswith(".zip"):
                raise ValueError("macOS candidate needs matching version, signing team and app ZIP")
        elif item.get("version") != f"{tag[1:]}-1":
            raise ValueError("Termux candidate version differs from the admitted release")
        rows[target] = item
    return rows


def validate_candidates(manifest: dict, tag: str, commit: str, public_base: str,
                        release_epoch: int | None = None, *, archive: str) -> dict:
    """`tag` is the plain payload identity; `archive` is the releases/tag/<ref>/
    prefix every artifact URL must live under. Stable attempts name the attempt
    ref as their archive; the two are separate fields and never overloaded.

    The smoke requirement lives here and not in the shared row checks:
    per-arch receipts are staged before the smokes run (decision 11), so a
    receipt without `smoke_results` is accepted while the final manifest
    never is.
    """
    rows = _validated_rows(manifest, tag, commit, public_base, release_epoch, archive=archive)
    accepted_smoke_results(manifest.get("smoke_results"))
    if any(target not in rows for target in DESKTOP_TARGETS):
        raise ValueError("Candidate manifest must cover Windows and macOS on both architectures")
    return rows


def validate_receipt(manifest: dict, receipt: str, tag: str, commit: str, public_base: str,
                     release_epoch: int | None = None, *, archive: str) -> dict:
    """One per-arch or bundle receipt: exactly that group's rows and no others."""
    targets = RECEIPT_TARGETS.get(receipt)
    if targets is None:
        raise ValueError(f"Unknown receipt: {receipt}")
    rows = _validated_rows(manifest, tag, commit, public_base, release_epoch, archive=archive)
    if set(rows) != set(targets):
        raise ValueError(
            f"Receipt {receipt} requires exactly {', '.join(targets)} and nothing else")
    return rows


def windows_version(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value):
        raise ValueError("Windows package version must have four numeric components")
    result = tuple(map(int, value.split(".")))
    if any(n > 65535 for n in result):
        raise ValueError("Windows package version exceeds 16 bits")
    return result


def _transition_row(target: str, left: dict, right: dict) -> dict:
    if left["identity"] != right["identity"] or left["commit"] == right["commit"] or left["artifact"]["sha256"] == right["artifact"]["sha256"]:
        raise ValueError("Update must preserve package identity and change the build")
    if right["platform"] == "windows":
        if (left["publisher"], left["applicationId"]) != (right["publisher"], right["applicationId"]):
            raise ValueError("Update must preserve publisher and applicationId")
        newer = windows_version(right["version"]) > windows_version(left["version"])
    else:
        if left["teamId"] != right["teamId"]:
            raise ValueError("Update must preserve signing team")
        newer = tuple(map(int, right["version"].split("."))) > tuple(map(int, left["version"].split(".")))
    if not newer:
        raise ValueError("New package version must increase")
    return {"target": target.replace("/", "-"), "transition": {
        "schema": 1, "platform": right["platform"], "arch": right["arch"], "old": left, "new": right,
    }}


def plan_transitions(previous: dict, candidate: dict, public_base: str) -> list[dict]:
    old = validate_candidates(previous, previous.get("tag"), previous.get("commit"), public_base,
                              archive=previous.get("archive"))
    new = validate_candidates(candidate, candidate.get("tag"), candidate.get("commit"), public_base,
                              archive=candidate.get("archive"))
    return [_transition_row(target, old[target], new[target]) for target in DESKTOP_TARGETS]


def plan_receipt_transitions(previous: dict, receipt_manifest: dict, receipt: str,
                             public_base: str) -> list[dict]:
    """The same transitions, but only for one receipt's rows."""
    targets = RECEIPT_TARGETS.get(receipt)
    if targets is None:
        raise ValueError(f"Unknown receipt: {receipt}")
    old = validate_candidates(previous, previous.get("tag"), previous.get("commit"), public_base,
                              archive=previous.get("archive"))
    new = validate_receipt(receipt_manifest, receipt, receipt_manifest.get("tag"),
                           receipt_manifest.get("commit"), public_base,
                           archive=receipt_manifest.get("archive"))
    return [_transition_row(target, old[target], new[target]) for target in targets]


def _receipt_manifest(root: Path, receipt: str, tag: str, commit: str, archive: str,
                      public_base: str, release_epoch: int) -> dict:
    """Rebuild one group's manifest rows from its staged, digest-verified handoffs."""
    from scripts.releases.handoff import fetch, receipt_name
    from scripts.releases.r2 import staging_key_for

    names = RECEIPT_HANDOFFS.get(receipt)
    if names is None:
        raise ValueError(f"Unknown receipt: {receipt}")
    # Re-downloading the staged bytes proves the group's handoff is complete
    # and matches its receipt before this receipt is published.
    fetch(tag=archive, commit=commit, names=list(names), root=root,
          includes=list(RECEIPT_INCLUDES))
    digests = {}
    for name in names:
        for row in json.loads((root / receipt_name(name)).read_text(encoding="utf-8-sig"))["files"]:
            digests[row["path"]] = row["sha256"]
    rows = [json.loads(file.read_text(encoding="utf-8-sig"))
            for file in sorted(root.glob("metadata-*.json"))]
    universal = None
    if receipt == "win32-bundle":
        bundles = [file.name for file in root.glob("*.msixbundle") if not file.name.startswith("Store-")]
        if len(bundles) != 1:
            raise ValueError(f"Expected one universal bundle, found {len(bundles)}")
        universal = bundles[0]
    packages = []
    for row in rows:
        filename = universal if row["platform"] == "windows" else row.get("filename")
        if not filename or filename not in digests or not (root / filename).is_file():
            raise ValueError(f"Receipt {receipt} is missing staged bytes for "
                             f"{row['platform']}/{row['arch']}")
        packages.append({
            key: value for key, value in {
                **row,
                "artifact": {"url": f"{public_base.rstrip('/')}/{staging_key_for(archive, filename)}",
                             "sha256": digests[filename]},
            }.items() if key != "filename"
        })
        if row["platform"] != "windows" and not filename.endswith(".zip"):
            raise ValueError(f"Receipt {receipt} needs a signed app ZIP for {row['platform']}/{row['arch']}")
    if receipt == "win32-bundle":
        from scripts.bundles.release_artifacts import validate_windows_bundle

        validate_windows_bundle(root / universal,
                                [row for row in rows if row["platform"] == "windows"])
    return {"schema": 2, "tag": tag, "commit": commit, "releaseEpoch": release_epoch,
            "archive": archive, "packages": packages}


def stage_receipt(env: dict, receipt: str) -> None:
    """Publish one group's signed receipt into its immutable attempt archive.

    The receipt names exactly that group's rows (decision 11): it is staged
    before the group's smokes run, so it carries no smoke results, while
    acceptance still blocks publication.
    """
    from scripts.releases.r2 import put

    tag, commit, claim = stable_context(env)
    base = env["CLOUDFLARE_R2_PUBLIC_URL"].rstrip("/")
    archive = claim["claim_tag"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _receipt_manifest(root, receipt, tag, commit, archive, base, claim["claim_epoch"])
        validate_receipt(manifest, receipt, tag, commit, base, claim["claim_epoch"], archive=archive)
        file = root / f"{receipt}-receipt.json"
        file.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        put(tag=archive, key=file.name, file=str(file), immutable=True)
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
    url = f"{base}/releases/tag/{archive}/{receipt}-receipt.json"
    print(url)
    print(digest)
    emit({"receipt-url": url, "receipt-sha256": digest}, env)


def read_manifest(url: str, expected_hash: str | None = None, *, expected_origin: str | None = None,
                  opener=None) -> dict:
    # Resolved per call: a default bound at import would pin the opener that
    # existed then and ignore the process's trust setup.
    opener = opener or urllib.request.urlopen
    location = urlsplit(url)
    origin = urlsplit(expected_origin or url)

    def check_origin(target):
        if target.scheme != "https" or not target.hostname or target.username or target.password:
            raise ValueError("Manifest origin must use HTTPS without credentials")
        if (target.scheme, target.hostname, target.port or 443) != (origin.scheme, origin.hostname, origin.port or 443):
            raise ValueError("Manifest is outside the expected release origin")

    check_origin(location)
    with opener(url, timeout=60) as response:
        check_origin(urlsplit(response.geturl()))
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Release manifest exceeds size limit")
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Candidate manifest digest mismatch")
    return json.loads(data)


def output(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, encoding="utf-8").strip()


def validate_claim(metadata: object, *, version: str, attempt: int, commit: str) -> dict:
    """The claim message's exact shape. It is the one record of the attempt's policy."""
    expected = {"schema": 1, "version": version, "attempt": attempt, "commit": commit}
    if (not isinstance(metadata, dict)
            or any(metadata.get(key) != value for key, value in expected.items())
            or any(not isinstance(metadata.get(flag), bool) for flag in CLAIM_FLAGS)
            or not isinstance(metadata.get("claimEpoch"), int)
            or metadata["claimEpoch"] <= 0
            or set(metadata) != {*expected, *CLAIM_FLAGS, "claimEpoch"}):
        raise ValueError("Stable claim metadata is invalid")
    return metadata


def _claim_metadata(raw: str, *, version: str, attempt: int, commit: str) -> dict:
    try:
        metadata = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Stable claim metadata is invalid") from error
    return validate_claim(metadata, version=version, attempt=attempt, commit=commit)


def validate_final(final: object, *, version: str, commit: str, claim_tag: str,
                   claim_object: str, claim: dict) -> dict:
    """The final receipt tag's exact shape, bound to its validated claim.

    A claim that skipped bundles has no candidate manifest, so its receipt
    records ``candidateManifestSha256: null``. Every other receipt pins one.
    """
    if not isinstance(final, dict):
        raise ValueError("Final tag metadata differs from its claim")
    expected = {
        "schema": 1, "version": version, "commit": commit,
        "claimTag": claim_tag, "claimTagObject": claim_object,
        "autopublish": claim["autopublish"],
        "claimEpoch": claim["claimEpoch"],
        "releaseId": final.get("releaseId"),
        "candidateManifestSha256": final.get("candidateManifestSha256"),
        "dockerManifestDigest": final.get("dockerManifestDigest"),
        "archive": f"releases/tag/{claim_tag}/",
    }
    manifest = final.get("candidateManifestSha256")
    manifest_ok = manifest is None if claim["skipBundles"] else bool(DIGEST.fullmatch(manifest or ""))
    if (final != expected
            or not isinstance(final["releaseId"], int) or final["releaseId"] <= 0
            or not manifest_ok
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", final["dockerManifestDigest"] or "")):
        raise ValueError("Final tag metadata differs from its claim")
    return final


def tagger_epoch(tag_object: str, run=output) -> int:
    body = run(["git", "cat-file", "-p", tag_object])
    tagger = next((line for line in body.splitlines() if line.startswith("tagger ")), None)
    if tagger is None:
        raise ValueError("Stable claim has no tagger timestamp")
    try:
        epoch = int(tagger.rsplit(" ", 2)[1])
    except (IndexError, ValueError) as error:
        raise ValueError("Stable claim tagger timestamp is invalid") from error
    if epoch <= 0:
        raise ValueError("Stable claim tagger timestamp is invalid")
    return epoch


def check_claim(env: dict, run=output) -> dict:
    """Bind the run to one remote annotated claim object and its commit."""
    claim_tag, commit = env.get("RELEASE_CLAIM_TAG"), env.get("GITHUB_SHA")
    if not isinstance(claim_tag, str) or env.get("GITHUB_REF") != f"refs/tags/{claim_tag}":
        raise ValueError("Stable release must run on its exact claim ref")
    if not isinstance(commit, str) or not SHA.fullmatch(commit):
        raise ValueError("Stable claim needs an exact commit")
    claim_ref = f"refs/tags/{claim_tag}"
    local_object = run(["git", "rev-parse", claim_ref])
    local_commit = run(["git", "rev-parse", f"{claim_ref}^{{commit}}"])
    if run(["git", "cat-file", "-t", local_object]) != "tag":
        raise ValueError("Stable claim must be an annotated tag")
    remote = dict(line.split()[::-1] for line in run(
        ["git", "ls-remote", "origin", claim_ref, f"{claim_ref}^{{}}"]
    ).splitlines())
    remote_object = remote.get(claim_ref)
    remote_commit = remote.get(f"{claim_ref}^{{}}")
    expected_object = env.get("RELEASE_CLAIM_OBJECT")
    if (local_commit != commit or remote_commit != commit or remote_object != local_object
            or (expected_object and remote_object != expected_object)
            or run(["git", "rev-parse", "HEAD"]) != commit):
        raise ValueError("Stable claim tag or checkout moved")
    run(["git", "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"])

    def on_main(sha: str) -> bool:
        try:
            run(["git", "merge-base", "--is-ancestor", sha, "origin/main"])
        except subprocess.CalledProcessError:
            return False
        return True

    admitted = admit_claim(claim_tag, commit, on_main=on_main)
    raw_metadata = run(["git", "tag", "-l", claim_tag, "--format=%(contents)"])
    metadata = _claim_metadata(raw_metadata, version=admitted["version"],
                               attempt=admitted["attempt"], commit=commit)
    claim_epoch = tagger_epoch(local_object, run)
    if metadata["claimEpoch"] != claim_epoch:
        raise ValueError("Stable claim epoch differs from its annotated tagger timestamp")
    return {**admitted, "claim_object": local_object,
            "autopublish": metadata["autopublish"], "skip_bundles": metadata["skipBundles"],
            "skip_tests": metadata["skipTests"], "claim_epoch": claim_epoch}


def stable_context(env: dict, run=output) -> tuple[str, str, dict]:
    claim = check_claim(env, run=run)
    tag = env.get("RELEASE_TAG")
    if not isinstance(tag, str) or tag != claim["tag"]:
        raise ValueError("Stable payload tag differs from the admitted claim")
    return tag, claim["commit"], claim


def final_context(env: dict, run=output) -> tuple[str, str, dict]:
    """Verify the final annotated receipt, its claim, and published release."""
    repository = env.get("GITHUB_REPOSITORY", "")
    tag = env.get("RELEASE_TAG", "")
    commit = env.get("RELEASE_COMMIT", "")
    claim_tag = env.get("RELEASE_CLAIM_TAG", "")
    claim_object = env.get("RELEASE_CLAIM_OBJECT", "")
    require_stable_identity(tag, commit)
    admitted = admit_claim(claim_tag, commit, on_main=lambda _commit: True)
    if admitted["tag"] != tag or not SHA.fullmatch(claim_object):
        raise ValueError("Final release differs from its claim")

    refs = {}
    for line in run(["git", "ls-remote", "origin",
                     f"refs/tags/{claim_tag}", f"refs/tags/{claim_tag}^{{}}",
                     f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"]).splitlines():
        sha, ref = line.split()
        refs[ref] = sha
    if (refs.get(f"refs/tags/{claim_tag}") != claim_object
            or refs.get(f"refs/tags/{claim_tag}^{{}}") != commit
            or refs.get(f"refs/tags/{tag}^{{}}") != commit
            or not refs.get(f"refs/tags/{tag}")):
        raise ValueError("Final release tag custody changed")

    run(["git", "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main",
         f"+refs/tags/{claim_tag}:refs/tags/{claim_tag}",
         f"+refs/tags/{tag}:refs/tags/{tag}"])
    for receipt, expected_object in ((claim_tag, claim_object),
                                     (tag, refs[f"refs/tags/{tag}"])):
        local_object = run(["git", "rev-parse", f"refs/tags/{receipt}"])
        if local_object != expected_object or run(["git", "cat-file", "-t", local_object]) != "tag":
            raise ValueError("Final release local tag differs from the remote")
    run(["git", "merge-base", "--is-ancestor", commit, "origin/main"])
    claim = _claim_metadata(
        run(["git", "tag", "-l", claim_tag, "--format=%(contents)"]),
        version=admitted["version"], attempt=admitted["attempt"], commit=commit,
    )
    final = validate_final(
        json.loads(run(["git", "tag", "-l", tag, "--format=%(contents)"])),
        version=admitted["version"], commit=commit, claim_tag=claim_tag,
        claim_object=claim_object, claim=claim,
    )
    release = json.loads(run([
        "gh", "api", f"repos/{repository}/releases/tags/{tag}",
    ]))
    if (release.get("id") != final["releaseId"] or release.get("tag_name") != tag
            or release.get("draft") is not False
            or release.get("prerelease") is not False or not release.get("published_at")):
        raise ValueError("Stable channel requires the published final release")
    return tag, commit, {**admitted, "claim_object": claim_object,
                         "autopublish": claim["autopublish"],
                         "skip_bundles": claim["skipBundles"],
                         "skip_tests": claim["skipTests"],
                         "claim_epoch": claim["claimEpoch"],
                         "release_id": final["releaseId"],
                         "candidate_manifest_sha256": final["candidateManifestSha256"],
                         "docker_manifest_digest": final["dockerManifestDigest"]}


def emit(values: dict, env: dict) -> None:
    with Path(env["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as file:
        for key, value in values.items():
            file.write(f"{key}={value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))}\n")


def read_candidate(env: dict) -> dict:
    digest = env.get("CANDIDATE_MANIFEST_SHA256", "")
    if not DIGEST.fullmatch(digest):
        raise ValueError("Pinned candidate manifest digest is required")
    return read_manifest(env["CANDIDATE_MANIFEST_URL"], digest)


def read_admitted_candidate(tag: str, commit: str, public_base: str, digest: str, *,
                            archive: str) -> dict:
    """The page and package promoter consume the same pinned admission."""
    if not DIGEST.fullmatch(digest or ""):
        raise ValueError("Pinned candidate manifest digest is required")
    require_stable_identity(tag, commit)
    manifest = read_manifest(f"{public_base.rstrip('/')}/releases/tag/{archive}/release-candidates.json",
                             digest, expected_origin=public_base)
    validate_candidates(manifest, tag, commit, public_base, archive=archive)
    return manifest


def summary(text: str, env: dict) -> None:
    with Path(env["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def admit(env: dict) -> None:
    """Admit the claim. The checkout carries 0.0.0, so the tag is the version."""
    admitted = check_claim(env)
    repository = env["GITHUB_REPOSITORY"]
    release = json.loads(output([
        "gh", "release", "view", admitted["claim_tag"], "--repo", repository,
        "--json", "databaseId,tagName,isDraft,isPrerelease",
    ]))
    if (release.get("tagName") != admitted["claim_tag"] or release.get("isDraft") is not True
            or release.get("isPrerelease") is not False or not isinstance(release.get("databaseId"), int)):
        raise ValueError("Stable claim must already own one non-prerelease draft")
    emit({
        "claim-tag": admitted["claim_tag"], "claim-object": admitted["claim_object"],
        "tag": admitted["tag"], "commit": admitted["commit"], "version": admitted["version"],
        "release-id": release["databaseId"], "release-epoch": admitted["claim_epoch"],
        "skip-bundles": "true" if admitted["skip_bundles"] else "false",
        "skip-tests": "true" if admitted["skip_tests"] else "false",
    }, env)
    skipped = [name for name, on in (("bundles", admitted["skip_bundles"]),
                                     ("tests", admitted["skip_tests"])) if on]
    summary(
        f"## Stable candidate {admitted['claim_tag']}\nCommit: {admitted['commit']}\n"
        f"Version: {admitted['version']}\nPayload tag: {admitted['tag']}\n"
        f"Skipped: {', '.join(skipped) or 'nothing'}\n",
        env,
    )


def verify(env: dict) -> None:
    """Revalidate claim custody in a reusable privileged workflow."""
    tag, commit, claim = stable_context(env)
    emit({"tag": tag, "sha": commit, "channel": "stable", "payload-version": tag[1:],
          "release-epoch": claim["claim_epoch"]}, env)


def _stage_transition(env: dict, archive: str, base: str, row: dict) -> dict:
    from scripts.releases.r2 import put

    transition = row["transition"]
    name = f"acceptance-{row['target']}.json"
    file = Path(env["RUNNER_TEMP"]) / name
    file.write_text(json.dumps(transition), encoding="utf-8")
    put(tag=archive, key=name, file=str(file), immutable=True)
    url = f"{base}/releases/tag/{archive}/{name}"
    if read_manifest(url) != transition:
        raise ValueError("Transition manifest read-back mismatch")
    return {"arch": transition["arch"], "manifest": url, "old": transition["old"]["tag"],
            "id": row["target"],
            "manifest_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}


def _published_baseline(env: dict, base: str) -> dict:
    try:
        previous = read_manifest(env.get("BASELINE_MANIFEST_URL") or f"{base}/releases/stable/release-candidates.json",
                                 expected_origin=base)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError("No published stable package baseline. Supply baseline-manifest for an actual previous stable release; acceptance cannot be skipped.") from error
        raise
    published = json.loads(output(["gh", "release", "view", previous["tag"], "--repo", env["GITHUB_REPOSITORY"], "--json", "tagName,isDraft,isPrerelease"]))
    if published["tagName"] != previous["tag"] or published["isDraft"] or published["isPrerelease"]:
        raise ValueError("Upgrade baseline must be a published stable release")
    return previous


def _receipt_from_env(env: dict, receipt: str, prefix: str, base: str) -> dict:
    digest = env.get(f"{prefix}_SHA256", "")
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"Pinned {receipt} receipt digest is required")
    return read_manifest(env[f"{prefix}_URL"], digest, expected_origin=base)


def transitions(env: dict) -> None:
    """Plan one install arm from its own receipt.

    Each transitions job reads exactly one group's receipt (decision 11) and
    emits only that group's rows, so a Mac arch and the Windows bundle start
    their install arms independently of the other groups.
    """
    receipt = env.get("RECEIPT", "")
    if receipt not in RECEIPT_TARGETS:
        raise ValueError(f"Unknown receipt: {receipt}")
    tag, commit, claim = stable_context(env)
    base = env["CLOUDFLARE_R2_PUBLIC_URL"].rstrip("/")
    archive = claim["claim_tag"]
    previous = _published_baseline(env, base)
    receipt_manifest = _receipt_from_env(env, receipt, "RECEIPT", base)
    matrices = {"windows": {"include": []}, "macos": {"include": []}}
    for row in plan_receipt_transitions(previous, receipt_manifest, receipt, base):
        matrices[row["transition"]["platform"]]["include"].append(
            _stage_transition(env, archive, base, row))
    emit(matrices, env)


# The candidate manifest is written after the smokes (decision 23): the smoke
# results it records are the candidate calls' own workflow results — each
# call's stable-phase-result only succeeds when its selected groups' smokes
# did, so a failed smoke leaves no accepted manifest behind.
CALL_SMOKE_JOBS = {
    "candidates-darwin-arm64": "smoke-darwin-arm64",
    "candidates-darwin-x64": "smoke-darwin-x64",
    "candidates-win32-arm64": "smoke-win32-arm64",
    "candidates-win32-x64": "smoke-win32-x64",
}
CANDIDATE_HANDOFFS = ("win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64",
                      "termux", "windows-universal")
CANDIDATE_INCLUDES = ("metadata-*.json", "*.msixbundle")


def candidate_manifest(env: dict) -> None:
    """Merge the staged handoffs into the accepted candidate manifest."""
    from scripts.bundles.release_artifacts import assemble
    from scripts.releases.handoff import fetch

    needs = json.loads(env.get("RELEASE_NEEDS", "{}"))
    if not isinstance(needs, dict):
        raise ValueError("Candidate call results must be a needs object")
    tag, commit, claim = stable_context(env)
    if claim["skip_bundles"]:
        raise ValueError("A claim that skipped bundles has no candidate manifest")
    if claim["skip_tests"]:
        # The calls built and staged their groups but ran no smoke. Record
        # that as skipped. Never let a green call stand in for a smoke.
        require_success(needs, list(CALL_SMOKE_JOBS))
        smoke = {job: {"result": "skipped"} for job in CALL_SMOKE_JOBS.values()}
    else:
        smoke = {job: {"result": (needs.get(call) or {}).get("result")}
                 for call, job in CALL_SMOKE_JOBS.items()}
        require_success(smoke, list(smoke))
    base = env["CLOUDFLARE_R2_PUBLIC_URL"].rstrip("/")
    archive = claim["claim_tag"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fetch(tag=archive, commit=commit, names=list(CANDIDATE_HANDOFFS), root=root,
              includes=list(CANDIDATE_INCLUDES))
        out = root / "release-candidates.json"
        assemble(root, tag, commit, base, out, smoke_results=smoke,
                 release_epoch=claim["claim_epoch"], archive=archive)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
    emit({"manifest-url": f"{base}/releases/tag/{archive}/release-candidates.json",
          "manifest-sha256": digest}, env)


def _final_metadata(tag: str, commit: str, claim: dict, candidate_manifest_sha256: str | None,
                    docker_manifest_digest: str, release_id: int) -> dict:
    if claim["skip_bundles"]:
        if candidate_manifest_sha256 is not None:
            raise ValueError("A claim that skipped bundles cannot bind a candidate manifest")
    elif not DIGEST.fullmatch(candidate_manifest_sha256 or ""):
        raise ValueError("Final tag candidate manifest digest is invalid")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", docker_manifest_digest):
        raise ValueError("Final tag Docker manifest digest is invalid")
    if not isinstance(release_id, int) or release_id <= 0:
        raise ValueError("Final tag release database ID is invalid")
    return {
        "schema": 1, "version": tag[1:], "commit": commit,
        "claimTag": claim["claim_tag"], "claimTagObject": claim["claim_object"],
        "autopublish": claim["autopublish"],
        "claimEpoch": claim["claim_epoch"],
        "releaseId": release_id,
        "candidateManifestSha256": candidate_manifest_sha256,
        "dockerManifestDigest": docker_manifest_digest,
        "archive": f"releases/tag/{claim['claim_tag']}/",
    }


def ensure_final_tag(tag: str, commit: str, claim: dict, *, candidate_manifest_sha256: str | None,
                     docker_manifest_digest: str, release_id: int, run=output) -> str:
    """Create or verify the immutable annotated final tag."""
    require_stable_identity(tag, commit)
    expected = _final_metadata(
        tag, commit, claim, candidate_manifest_sha256, docker_manifest_digest, release_id,
    )
    ref = f"refs/tags/{tag}"
    remote_raw = run(["git", "ls-remote", "origin", ref, f"{ref}^{{}}"])
    if not remote_raw:
        try:
            local_object = run(["git", "rev-parse", "--verify", ref])
        except subprocess.CalledProcessError:
            message = json.dumps(expected, sort_keys=True, separators=(",", ":"))
            run([
                "git", "-c", "user.name=Hermes Release Automation",
                "-c", "user.email=release-bot@users.noreply.github.com",
                "tag", "-a", tag, commit, "-m", message,
            ])
        else:
            if (run(["git", "cat-file", "-t", local_object]) != "tag"
                    or run(["git", "rev-parse", f"{ref}^{{commit}}"]) != commit
                    or json.loads(run(["git", "tag", "-l", tag, "--format=%(contents)"])) != expected):
                raise ValueError("Local final tag collision")
        run(["git", "push", "origin", ref])
        remote_raw = run(["git", "ls-remote", "origin", ref, f"{ref}^{{}}"])
    remote = dict(line.split()[::-1] for line in remote_raw.splitlines())
    tag_object, peeled = remote.get(ref), remote.get(f"{ref}^{{}}")
    if not tag_object or peeled != commit:
        raise ValueError("Final stable tag points at the wrong commit or is lightweight")
    try:
        local_object = run(["git", "rev-parse", ref])
    except subprocess.CalledProcessError:
        run(["git", "fetch", "origin", f"{ref}:{ref}"])
        local_object = run(["git", "rev-parse", ref])
    if local_object != tag_object or run(["git", "cat-file", "-t", local_object]) != "tag":
        raise ValueError("Final stable tag object differs from the verified remote")
    metadata = json.loads(run(["git", "tag", "-l", tag, "--format=%(contents)"]))
    if metadata != expected:
        raise ValueError("Final stable tag metadata differs from the accepted artifacts")
    return tag_object


def edit_draft_release(repository: str, release_id: int, tag: str, commit: str, *,
                       run=output) -> None:
    """Retarget the draft onto the receipt tag and strip the warning blocks.

    Immutable releases take no edits after publication, so every edit happens
    here while the release is still a draft, and the tag name, draft flag, and
    body are read back before anything else touches the release. A release that
    is already public is left alone: nothing can repair it.
    """
    endpoint = f"repos/{repository}/releases/{release_id}"
    current = json.loads(run(["gh", "api", endpoint]))
    if current.get("id") != release_id:
        raise ValueError("Stable draft release id changed")
    if (current.get("tag_name") == tag and current.get("draft") is False
            and current.get("prerelease") is False):
        return
    if current.get("draft") is not True:
        raise ValueError("Stable release is no longer a draft and cannot be repaired")
    body = strip_draft_warning(current.get("body") or "")
    run([
        "gh", "api", "--method", "PATCH", endpoint,
        "--raw-field", f"tag_name={tag}", "--raw-field", f"target_commitish={commit}",
        "--raw-field", "make_latest=true",
        "--field", "prerelease=false", "--field", "draft=true",
        "--raw-field", f"body={body}",
    ])
    release = json.loads(run(["gh", "api", endpoint]))
    if (release.get("id") != release_id or release.get("tag_name") != tag
            or release.get("prerelease") is not False or release.get("draft") is not True):
        raise ValueError("Stable draft retarget did not persist")
    # A fence that survives the edit — balanced or not — means the body was
    # changed underneath this call, and the release must not go public.
    if strip_draft_warning(release.get("body") or "") != body:
        raise ValueError("Stable draft body edit did not persist")


def publish_release_draft(repository: str, release_id: int, tag: str, *, run=output) -> None:
    """Make the release public as its own final call.

    Under immutable releases this is the last edit the release ever takes, so
    it runs only after the retarget and the strip have both been read back.
    """
    endpoint = f"repos/{repository}/releases/{release_id}"
    run(["gh", "api", "--method", "PATCH", endpoint, "--field", "draft=false"])
    release = json.loads(run(["gh", "api", endpoint]))
    if (release.get("id") != release_id or release.get("tag_name") != tag
            or release.get("prerelease") is not False or release.get("draft") is not False
            or not release.get("published_at")):
        raise ValueError("Stable release publication did not persist")


def publish_attempt(record: dict, *, repository: str, run=output, read_archive) -> str:
    """The one ordered publication pass, steps 1-4, each read back before the next.

    Explicit publish and autopublish converge here. The manifest digest is
    hashed from the attempt archive (nothing records it earlier), the receipt
    tag is written, the draft is retargeted and stripped while still a draft,
    and only then does the final call make it public. Returns the Docker
    manifest digest the receipt binds, for the alias move that follows.
    """
    from scripts.releases import docker

    claim = {"claim_tag": record["claim_tag"], "claim_object": record["claim_object"],
             "autopublish": record["autopublish"], "skip_bundles": record["skip_bundles"],
             "claim_epoch": record["claim_epoch"]}
    # A claim that skipped bundles staged no archive, so its receipt binds no manifest.
    manifest_sha256 = None if record["skip_bundles"] else hashlib.sha256(
        read_archive(f"releases/tag/{record['claim_tag']}/release-candidates.json")).hexdigest()
    docker_digest = docker.published_digest(record["claim_tag"], run)
    ensure_final_tag(record["tag"], record["commit"], claim,
                     candidate_manifest_sha256=manifest_sha256,
                     docker_manifest_digest=docker_digest,
                     release_id=record["release_id"], run=run)
    edit_draft_release(repository, record["release_id"], record["tag"], record["commit"], run=run)
    publish_release_draft(repository, record["release_id"], record["tag"], run=run)
    return docker_digest


def complete(env: dict) -> None:
    """Validate the accepted candidate archive. The final tag moves to publish."""
    tag, commit, claim = stable_context(env)
    if claim["skip_bundles"]:
        raise ValueError("A claim that skipped bundles has no candidate archive to validate")
    base = env["CLOUDFLARE_R2_PUBLIC_URL"].rstrip("/")
    candidate = read_candidate(env)
    validate_candidates(candidate, tag, commit, base, claim["claim_epoch"], archive=claim["claim_tag"])
    require_smokes_match_claim(candidate, skip_tests=claim["skip_tests"])


def _flag(env: dict, name: str) -> bool:
    value = env.get(name)
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be the admitted claim's true or false, not {value!r}")
    return value == "true"


def main(argv: list[str] | None = None, env: dict | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env
    if argv and argv[0] == "gate":
        needs = json.loads(env["RELEASE_NEEDS"])
        summary("\n".join(f"- {name}: {needs.get(name, {}).get('result', 'missing')}" for name in argv[1:]), env)
        require_gate(needs, argv[1:], skip_bundles=_flag(env, "SKIP_BUNDLES"),
                     skip_tests=_flag(env, "SKIP_TESTS"))
        return
    if argv and argv[0] == "stage-receipt":
        if len(argv) != 3 or argv[1] != "--receipt" or argv[2] not in RECEIPT_TARGETS:
            raise ValueError("Expected stage-receipt --receipt darwin-arm64|darwin-x64|win32-bundle")
        stage_receipt(env, argv[2])
        return
    commands = {"admit": admit, "verify": verify, "transitions": transitions,
                "candidate-manifest": candidate_manifest, "complete": complete}
    if len(argv) != 1 or argv[0] not in commands:
        raise ValueError("Expected admit, verify, gate, transitions, candidate-manifest, stage-receipt or complete")
    commands[argv[0]](env)


if __name__ == "__main__":
    main()
