"""Stable release admission, signed-package transitions and final release receipt."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit

from scripts.releases.semver import STABLE_TAG
SHA = re.compile(r"[a-f0-9]{40}")
DIGEST = re.compile(r"[a-f0-9]{64}")
DESKTOP_TARGETS = ("windows/x64", "windows/arm64", "macos/x64", "macos/arm64")


def require_stable_identity(tag: str, commit: str, ref: str) -> None:
    if not isinstance(tag, str) or not STABLE_TAG.fullmatch(tag) or not SHA.fullmatch(commit or "") or ref != f"refs/tags/{tag}":
        raise ValueError("Stable release must run on its exact stable tag and commit")


def require_success(needs: dict, required: list[str]) -> None:
    if not required or len(set(required)) != len(required):
        raise ValueError("Invalid required-job list")
    failures = [f"{name}={needs.get(name, {}).get('result', 'missing')}"
                for name in required if needs.get(name, {}).get("result") != "success"]
    if failures:
        raise ValueError("Release blocked: " + ", ".join(failures))


def validate_candidates(manifest: dict, tag: str, commit: str, public_base: str) -> dict:
    require_stable_identity(tag, commit, f"refs/tags/{tag}")
    if manifest.get("schema") != 1 or manifest.get("tag") != tag or manifest.get("commit") != commit or not isinstance(manifest.get("packages"), list):
        raise ValueError("Candidate manifest does not match release identity")
    prefix = urlsplit(f"{public_base.rstrip('/')}/releases/tag/{tag}/")
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
            windows_version(item.get("version", ""))
            if item["version"] != f"{tag[1:]}.0":
                raise ValueError("Stable Windows package version must match its release tag")
            if not item.get("publisher") or not item.get("applicationId") or not url.path.endswith(".msixbundle"):
                raise ValueError("Windows candidate needs publisher, applicationId and MSIX bundle")
        elif item["platform"] == "macos":
            if item.get("version") != tag[1:] or not re.fullmatch(r"[A-Z0-9]{10}", item.get("teamId", "")) or not url.path.endswith(".zip"):
                raise ValueError("macOS candidate needs matching version, signing team and app ZIP")
        rows[target] = item
    if any(target not in rows for target in DESKTOP_TARGETS):
        raise ValueError("Candidate manifest must cover Windows and macOS on both architectures")
    return rows


def windows_version(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value):
        raise ValueError("Windows package version must have four numeric components")
    result = tuple(map(int, value.split(".")))
    if any(n > 65535 for n in result):
        raise ValueError("Windows package version exceeds 16 bits")
    return result


def plan_transitions(previous: dict, candidate: dict, public_base: str) -> list[dict]:
    old = validate_candidates(previous, previous.get("tag"), previous.get("commit"), public_base)
    new = validate_candidates(candidate, candidate.get("tag"), candidate.get("commit"), public_base)
    result = []
    for target in DESKTOP_TARGETS:
        left, right = old[target], new[target]
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
        result.append({"target": target.replace("/", "-"), "transition": {
            "schema": 1, "platform": right["platform"], "arch": right["arch"], "old": left, "new": right,
        }})
    return result


def read_manifest(url: str, expected_hash: str | None = None, *, expected_origin: str | None = None,
                  opener=urllib.request.urlopen) -> dict:
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


def check_tag(env: dict, run=output) -> tuple[str, str]:
    tag, commit = env.get("RELEASE_TAG"), env.get("GITHUB_SHA")
    require_stable_identity(tag, commit, env.get("GITHUB_REF"))
    actual = run(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"])
    remote = dict(line.split()[::-1] for line in run(["git", "ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"] ).splitlines())
    remote_commit = remote.get(f"refs/tags/{tag}^{{}}", remote.get(f"refs/tags/{tag}"))
    if actual != commit or remote_commit != commit or run(["git", "rev-parse", "HEAD"]) != commit:
        raise ValueError("Release tag or checkout moved")
    run(["git", "fetch", "origin", "main"])
    run(["git", "merge-base", "--is-ancestor", commit, "origin/main"])
    return tag, commit


def emit(values: dict, env: dict) -> None:
    with Path(env["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as file:
        for key, value in values.items():
            file.write(f"{key}={value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))}\n")


def read_candidate(env: dict) -> dict:
    digest = env.get("CANDIDATE_MANIFEST_SHA256", "")
    if not DIGEST.fullmatch(digest):
        raise ValueError("Pinned candidate manifest digest is required")
    return read_manifest(env["CANDIDATE_MANIFEST_URL"], digest)


def summary(text: str, env: dict) -> None:
    with Path(env["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def admit(env: dict) -> None:
    tag, commit = check_tag(env)
    with Path("pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    if f"v{version}" != tag:
        raise ValueError("Stable tag must match the project version")
    release = json.loads(output(["gh", "release", "view", tag, "--repo", env["GITHUB_REPOSITORY"], "--json", "tagName,isDraft,isPrerelease"]))
    if release["tagName"] != tag or not release["isDraft"] or release["isPrerelease"]:
        raise ValueError("Stable candidate must have a non-prerelease draft")
    emit({"tag": tag, "commit": commit}, env)
    summary(f"## Stable candidate {tag}\nCommit: {commit}\n\nDesktop Playwright E2E: deferred by owner, not passed.\nOSV findings retain the existing advisory policy.", env)


def transitions(env: dict) -> None:
    from scripts.releases.r2 import put

    tag, commit = check_tag(env)
    base = env["CLOUDFLARE_R2_PUBLIC_URL"].rstrip("/")
    candidate = read_candidate(env)
    validate_candidates(candidate, tag, commit, base)
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
    matrices = {"windows": {"include": []}, "macos": {"include": []}}
    for row in plan_transitions(previous, candidate, base):
        transition = row["transition"]
        name = f"acceptance-{row['target']}.json"
        file = Path(env["RUNNER_TEMP"]) / name
        file.write_text(json.dumps(transition), encoding="utf-8")
        put(tag=tag, key=name, file=file, immutable=True)
        url = f"{base}/releases/tag/{tag}/{name}"
        if read_manifest(url) != transition:
            raise ValueError("Transition manifest read-back mismatch")
        matrices[transition["platform"]]["include"].append({"arch": transition["arch"], "manifest": url, "old": transition["old"]["tag"], "id": row["target"], "manifest_sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    emit(matrices, env)


def complete(env: dict) -> None:
    from scripts.releases.r2 import put

    tag, commit = check_tag(env)
    base = env["CLOUDFLARE_R2_PUBLIC_URL"].rstrip("/")
    candidate = read_candidate(env)
    validate_candidates(candidate, tag, commit, base)
    file = Path(env["RUNNER_TEMP"]) / "release-candidates.json"
    file.write_text(json.dumps(candidate), encoding="utf-8")
    put(tag=tag, key="releases/stable/release-candidates.json", key_is_full=True, file=file)
    if read_manifest(f"{base}/releases/stable/release-candidates.json") != candidate:
        raise ValueError("Stable manifest read-back mismatch")
    output(["gh", "release", "edit", tag, "--repo", env["GITHUB_REPOSITORY"], "--draft=false"])
    release = json.loads(output(["gh", "release", "view", tag, "--repo", env["GITHUB_REPOSITORY"], "--json", "isDraft"]))
    if release["isDraft"]:
        raise ValueError("Stable release remained a draft")


def main(argv: list[str] | None = None, env: dict | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env
    if argv and argv[0] == "gate":
        needs = json.loads(env["RELEASE_NEEDS"])
        summary("\n".join(f"- {name}: {needs.get(name, {}).get('result', 'missing')}" for name in argv[1:]), env)
        require_success(needs, argv[1:])
        return
    commands = {"admit": admit, "transitions": transitions, "complete": complete}
    if len(argv) != 1 or argv[0] not in commands:
        raise ValueError("Expected admit, gate, transitions or complete")
    commands[argv[0]](env)


if __name__ == "__main__":
    main()
