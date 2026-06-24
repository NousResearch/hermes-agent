"""Skill frontmatter validator + vetting helpers.

Adds a strict ``validate_skill_frontmatter(path)`` function used by the skill
loader to reject malformed SKILL.md files before they reach the gateway
startup path.  Lives in ``hermes_cli`` (not ``agent``) so the validator can
be reused by the ``hermes skills`` CLI subcommand, the curator, and the
prompt-builder snapshot loader without creating a circular import.

The contract:

* ``validate_skill_frontmatter(path) -> (ok: bool, errors: list[str])``
* Returns ``(True, [])`` for a fully valid SKILL.md.
* Returns ``(False, [...])`` with one actionable error per problem found
  (``missing required field 'name'``, ``status 'production' is not one of
  {draft, vetted, deprecated}``, ``version '1.0' is not valid semver
  (expected MAJOR.MINOR.PATCH)``, ``name 'foo' does not match parent
  directory 'bar'``).
* Never raises — file-system and YAML parse errors are caught and reported
  in the returned ``errors`` list so callers can aggregate.

Required fields: ``name``, ``version``, ``status``, ``description``,
``author``.  ``status`` must be one of ``draft`` / ``vetted`` /
``deprecated``.  ``version`` must match MAJOR.MINOR.PATCH semver (per the
official semver.org regex).  ``name`` must match the parent directory name
so a skill at ``.../skills/<name>/SKILL.md`` cannot silently rename itself
in the metadata.

Vetting (t_8a86fc9c)
-------------------
The ``vetted_at`` (ISO-8601) and ``vetted_by`` (string) frontmatter fields
record that a human (or ``wags-reviewer``) has passed a SKILL.md through
all four validators — frontmatter, security, content-quality, and
link/file-existence.  Both default to ``"unvetted"`` and the
``hermes skills vet`` subcommand stamps them only when every validator
passes.  ``is_vetted(frontmatter)`` is the canonical predicate used by
``hermes skills list --vetted/--unvetted`` and by the
``--require-vetting`` install flag.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Constants ──────────────────────────────────────────────────────────────

REQUIRED_FIELDS: Tuple[str, ...] = (
    "name",
    "version",
    "status",
    "description",
    "author",
)

ALLOWED_STATUSES: Tuple[str, ...] = ("draft", "vetted", "deprecated")

# Sentinel value for an unvetted skill.  Both ``vetted_at`` and
# ``vetted_by`` carry this string when the skill has never been vetted.
UNVETTED: str = "unvetted"

# A vetted skill's ``vetted_by`` field must be a non-empty string that is
# not the ``UNVETTED`` sentinel and does not contain whitespace, control
# characters, or shell metacharacters.  This is a deliberately tight
# allow-list because the value is rendered in audit logs and used in
# filename suggestions for backup paths — keeping the surface narrow
# avoids quoting hell and injection risks.
_VETTED_BY_RE = re.compile(r"^[A-Za-z0-9_.@:+-]{1,64}$")

# Official semver.org regular expression (slightly trimmed — we don't accept
# build metadata with leading ``+`` without separators, matching the strict
# reading on https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string).
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*)?"
    r"(?:\+[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?$"
)


# ── Local fallback frontmatter parser ──────────────────────────────────────


def _local_parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Minimal YAML-frontmatter splitter used when agent.skill_utils is unavailable.

    Splits on ``---\\n`` markers at the top of the file.  Returns
    ``({}, content)`` if no markers are present (matching the agent helper's
    fallback semantics).  Body parsing is intentionally dumb — we use a
    plain ``key: value`` split because the validator only needs scalar
    fields; structured YAML can wait for the agent helper.
    """
    frontmatter: Dict[str, Any] = {}
    if not content.startswith("---"):
        return frontmatter, content

    # Find the closing "\n---" (allow trailing whitespace) on a fresh line.
    end_match = re.search(r"\n---\s*(?:\n|$)", content[3:])
    if not end_match:
        return frontmatter, content

    yaml_block = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    for raw_line in yaml_block.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Strip matching quotes so ``name: "foo-bar"`` reads as ``foo-bar``.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        frontmatter.setdefault(key, value)

    return frontmatter, body


# ── Validation helpers ─────────────────────────────────────────────────────


def _is_semver(value: str) -> bool:
    return bool(_SEMVER_RE.match(value))


def _validate_field_present(errors: List[str], frontmatter: Dict[str, Any], field: str) -> Any:
    """Append a missing-field error and return ``None``; return value otherwise.

    Tracks whether the field is actually absent vs. present-but-empty so the
    error message is specific (``missing`` vs. ``empty``).
    """
    if field not in frontmatter:
        errors.append(f"missing required field '{field}'")
        return None
    value = frontmatter[field]
    # Allow ``None`` explicitly (YAML literal ``null``); treat everything
    # else that is "empty" as missing.
    if value is None or (isinstance(value, str) and not value.strip()):
        errors.append(f"missing required field '{field}' (empty value)")
        return None
    return value


# ── Public API ─────────────────────────────────────────────────────────────


def parse_skill_frontmatter(path: Path) -> Tuple[Dict[str, Any], str]:
    """Parse the YAML frontmatter block at the top of a SKILL.md file.

    Prefers ``agent.skill_utils.parse_frontmatter`` (full YAML support
    including nested ``metadata:`` blocks); falls back to the local
    splitter when the agent package isn't importable.  The agent helper
    is lazy-imported here so this module can be imported without
    triggering a hermes_constants cold-start chain.
    """
    try:  # pragma: no cover - exercised in integration, not unit tests
        from agent.skill_utils import parse_frontmatter as _agent_parse_frontmatter
    except Exception:  # pragma: no cover
        _agent_parse_frontmatter = None

    content = Path(path).read_text(encoding="utf-8")
    if _agent_parse_frontmatter is not None:
        return _agent_parse_frontmatter(content)
    return _local_parse_frontmatter(content)


def validate_skill_frontmatter(
    path: Path | str,
) -> Tuple[bool, List[str]]:
    """Validate a SKILL.md file's frontmatter.

    Parameters
    ----------
    path:
        Filesystem path to the SKILL.md file.  Strings are accepted for
        convenience but coerced to ``pathlib.Path``.

    Returns
    -------
    (ok, errors)
        ``ok`` is ``True`` iff ``errors`` is empty.  ``errors`` is a list
        of human-readable strings — each one names exactly one problem so
        callers can show the user a checklist of fixes.
    """
    errors: List[str] = []
    skill_path = Path(path)

    # 1. File exists and is readable.
    if not skill_path.is_file():
        return False, [f"SKILL.md not found at {skill_path}"]

    # 2. Parse frontmatter.  YAML errors are reported, not raised.
    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read {skill_path}: {exc}"]

    try:
        frontmatter, _body = parse_skill_frontmatter(skill_path)
    except Exception as exc:  # pragma: no cover - defensive
        return False, [f"frontmatter parse error in {skill_path}: {exc}"]

    if not frontmatter:
        errors.append(
            "no YAML frontmatter block found "
            "(expected '---' on line 1 and a closing '---' line)"
        )
        # Cannot validate field-level rules without parsed metadata.
        return False, errors

    # 3. Required-field presence checks.
    name = _validate_field_present(errors, frontmatter, "name")
    version = _validate_field_present(errors, frontmatter, "version")
    status = _validate_field_present(errors, frontmatter, "status")
    _validate_field_present(errors, frontmatter, "description")
    _validate_field_present(errors, frontmatter, "author")

    # 4. Field-level rules — only run when the field is present so we
    #    don't double-report "missing X" AND "X is invalid".
    if status is not None:
        status_str = str(status).strip().lower()
        if status_str not in ALLOWED_STATUSES:
            allowed = ", ".join(ALLOWED_STATUSES)
            errors.append(
                f"status {status_str!r} is not one of {{{allowed}}}"
            )

    if version is not None:
        version_str = str(version).strip()
        if not _is_semver(version_str):
            errors.append(
                f"version {version_str!r} is not valid semver "
                f"(expected MAJOR.MINOR.PATCH, e.g. '1.2.3' or '0.1.0-rc.1')"
            )

    # 5. name ↔ parent directory consistency.
    if name is not None:
        name_str = str(name).strip()
        parent_dir_name = skill_path.parent.name
        if name_str != parent_dir_name:
            errors.append(
                f"name {name_str!r} does not match parent directory "
                f"{parent_dir_name!r} (skills must live in skills/<name>/)"
            )

    return (len(errors) == 0), errors


def validate_all_skills(
    skills_root: Path | str,
) -> Dict[str, Tuple[bool, List[str]]]:
    """Run :func:`validate_skill_frontmatter` against every SKILL.md under ``skills_root``.

    Walks one level deep (``skills_root/<name>/SKILL.md``) — matches the
    current Hermes skills layout.  Nested-category skills
    (``skills_root/<category>/<name>/SKILL.md``) are also walked because
    `iter_skill_index_files` in ``agent.skill_utils`` does the same.

    Returns a dict keyed by the absolute path of the SKILL.md file.
    """
    root = Path(skills_root)
    results: Dict[str, Tuple[bool, List[str]]] = {}
    if not root.is_dir():
        return results

    for skill_md in sorted(root.rglob("SKILL.md")):
        results[str(skill_md)] = validate_skill_frontmatter(skill_md)
    return results


def validate_or_warn(
    skill_path: Path | str,
    *,
    logger: Any = None,
) -> bool:
    """Validate a SKILL.md and log a warning on failure.

    Convenience wrapper for the agent loader: returns ``True`` iff the
    frontmatter is valid.  On failure, logs each error message at WARNING
    level (or via ``print`` if no logger is provided) and returns ``False``
    so the caller can skip auto-loading the skill.

    This function never raises — it's the wire-in point that keeps the
    gateway startup crash-free even when skills are malformed.

    Parameters
    ----------
    skill_path:
        Filesystem path to the SKILL.md file.
    logger:
        Optional logger instance.  If ``None``, falls back to the module
        logger (``hermes_cli.skill_loader``) and finally to ``print`` if
        the module logger has no handlers.  Pass ``logging.getLogger()``
        from your caller to keep log records attributed correctly.
    """
    ok, errors = validate_skill_frontmatter(skill_path)
    if ok:
        return True

    # Lazy logger resolution — avoids forcing every caller to construct one.
    if logger is None:
        import logging

        logger = logging.getLogger("hermes_cli.skill_loader")

    for err in errors:
        logger.warning("skill frontmatter validation failed for %s: %s", skill_path, err)
    return False


# ── Vetting predicates + helpers ──────────────────────────────────────────


def is_vetted(frontmatter: Dict[str, Any]) -> bool:
    """Return True iff the frontmatter marks this skill as vetted.

    A skill is vetted when both ``vetted_at`` and ``vetted_by`` are
    present, non-empty, and ``vetted_by`` is *not* the ``UNVETTED``
    sentinel.  The presence of ``vetted_at`` alone is not sufficient —
    every vetting stamp must carry a reviewer identity.

    Accepts the raw ``frontmatter`` dict from
    :func:`parse_skill_frontmatter` (values are typically strings after
    YAML scalar parsing).
    """
    if not frontmatter:
        return False
    vetted_at = frontmatter.get("vetted_at")
    vetted_by = frontmatter.get("vetted_by")
    if not vetted_at or not isinstance(vetted_at, str):
        return False
    if not vetted_by or not isinstance(vetted_by, str):
        return False
    by = vetted_by.strip()
    if not by or by == UNVETTED:
        return False
    return True


def vetting_state(frontmatter: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{vetted_at, vetted_by}`` as strings, with the unvetted
    sentinel filled in when the fields are absent.

    Always returns a 2-key dict so callers can render a row without
    checking for key absence.  Use :func:`is_vetted` for the boolean
    decision — this helper is for *display only*.
    """
    if not frontmatter:
        return {"vetted_at": UNVETTED, "vetted_by": UNVETTED}
    return {
        "vetted_at": str(frontmatter.get("vetted_at") or UNVETTED),
        "vetted_by": str(frontmatter.get("vetted_by") or UNVETTED),
    }


def parse_vetted_at(frontmatter: Dict[str, Any]) -> _dt.datetime | None:
    """Parse the ``vetted_at`` ISO-8601 timestamp from frontmatter.

    Returns ``None`` when the field is missing, the ``UNVETTED``
    sentinel, or fails to parse.  Accepts both
    ``2026-06-23T18:30:00Z`` and ``2026-06-23T18:30:00+00:00`` shapes.
    """
    raw = frontmatter.get("vetted_at") if frontmatter else None
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw or raw == UNVETTED:
        return None
    # Normalize trailing Z to +00:00 so fromisoformat accepts it.
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return _dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _now_iso_utc() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix.

    Matches the form rendered in audit logs and emitted by other Hermes
    subsystems (``agent.skill_utils`` etc.) so vetting timestamps align
    with the rest of the platform.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_vetted_by(value: Any) -> str | None:
    """Return an error string if ``value`` is not a usable reviewer id,
    ``None`` when valid.  Internal helper for the CLI vetting flow.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return "vetted_by must be a non-empty string (e.g. 'wags-reviewer' or 'human:charlie')"
    s = str(value).strip()
    if s == UNVETTED:
        return "vetted_by cannot be the 'unvetted' sentinel; pass --by <reviewer>"
    if not _VETTED_BY_RE.match(s):
        return (
            f"vetted_by {s!r} contains characters outside [A-Za-z0-9_.@:+-] "
            "or is longer than 64 characters"
        )
    return None


# ── Content-quality validator ─────────────────────────────────────────────
#
# Conceptually the 4th of the four vetting validators: frontmatter
# (validate_skill_frontmatter), security (tools.skills_guard.scan_skill),
# content-quality (this function), and link/file-existence
# (validate_skill_links_files below).
#
# The bar is intentionally low — a SKILL.md that passes the frontmatter
# and security checks is *probably* loadable, but content-quality
# failures are still useful signals (a 3-line SKILL.md, or a body with
# no headings, is almost certainly not a useful skill for an LLM).


def validate_skill_content_quality(
    path: Path | str,
) -> Tuple[bool, List[str]]:
    """Linter-style checks on SKILL.md body content.

    Rules enforced:

    * ``description`` is at least 16 characters (catches placeholders
      like ``"TODO"`` and ``"fill this in"``).
    * body length is at least 200 characters (a single-sentence SKILL.md
      does not give the LLM enough context to act on).
    * body contains at least one markdown heading (``#`` at line start
      or ``##``/``###``/etc.) — skills should be skimmable.
    * body does not contain ``<script>`` tags or ``javascript:`` URLs
      (defence-in-depth on top of the security scanner; this catches
      shape that ``scan_skill`` may not flag if the scanner treats
      ``javascript:`` as a flag-only finding).

    Returns ``(ok, errors)`` in the same shape as
    :func:`validate_skill_frontmatter`.  Never raises.
    """
    errors: List[str] = []
    skill_path = Path(path)
    if not skill_path.is_file():
        return False, [f"SKILL.md not found at {skill_path}"]
    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read {skill_path}: {exc}"]

    # Re-use the local frontmatter splitter so the description field is
    # available without forcing a YAML dependency.  agent.skill_utils
    # may not be importable in the test environment, and the
    # description is the only field we need.
    frontmatter, body = _local_parse_frontmatter(content)

    desc = (frontmatter.get("description") or "").strip()
    if len(desc) < 16:
        errors.append(
            f"description is too short ({len(desc)} chars; "
            "minimum 16) — placeholder descriptions like 'TODO' are not allowed"
        )

    body_clean = body.strip()
    if len(body_clean) < 200:
        errors.append(
            f"body is too short ({len(body_clean)} chars; minimum 200) — "
            "skills need enough context for the LLM to act on"
        )

    if not any(re.match(r"^#{1,6}\s+\S", line) for line in body_clean.splitlines()):
        errors.append(
            "body has no markdown headings — skills should be skimmable "
            "(add at least one '#', '##', etc. line)"
        )

    # Defence-in-depth: the security scanner flags prompt-injection
    # phrases; this catches raw HTML / JS that may slip through if the
    # scanner is downgraded to flag-only in a future change.  We check
    # only the *executable* shapes (script tags, link/iframe/script
    # src/href targets) — bare prose mentions of "javascript:" inside
    # an explanatory paragraph are fine.
    lower = body_clean.lower()
    if re.search(r"<\s*script\b", lower):
        errors.append("body contains a <script> tag — disallowed in SKILL.md")
    # javascript:/data:/vbscript: URLs only when used as a link target
    # or in an HTML attribute (src=, href=, action=, etc.).
    if re.search(
        r"(?:\]\(|href\s*=|src\s*=|action\s*=|formaction\s*=)\s*[\"']?"
        r"(?:javascript|data|vbscript)\s*:",
        lower,
    ):
        errors.append(
            "body contains an executable URL (javascript:/data:/vbscript:) "
            "used as a link or HTML attribute — disallowed in SKILL.md"
        )

    return (len(errors) == 0), errors


# ── Link / file existence validator ───────────────────────────────────────


def validate_skill_links_files(
    path: Path | str,
    *,
    root: Path | str | None = None,
) -> Tuple[bool, List[str]]:
    """Check that files referenced from the SKILL.md body exist on disk.

    Captures two classes of broken reference:

    * **Markdown links** to relative files — ``[text](relative/path.md)``
      whose target is missing from the skill directory.
    * **Markdown image references** to relative files — same shape, just
      a different role.

    External links (``http://``, ``https://``) are NOT checked — we don't
    want a vetting pass to depend on network reachability.  Anchors
    (``#section-name``) and absolute paths are ignored.

    Parameters
    ----------
    path:
        Path to the SKILL.md file.
    root:
        Optional base directory used to resolve relative references.  When
        ``None`` (the default), uses ``path.parent`` — the skill's own
        directory, which is the natural interpretation for a
        ``[text](other.md)`` link in a SKILL.md.

    Returns ``(ok, errors)`` in the same shape as
    :func:`validate_skill_frontmatter`.  Never raises.
    """
    errors: List[str] = []
    skill_path = Path(path)
    if not skill_path.is_file():
        return False, [f"SKILL.md not found at {skill_path}"]
    base_dir = Path(root) if root is not None else skill_path.parent

    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read {skill_path}: {exc}"]

    # Strip fenced code blocks so ``[link](file.md)`` in a code sample
    # doesn't trip the check.
    no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

    # Markdown link / image references.  We capture only the URL part
    # in the second group; alt/title text is the first group and is
    # ignored.
    link_re = re.compile(r"!?\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
    for match in link_re.finditer(no_code):
        url = match.group(1).strip()
        # Skip non-file targets.
        if not url or url.startswith(("#", "http://", "https://", "mailto:")):
            continue
        # Resolve relative to the skill directory.
        target = (base_dir / url).resolve()
        if not target.exists():
            # Try a few common alt-spellings so a vetted skill doesn't
            # fail on a typo that one rewording would fix.
            candidates = [target]
            if target.suffix == "":
                candidates.append(target.with_suffix(".md"))
                candidates.append(target.with_suffix(".txt"))
            if not any(p.exists() for p in candidates):
                # Compute the line number for the error message.
                line_no = no_code[: match.start()].count("\n") + 1
                errors.append(
                    f"body line {line_no}: link target {url!r} not found "
                    f"under {base_dir}"
                )

    return (len(errors) == 0), errors


# ── Vetting orchestrator ──────────────────────────────────────────────────


def stamp_skill_vetted(
    path: Path | str,
    *,
    reviewer: str,
    when: str | None = None,
) -> str:
    """Write ``vetted_at`` + ``vetted_by`` into the SKILL.md frontmatter.

    Idempotent — calling twice with the same reviewer overwrites the
    previous stamp with a fresh ``vetted_at``.  The original file is
    overwritten in place; callers that need a backup should make one
    first (the CLI does not pre-backup because the operation is meant
    to be cheap and re-runnable).

    Parameters
    ----------
    path:
        SKILL.md file to stamp.
    reviewer:
        Non-empty string matching ``_VETTED_BY_RE``.  Raises
        :class:`ValueError` on a bad value.
    when:
        ISO-8601 timestamp string.  ``None`` means "now" (UTC, ``Z``
        suffix).

    Returns
    -------
    The timestamp that was written (always a string), so the caller
    can log it in an audit row without re-reading the file.

    Raises
    ------
    ValueError
        When ``reviewer`` is empty, the ``UNVETTED`` sentinel, or
        fails the character allow-list.
    OSError
        When the file cannot be read or written.
    """
    skill_path = Path(path)
    if not skill_path.is_file():
        raise OSError(f"SKILL.md not found at {skill_path}")

    err = _validate_vetted_by(reviewer)
    if err is not None:
        raise ValueError(err)

    timestamp = when or _now_iso_utc()
    content = skill_path.read_text(encoding="utf-8")

    # Re-write the frontmatter block: split the file, drop any existing
    # vetted_at / vetted_by lines, append the new ones at the END of
    # the frontmatter (so a freshly-stamped skill's two lines land in a
    # predictable place, regardless of the original ordering).
    if not content.startswith("---"):
        raise ValueError(
            f"{skill_path} has no YAML frontmatter block; cannot stamp vetted fields"
        )
    end_match = re.search(r"\n---\s*(?:\n|$)", content[3:])
    if not end_match:
        raise ValueError(
            f"{skill_path} has an unterminated YAML frontmatter block"
        )
    yaml_block = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    # Drop any existing vetted_at / vetted_by lines (case-insensitive
    # on the key — YAML is case-sensitive but we want to be forgiving
    # about hand-written skills that used different casing).
    cleaned_lines: List[str] = []
    for raw_line in yaml_block.splitlines():
        key = raw_line.split(":", 1)[0].strip().lower()
        if key in ("vetted_at", "vetted_by"):
            continue
        cleaned_lines.append(raw_line)
    # The yaml_block starts with a leading newline (the regex match
    # anchored on the closing ``\n---\n`` so the captured block opens
    # with the newline that preceded the first YAML key).  Strip it
    # so the rewrite doesn't end up with ``---\n\nname: ...``.
    yaml_new = "\n".join(cleaned_lines).lstrip("\n").rstrip("\n")
    # Quoting the timestamp protects against YAML interpreting a
    # date-shaped string as a date object.
    yaml_new += f'\nvetted_at: "{timestamp}"\nvetted_by: "{reviewer}"'

    # Ensure exactly one blank line between the closing ``---`` and the
    # body so the resulting file is well-formed.  Some original files
    # put body content on the very next line with no separator and
    # ``parse_skill_frontmatter`` (or downstream YAML) would still
    # tolerate it, but a consistent separator is friendlier to humans
    # diffing the file.
    if body.startswith("\n"):
        body = "\n" + body.lstrip("\n")
    else:
        body = "\n" + body

    new_content = f"---\n{yaml_new}\n---{body}"
    skill_path.write_text(new_content, encoding="utf-8")
    return timestamp


def run_vet(
    path: Path | str,
    *,
    reviewer: str,
    when: str | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run all four vetting validators and (on success) stamp the skill.

    Returns a dict with ``ok``, ``validators`` (per-validator result),
    ``stamped`` (whether the file was written), and ``timestamp`` /
    ``reviewer`` when a stamp happened.  Never raises — file errors and
    bad reviewer strings are folded into the result so the CLI can show
    a checklist of failures.
    """
    skill_path = Path(path)
    out: Dict[str, Any] = {
        "path": str(skill_path),
        "ok": False,
        "validators": {},
        "stamped": False,
        "timestamp": None,
        "reviewer": reviewer,
    }

    # Validator 1: frontmatter.
    fm_ok, fm_errors = validate_skill_frontmatter(skill_path)
    out["validators"]["frontmatter"] = {"ok": fm_ok, "errors": fm_errors}

    # Validator 2: security.  Best-effort: the scanner lives in
    # ``tools.skills_guard`` and is lazy-imported so a missing module
    # (e.g. test environment) surfaces as a "skipped" entry rather than
    # a crash.  We treat any verdict other than "safe" as a failure to
    # be conservative — caution and dangerous both gate the stamp.
    try:
        from tools.skills_guard import scan_skill
        scan_result = scan_skill(skill_path, source="local-vetting")
        # Verdict contract: "safe" | "caution" | "dangerous".  Only
        # "safe" passes the vetting gate.  Listing the rule id of every
        # block-severity finding gives the reviewer a checklist.
        sec_ok = scan_result.verdict == "safe"
        sec_errors: List[str] = []
        if not sec_ok:
            for finding in scan_result.findings:
                if finding.severity in ("critical", "high"):
                    sec_errors.append(
                        f"{finding.pattern_id} (line {finding.line}): "
                        f"{finding.description}"
                    )
            if not sec_errors:
                sec_errors = [f"security verdict: {scan_result.verdict}"]
        out["validators"]["security"] = {
            "ok": sec_ok,
            "errors": sec_errors,
            "verdict": scan_result.verdict,
        }
    except Exception as exc:  # pragma: no cover - best-effort
        out["validators"]["security"] = {
            "ok": False,
            "errors": [f"security scan failed: {exc}"],
            "verdict": "skipped",
        }

    # Validator 3: content quality.
    cq_ok, cq_errors = validate_skill_content_quality(skill_path)
    out["validators"]["content_quality"] = {"ok": cq_ok, "errors": cq_errors}

    # Validator 4: link / file existence.
    lf_ok, lf_errors = validate_skill_links_files(skill_path)
    out["validators"]["links_files"] = {"ok": lf_ok, "errors": lf_errors}

    all_ok = all(v["ok"] for v in out["validators"].values())
    out["ok"] = all_ok
    if not all_ok:
        return out

    if dry_run:
        out["stamped"] = False
        out["timestamp"] = when or _now_iso_utc()
        return out

    try:
        ts = stamp_skill_vetted(skill_path, reviewer=reviewer, when=when)
    except (ValueError, OSError) as exc:
        # A bad reviewer string or a write failure shouldn't happen here
        # because the helpers already validated, but guard against
        # permission / disk errors so the orchestrator never raises.
        out["validators"]["stamp"] = {"ok": False, "errors": [str(exc)]}
        out["ok"] = False
        return out

    out["stamped"] = True
    out["timestamp"] = ts
    return out


__all__ = [
    "REQUIRED_FIELDS",
    "ALLOWED_STATUSES",
    "UNVETTED",
    "parse_skill_frontmatter",
    "validate_skill_frontmatter",
    "validate_all_skills",
    "validate_or_warn",
    "is_vetted",
    "vetting_state",
    "parse_vetted_at",
    "validate_skill_content_quality",
    "validate_skill_links_files",
    "stamp_skill_vetted",
    "run_vet",
]