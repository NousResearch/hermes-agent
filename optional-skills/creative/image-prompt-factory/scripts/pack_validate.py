"""Deterministic prompt-pack validator for the image-prompt-factory skill.

The skill's citation rule -- never stamp a provenance the corpus did not
resolve -- is written into SKILL.md, which a model follows. An instruction is
a suggestion. This validator re-checks the pack against the physical
grounding artifact (read the artifacts, never a claim about them) and exits 1
on any violation, so an invalid pack never reaches a renderer.

Pure stdlib, offline, no LLM. Reads three files from --workdir:
    brief.json            the parsed operator brief
    grounding.local.json  written by scripts/style_corpus.py ground
    prompt-pack.json      the pack the model wrote
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The literal token the pack must place in the Subject: field when the brief
# asks for subject_mode=placeholder. A renderer replaces it with the real
# subject; the pack itself never invents one.
SUBJECT_SENTINEL = "[SUBJECT SUPPLIED AT RENDER TIME]"
SUBJECT_PRESERVATION_DIRECTIVE = (
    "preserve the attached reference subject's identity, features, and proportions "
    "exactly; do not invent, describe, or restyle the subject."
)
_PLACEHOLDER_SUBJECT_VALUE = f"{SUBJECT_SENTINEL}; {SUBJECT_PRESERVATION_DIRECTIVE}"
_PROMPT_FIELDS = {
    "Use case",
    "Template",
    "Primary request",
    "Input references",
    "Scene/backdrop",
    "Subject",
    "Style/medium",
    "Composition/framing",
    "Lighting/mood",
    "Color palette",
    "Text handling",
    "Constraints",
    "Avoid",
}

# Placeholder prompts use a closed field grammar, then every non-Subject value
# is audited for subject-bearing nouns and physical-trait language. The list is
# deliberately conservative but leaves scene direction (pose, framing, lens,
# light, mood, style, palette, wardrobe and expression) available.
_PLACEHOLDER_TRAIT_LANGUAGE = re.compile(
    r"(?ix)"
    r"\b(?:woman|women|man|men|girl|girls|boy|boys|lady|ladies|gentleman|"
    r"gentlemen|person|people|human|child|children|teenagers?|adults?|male|"
    r"female)\b|"
    r"\b[\w]+-(?:eyed|haired)\b|"
    r"\b(?:black|blond|blonde|blue|brown|gray|grey|green|hazel|red|silver|"
    r"white)\s+(?:eyes|hair)\b|"
    r"\b(?:hair|eyes|skin|complexion|ethnicity|race|facial features?|nose|"
    r"lips|jawline|freckles|tattoos?|body type|build|height|weight)\b"
)

# Provenance keys that may exist ONLY on a grounded pack, and must match the
# grounding artifact byte-for-byte when they do.
_PROVENANCE_EQUAL = (
    "prompt_engine",
    "corpus_pin",
    "corpus_source",
    "corpus_sha256",
    "license",
)
_PROVENANCE_FORBIDDEN_UNGROUNDED = _PROVENANCE_EQUAL + (
    "prompt_engine_attribution",
    "example_case_ids",
)

_MAX_CONCEPTS = 8

# A pack must never carry an absolute local path (it may be shared, posted, or
# committed). The lookbehind keeps `https://` (the corpus_source URL) from
# matching as a one-letter drive: `s:` is preceded by a letter, `C:` at a path
# start is not.
_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/]|(?:^|[\s\"'(])/(?:home|root|Users)/"
)


class PackInvalid(Exception):
    """Raised with the full violation list; maps to exit 1."""

    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations))
        self.violations = violations


def _read_json(path: Path, violations: list[str]) -> "dict | None":
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        violations.append(f"missing artifact: {path.name}")
    except (OSError, json.JSONDecodeError) as exc:
        violations.append(f"unreadable artifact {path.name}: {exc}")
    else:
        if isinstance(value, dict):
            return value
        violations.append(f"{path.name} must contain a JSON object")
    return None


def _integer_array(value, label: str, violations: list[str]) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        violations.append(f"{label} must be an array of integers")
        return []
    return value


def _string_array(value, label: str, violations: list[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        violations.append(f"{label} must be an array of strings")
        return []
    return value


def _validate_grounding_shapes(grounding: dict, violations: list[str]) -> bool:
    grounded = grounding.get("grounded")
    if not isinstance(grounded, bool):
        violations.append("grounding grounded must be a boolean")
        grounded = False
    if "matched" in grounding and (
        not isinstance(grounding["matched"], int)
        or isinstance(grounding["matched"], bool)
    ):
        violations.append("grounding matched must be an integer")
    for key in ("resolved_case_ids", "unresolved_case_ids"):
        if key in grounding:
            _integer_array(grounding[key], f"grounding {key}", violations)

    exemplars = grounding.get("exemplars")
    if exemplars is not None:
        if not isinstance(exemplars, list) or any(
            not isinstance(item, dict) for item in exemplars
        ):
            violations.append("grounding exemplars must be an array of objects")
        else:
            for index, exemplar in enumerate(exemplars, start=1):
                prefix = f"grounding exemplar {index}"
                if "id" in exemplar and (
                    not isinstance(exemplar["id"], int)
                    or isinstance(exemplar["id"], bool)
                ):
                    violations.append(f"{prefix} id must be an integer")
                for key in ("title", "prompt", "category", "source_url"):
                    if key in exemplar and not isinstance(exemplar[key], str):
                        violations.append(f"{prefix} {key} must be a string")
                for key in ("styles", "scenes"):
                    if key in exemplar:
                        _string_array(exemplar[key], f"{prefix} {key}", violations)
                if "truncated" in exemplar and not isinstance(
                    exemplar["truncated"], bool
                ):
                    violations.append(f"{prefix} truncated must be a boolean")
    return grounded


def _validate_placeholder_prompt(
    text: str, concept_number: int, prompt_key: str, violations: list[str]
) -> None:
    fields: dict[str, str] = {}
    grammar_ok = True
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            violations.append(
                f"concept {concept_number}: {prompt_key} line {line_number} must use "
                "the documented '<field>: <value>' grammar"
            )
            grammar_ok = False
            continue
        field_name, value = (part.strip() for part in line.split(":", 1))
        if field_name not in _PROMPT_FIELDS:
            violations.append(
                f"concept {concept_number}: {prompt_key} has unknown prompt field "
                f"{field_name!r}"
            )
            grammar_ok = False
            continue
        if field_name in fields:
            violations.append(
                f"concept {concept_number}: {prompt_key} duplicates {field_name}: field"
            )
            grammar_ok = False
            continue
        fields[field_name] = value

    subject_value = fields.get("Subject")
    if subject_value is None or not subject_value.startswith(SUBJECT_SENTINEL):
        violations.append(
            f"concept {concept_number}: {prompt_key} Subject: field must begin with "
            f"{SUBJECT_SENTINEL}"
        )
    elif subject_value != _PLACEHOLDER_SUBJECT_VALUE:
        violations.append(
            f"concept {concept_number}: {prompt_key} Subject: field must use the "
            "canonical placeholder directive without invented traits"
        )
    if text.count(SUBJECT_SENTINEL) != 1:
        violations.append(
            f"concept {concept_number}: {prompt_key} must place the placeholder "
            "sentinel exactly once, in the Subject: field"
        )

    if grammar_ok:
        for field_name, value in fields.items():
            if field_name != "Subject" and _PLACEHOLDER_TRAIT_LANGUAGE.search(value):
                violations.append(
                    f"concept {concept_number}: {prompt_key} contains forbidden "
                    "subject identity/appearance language in "
                    f"{field_name}: field"
                )


def validate_pack(workdir: Path) -> dict:
    """Validate the pack against the brief + grounding artifacts.

    Returns the summary dict on success; raises PackInvalid listing EVERY
    violation found (not just the first -- an operator fixing a failed run
    should see the whole bill at once).
    """
    violations: list[str] = []
    brief = _read_json(workdir / "brief.json", violations)
    grounding = _read_json(workdir / "grounding.local.json", violations)
    pack = _read_json(workdir / "prompt-pack.json", violations)
    if violations:
        raise PackInvalid(violations)

    grounded = _validate_grounding_shapes(grounding, violations)

    concepts = pack.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        violations.append("pack has no concepts")
        concepts = []
    if len(concepts) > _MAX_CONCEPTS:
        violations.append(
            f"{len(concepts)} concepts exceeds the cap of {_MAX_CONCEPTS}"
        )

    declared = pack.get("prompt_count")
    if declared is not None and (
        not isinstance(declared, int) or isinstance(declared, bool)
    ):
        violations.append("pack prompt_count must be an integer")
    elif declared is not None and concepts and declared != len(concepts):
        violations.append(
            f"prompt_count says {declared} but pack has {len(concepts)} concepts"
        )

    try:
        brief_count = int(brief.get("count", 1))
    except (TypeError, ValueError):
        brief_count = 1
        violations.append(f"brief count is not a number: {brief.get('count')!r}")
    if brief_count > _MAX_CONCEPTS:
        violations.append(
            f"brief count={brief_count} exceeds the cap of {_MAX_CONCEPTS}"
        )

    placeholder = (
        str(brief.get("subject_mode", "generic")).strip().lower() == "placeholder"
    )
    for i, c in enumerate(concepts, start=1):
        if not isinstance(c, dict):
            violations.append(f"concept {i}: must be a JSON object")
            continue
        for key in ("concept_id", "template_id", "aspect", "panel_side"):
            if key in c and not isinstance(c[key], str):
                violations.append(f"concept {i}: {key} must be a string")
        for key in ("baked_prompt", "overlay_prompt"):
            text = c.get(key)
            if not isinstance(text, str) or not text.strip():
                violations.append(f"concept {i}: empty {key}")
            elif placeholder:
                _validate_placeholder_prompt(text, i, key, violations)
        copy = c.get("copy")
        if not isinstance(copy, dict):
            violations.append(f"concept {i}: missing copy object")
        elif any(not isinstance(value, str) for value in copy.values()):
            violations.append(f"concept {i}: copy values must be strings")

    if grounded:
        resolved = set(
            _integer_array(
                grounding.get("resolved_case_ids"),
                "grounding resolved_case_ids",
                violations,
            )
        )
        cited = pack.get("example_case_ids")
        if not isinstance(cited, list) or not cited:
            violations.append("grounded run but pack cites no example_case_ids")
        else:
            valid_cited = _integer_array(cited, "pack example_case_ids", violations)
            stray = [case_id for case_id in valid_cited if case_id not in resolved]
            if stray:
                violations.append(
                    f"pack cites case ids the grounding never resolved: {stray}"
                )
        for key in _PROVENANCE_EQUAL:
            if not isinstance(grounding.get(key), str):
                violations.append(f"grounding {key} must be a string")
            if not isinstance(pack.get(key), str):
                violations.append(f"pack {key} must be a string")
            if pack.get(key) != grounding.get(key):
                violations.append(
                    f"provenance mismatch on {key}: pack={pack.get(key)!r}"
                    f" grounding={grounding.get(key)!r}"
                )
    else:
        for key in _PROVENANCE_FORBIDDEN_UNGROUNDED:
            if key in pack:
                violations.append(
                    f"HOLLOW CITATION: grounded=false but pack carries {key}"
                )
        if pack.get("self_authored") is not True:
            violations.append(
                "grounded=false but pack does not declare self_authored: true"
            )

    pack_text = json.dumps(pack, ensure_ascii=False)
    if _LOCAL_PATH.search(pack_text):
        violations.append("pack text contains an absolute local path")

    if violations:
        raise PackInvalid(violations)

    return {
        "pack_valid": True,
        "concepts": len(concepts),
        "grounded": grounded,
        "subject_mode": "placeholder" if placeholder else "generic",
        "cited_case_ids": list(pack.get("example_case_ids") or []),
    }


def _cmd_validate(args) -> int:
    try:
        summary = validate_pack(Path(args.workdir))
    except PackInvalid as exc:
        for v in exc.violations:
            print(f"pack_validate: {v}", file=sys.stderr)
        print(
            json.dumps(
                {"pack_valid": False, "violations": exc.violations}, ensure_ascii=False
            )
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pack_validate", description=__doc__.splitlines()[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("validate", help="validate the pack in --workdir")
    sv.add_argument(
        "--workdir",
        required=True,
        help="directory containing brief.json, grounding.local.json, prompt-pack.json",
    )
    sv.set_defaults(func=_cmd_validate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
