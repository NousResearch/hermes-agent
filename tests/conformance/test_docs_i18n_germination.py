"""Cross-language docs germination conformance (CI gate).

Every localized root doc must reproduce the ENGLISH source's technical
graph: identical code-fence sequence, identical backtick technical
identifiers, identical link targets (locale-rewritten), resolvable internal
anchors, and — for germinated files — identical heading structure. Legacy
translations (es/zh-CN/ur-pk) run the mechanical gates; their heading debt
is reported as warnings, never silently.

The gate imports the canonical pipeline (`scripts/docs_germination.py`), the
same module the germinator CLI ships. Behavior contracts only — no
snapshots, no change-detectors: these tests fail when a localized doc drifts
from its English source, which is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from docs_germination import (  # noqa: E402
    MANIFEST,
    ROOT_DOCS,
    TARGET_LOCALE_CONTRACT,
    TARGET_LOCALES,
    check_all,
    check_doc_parity,
    extract_anchors,
    extract_code_spans,
    extract_fences,
    extract_headings,
    extract_links,
    germinate,
    locale_file,
    rewrite_target,
    slugify,
)


def _read(doc: str) -> str:
    return (REPO_ROOT / doc).read_text(encoding="utf-8")


def test_manifest_matches_versioned_non_english_target_contract():
    """One versioned tuple is the completeness authority; the manifest and
    every test consume it directly rather than repeating a drifting subset."""
    assert TARGET_LOCALE_CONTRACT == {
        "version": "ethnologue-200-29e-2026-all-users-top-10-non-english",
        "scope": "top 10 non-English documentation locales",
        "source": "Ethnologue 200",
        "edition": 29,
        "year": 2026,
        "metric": "All Users",
        "speaker_basis": "L1+L2",
        "canonical_source_locale": "en",
    }
    assert len(TARGET_LOCALES) == 10
    assert len(set(TARGET_LOCALES)) == len(TARGET_LOCALES)
    assert "en" not in TARGET_LOCALES
    assert tuple(MANIFEST) == TARGET_LOCALES
    for code in TARGET_LOCALES:
        assert code in MANIFEST, f"target locale {code} missing from manifest"

    assert {"id", "ru"} <= set(TARGET_LOCALES)
    assert MANIFEST["fr"]["status"] == "germinated"
    assert {code for code, meta in MANIFEST.items() if meta["status"] == "manual"} == {
        "zh-CN",
        "es",
        "ur-pk",
    }
    id_contract = f"{MANIFEST['id']['provenance']} {MANIFEST['id']['notes']}"
    assert "#92191" in id_contract and "#92192" in id_contract
    assert "11th" not in id_contract and "next in line" not in id_contract


def test_french_docs_are_present_and_germinated():
    for doc in ROOT_DOCS:
        p = REPO_ROOT / locale_file(doc, "fr")
        assert p.exists(), f"{p.name} missing — French must ship all three root docs"
        text = p.read_text(encoding="utf-8")
        assert len(text) > 500, f"{p.name} looks empty"
        # Germinated French must not be a verbatim copy of English.
        assert text != _read(doc), f"{p.name} is a verbatim English copy"


def test_french_passes_full_parity_gate():
    """The germinated French docs must pass every mechanical gate AND the
    heading-structure gate (germinated status gates on it)."""
    for doc in ROOT_DOCS:
        issues = check_doc_parity(
            _read(doc), _read(locale_file(doc, "fr")), doc, "fr", "germinated"
        )
        errors = [i for i in issues if i["severity"] == "error"]
        assert not errors, f"{doc}: French parity errors: {errors}"


def test_whole_gate_has_no_errors_on_any_present_locale():
    """check_all() over the whole tree. Germinated locales must pass every
    class; manual/legacy locales report ALL drift as warnings (debt visible,
    CI green) — the roadmap is to re-germinate them through the pipeline.
    Missing files count as errors only for germinated locales."""
    report = check_all(REPO_ROOT)
    fr_checks = [c for c in report["checks"] if c["locale"] == "fr"]
    assert fr_checks and all(c["result"] == "pass" for c in fr_checks), (
        f"germinated French must be clean: {fr_checks}"
    )
    for c in report["checks"]:
        if c["result"] == "missing":
            # A germinated locale must ship every root doc.
            assert c["status"] != "germinated", (
                f"{c['locale']} is germinated but {c['doc']} is missing"
            )
            continue
        if c["result"] == "fail":
            # Only germinated locales may fail; manual locales downgrade
            # every class to a warning and therefore cannot fail.
            assert c["status"] == "germinated", (
                f"{c['locale']} {c['doc']} failed but is {c['status']}: "
                f"{[i['class'] for i in c['issues']]}"
            )
        if c["status"] == "manual":
            # Manual locales never emit error-severity issues.
            assert not any(i["severity"] == "error" for i in c["issues"])


# ── extractor contracts ──────────────────────────────────────────────────────


def test_slugify_github_style():
    assert slugify("Installation rapide") == "installation-rapide"
    assert slugify("Env vars & secrets!") == "env-vars--secrets"
    assert slugify("中文 標題") == "中文-標題"


def test_extract_fences_counts_and_languages():
    text = "a\n```bash\nhermes setup\n```\nb\n```\nplain\n```\n"
    fences = extract_fences(text)
    assert [f["lang"] for f in fences] == ["bash", ""]
    assert len(fences) == 2


def test_fence_scanner_gfm_closing_rules():
    """A ```yaml line inside a plain ``` block is CODE, not a close (GFM:
    closing fences carry no info string). The regex backreference form
    closes early on it and leaks the rest of the block as prose."""
    text = "```\nkey: value\n```yaml\n# comment inside the same block\nstill code\n```\n"
    fences = extract_fences(text)
    assert len(fences) == 1, f"expected ONE fence, got {fences}"
    body_hashes = {f["body_sha256"] for f in fences}
    # The single block contains all three body lines; the hash is of the
    # COMMENT-NORMALIZED body (comments are localizable prose).
    import hashlib
    norm_body = "key: value\n```yaml\n# comment inside the same block\nstill code"
    from docs_germination import _comment_normalized  # noqa: E402
    expect = hashlib.sha256(_comment_normalized(norm_body).encode()).hexdigest()[:16]
    assert body_hashes == {expect}
    # And the ```yaml interior line must not be scanned as prose.
    assert extract_code_spans(text) == []
    assert extract_headings(text) == []


def test_fence_comments_are_localizable_commands_are_not():
    """A translated comment inside a fence passes parity; an altered
    command line fails. Comment code spans are still required."""
    en = "```bash\n# Preferred — matches CI (hermetic `env -i`)\nscripts/run_tests.sh\n```\n"
    # Comment translated to French, command byte-identical, `env -i` kept:
    loc_ok = "```bash\n# Préféré — correspond à CI (hermétique `env -i`)\nscripts/run_tests.sh\n```\n"
    issues = check_doc_parity(en, loc_ok, "README.md", "xx", "germinated")
    hard = [i for i in issues if i["class"] in ("fence_parity", "code_span_parity")]
    assert not hard, f"translated comment should pass: {hard}"
    # Command altered -> fence_parity error:
    loc_bad = "```bash\n# Préféré — correspond à CI (hermétique `env -i`)\nscripts/run_tests.sh --xdist\n```\n"
    issues = check_doc_parity(en, loc_bad, "README.md", "xx", "germinated")
    assert any(i["class"] == "fence_parity" for i in issues)
    # Comment drops `env -i` -> code_span_parity error:
    loc_missing = "```bash\n# Préféré — correspond à CI\nscripts/run_tests.sh\n```\n"
    issues = check_doc_parity(en, loc_missing, "README.md", "xx", "germinated")
    assert any(i["class"] == "code_span_parity" for i in issues)


def test_extract_code_spans_verbatim():
    text = "run `hermes model` then `~/.hermes/config.yaml`"
    assert extract_code_spans(text) == ["hermes model", "~/.hermes/config.yaml"]


def test_code_span_pairing_survives_unbalanced_backticks():
    """A dangling backtick must not swallow the next line's real pair into a
    phantom multi-line span (EN CONTRIBUTING has several)."""
    text = "see `discover_builtin_tools()` in `tools/registry.py` when `model_tools`\n"
    assert extract_code_spans(text) == [
        "discover_builtin_tools()",
        "tools/registry.py",
        "model_tools",
    ]
    unbalanced = "a `unterminated\nb `c` d\n"
    spans = extract_code_spans(unbalanced)
    assert all("\n" not in s for s in spans)
    assert "unterminated" not in spans


def test_extract_links_markdown_and_href():
    text = '[docs](https://x.test/) and <a href="README.es.md">ES</a>'
    assert extract_links(text) == [("docs", "https://x.test/"), ("README.es.md", "README.es.md")]


def test_extract_headings_levels_and_anchors():
    text = "# Title\n\n## Sub\n\n### Sub-sub\n"
    assert extract_headings(text) == [(1, "Title"), (2, "Sub"), (3, "Sub-sub")]
    assert extract_anchors(text) == {"title", "sub", "sub-sub"}


def test_rewrite_target_locale_twin():
    assert rewrite_target("CONTRIBUTING.md", "README.md", "fr") == "CONTRIBUTING.fr.md"
    assert rewrite_target("CONTRIBUTING.md#code-of-conduct", "README.md", "fr") == (
        "CONTRIBUTING.fr.md#code-of-conduct"
    )
    assert rewrite_target("assets/banner.png", "README.md", "fr") == "assets/banner.png"
    assert rewrite_target("https://x.test/", "README.md", "fr") == "https://x.test/"


def test_parity_gate_detects_fence_drift():
    en = "Intro\n\n```bash\nhermes setup\n```\n"
    loc = "Intro\n\n```bash\nhermes setup --portal\n```\n"
    issues = check_doc_parity(en, loc, "README.md", "xx", "germinated")
    classes = {i["class"] for i in issues}
    assert "fence_parity" in classes


def test_parity_gate_detects_missing_code_spans():
    en = "Use `hermes model --provider x` today.\n"
    loc = "Use the model command today.\n"
    issues = check_doc_parity(en, loc, "README.md", "xx", "germinated")
    assert any(i["class"] == "code_span_parity" for i in issues)


def test_parity_gate_detects_dangling_fragment():
    en = "[home](#)\n\n# Real heading\n"
    # Locale has the heading but links to a fragment that doesn't exist.
    loc = "[missing](#does-not-exist)\n\n# Real heading\n"
    issues = check_doc_parity(en, loc, "README.md", "xx", "germinated")
    assert any(i["class"] == "anchor_parity" for i in issues)


def test_parity_gate_heading_drift_is_warning_for_legacy():
    en = "# A\n\n## B\n"
    loc = "# A\n"
    issues = check_doc_parity(en, loc, "README.md", "xx", "manual")
    h = [i for i in issues if i["class"] == "heading_parity"]
    assert h and h[0]["severity"] == "warning"


def test_legacy_mechanical_drift_is_warning_not_error():
    """Manual-status locales downgrade EVERY drift class to a warning —
    the legacy debt is visible in CI without blocking it."""
    en = "Use `hermes model` today.\n\n```bash\nhermes setup\n```\n"
    loc = "Use the model command today.\n"
    issues = check_doc_parity(en, loc, "README.md", "xx", "manual")
    assert issues, "expected drift to be reported"
    assert all(i["severity"] == "warning" for i in issues)


def test_germinated_heading_drift_is_error():
    en = "# A\n\n## B\n"
    loc = "# A\n"
    issues = check_doc_parity(en, loc, "README.md", "xx", "germinated")
    h = [i for i in issues if i["class"] == "heading_parity"]
    assert h and h[0]["severity"] == "error"


@pytest.mark.parametrize("doc", ROOT_DOCS)
def test_english_sources_have_no_missing_twin_targets(doc):
    """Sanity: the English docs' internal links to other root docs must
    resolve (the parity gate's rewrite logic is only meaningful if the
    English graph itself is closed)."""
    text = _read(doc)
    anchors = extract_anchors(text)
    for _, t in extract_links(text):
        if t.startswith("#") and t[1:]:
            assert t[1:] in anchors, f"{doc}: dangling fragment {t}"


# ── germination runner (mocked LLM — no network in CI) ──────────────────────


def _fake_llm_that_translates_prose(monkeypatch, translation_map):
    """The germinate action pipes the prompt to the LLM command and treats
    stdout as the translation. This fake verifies the prompt carried the
    template, and returns a translation whose prose lines are the marker
    content suffixed with the locale (a crude 'translation')."""
    import subprocess

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, input, capture_output, text, encoding, timeout):
        captured["cmd"] = cmd
        captured["prompt"] = input
        # Simulate an LLM that returns ONLY the translated document (it
        # drops the instruction preamble and the end marker) and keeps
        # everything but prose markers, which it 'translates' by suffixing
        # the locale code.
        doc_lines = input.split("=== DOCUMENT START ===")[1].split(
            "=== DOCUMENT END ==="
        )[0].splitlines()
        out_lines = []
        for line in doc_lines:
            if line.startswith("⟪"):
                # ⟪fr:Some prose⟫ -> "Some prose [fr]"
                inner = line.split(":", 1)[1].rsplit("⟫", 1)[0]
                out_lines.append(f"{inner} [{translation_map.get(inner, 'fr')}]")
            else:
                out_lines.append(line)
        FakeProc.stdout = "\n".join(out_lines)
        return FakeProc

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_germinate_writes_file_and_runs_gate(monkeypatch, tmp_path):
    """germinate() writes the LLM output to the locale file and returns the
    gate verdict. A translation that keeps every technical edge passes; the
    prompt must carry the template with prose placeholders."""
    import subprocess

    captured = _fake_llm_that_translates_prose(monkeypatch, {})
    src = _read("README.md")
    target, out, issues = germinate("fr", "README.md", ["fake-llm"], out_dir=tmp_path)
    assert target == tmp_path / "README.fr.md"
    assert target.exists()
    # The prompt carried the template (markers present).
    assert "⟪fr:" in captured["prompt"]
    # Prose markers got replaced, technical edges preserved.
    assert "⟪fr:" not in out
    assert "hermes model" in out  # code span survived the LLM
    # The gate ran on the output: prose-only translation keeps every edge.
    errors = [i for i in issues if i["severity"] == "error"]
    assert not errors, f"gate errors: {errors}"


def test_germinate_rejects_llm_that_drops_code_spans(monkeypatch, tmp_path):
    """A translator that drops a backtick identifier fails the gate — the
    action reports the drift instead of shipping it."""

    def evil_run(cmd, input, capture_output, text, encoding, timeout):
        class P:
            returncode = 0
            stdout = input.replace("`hermes model`", "the model command")
            stderr = ""

        return P()

    import subprocess

    monkeypatch.setattr(subprocess, "run", evil_run)
    _, out, issues = germinate("fr", "README.md", ["evil-llm"], out_dir=tmp_path)
    assert "the model command" in out
    classes = {i["class"] for i in issues}
    assert "code_span_parity" in classes


def test_germinate_fails_loudly_on_llm_error(monkeypatch, tmp_path):
    import subprocess

    class FailProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FailProc())
    with pytest.raises(RuntimeError, match="LLM command failed"):
        germinate("fr", "README.md", ["broken-llm"], out_dir=tmp_path)
