"""RED acceptance suite (red-team) for the warn-only kanban config form checks.

Contract SSOT: ``.autopilot/runtime/sessions/forge-doctor-kanban-config-forms/requirements/
20260919-forge-卡-t_232c31fd-—-在/state.md`` §``### 契约规约`` + §``## 验收场景``.

Contract under test (design intent only — implementation does not exist yet, which is exactly
why this suite is red):

- ``hermes_cli.doctor_config.collect_kanban_config_findings(raw_config)`` → ``list[tuple[str, str, str]]``
  of ``(key_path, warn_text, detail)``. Pure function: no I/O, no input mutation, findings never
  become blocking issues.
- Module-level closed-set constants ``_KANBAN_INT_SETTINGS`` / ``_KANBAN_BOOL_SETTINGS`` (values and
  order pinned by the contract).
- Input-shape dispatch: non-dict raw config → ``[]``; missing/``None`` kanban section → ``[]``;
  non-mapping kanban section → exactly 1 section-level finding (``key_path == "kanban"``, warn_text
  says "mapping") with all per-key checks skipped; mapping kanban → per-key checks for closed-set
  keys that EXIST (out-of-set keys are never looked at).
- int-key form buckets (bool checked BEFORE int — ``isinstance(True, int)`` is true): dict/list and
  non-numeric strings are not coercible → detail must contain ``"silently ignored"``; ``true``/``"3"``/
  ``1.5`` coerce via ``int()`` → detail must contain ``"int("`` and must NOT contain ``"ignored"``;
  ``false``/``0``/negatives parse but violate the ``>= 1`` floor → fallback-to-default wording;
  valid positive ints / missing keys / explicit nulls → zero findings.
- bool-key form buckets: any string value → detail must contain the literal ``bool("false") is True``
  (intent reversal); explicit null and numeric values get a finding; real bools / missing keys → zero.
- Doctor wiring: drift step ``_drift_kanban_settings(f, should_fix, config_path)`` is warn-only
  (``fixed == 0 and issues == [] and manual_issues == []`` always), silent when there are no
  findings, and sits in ``_CONFIG_DRIFT_STEPS`` immediately after ``_drift_structure``.

Deliberate import discipline: the new attributes are only touched inside test bodies, so a missing
implementation surfaces as per-case AttributeError/FAIL while this file still collects.
"""

from __future__ import annotations

import contextlib
import copy
import io

import pytest

from hermes_cli import doctor_config
from hermes_cli.config import read_user_config_raw
from hermes_cli.doctor_report import Finding

# Pinned verbatim by the contract: any STRING bool-slot value must produce a detail containing this
# exact literal (straight double quotes) — runtime bool() treats every non-empty string as True.
BOOL_INTENT_REVERSAL = 'bool("false") is True'

INT_KEY_PATHS = ("kanban.max_in_progress", "kanban.max_in_progress_per_profile")
BOOL_KEY_PATHS = ("kanban.auto_decompose", "kanban.auto_subscribe_on_create")

# The real incident shape (P4 live-probe form): a mapping where an int is expected (runtime
# int(mapping) raises → silently falls back) next to a string where a bool is expected
# (bool("false") is True → the user tried to switch it OFF and it stays ON).
_INCIDENT_YAML = (
    "kanban:\n"
    "  max_in_progress_per_profile:\n"
    "    coder: 1\n"
    "  auto_decompose: \"false\"\n"
)


def _write_config(tmp_path, yaml_text):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    return cfg


def _collect_raw(raw_config):
    """Single funnel to the contract entry point (AttributeError here = per-case red, by design)."""
    return doctor_config.collect_kanban_config_findings(raw_config)


def _collect_from_yaml(tmp_path, yaml_text):
    """Feed the collector exactly the way the drift step does: raw on-disk YAML, no defaults."""
    return _collect_raw(read_user_config_raw(_write_config(tmp_path, yaml_text)))


def _key_paths(findings):
    return [key_path for key_path, _warn_text, _detail in findings]


def _findings_for(findings, key_path):
    return [f for f in findings if f[0] == key_path]


class TestKanbanFindingsContract:
    """Signature, purity and closed-set constant contracts of the collector."""

    def test_module_constants_are_the_pinned_closed_sets(self):
        assert doctor_config._KANBAN_INT_SETTINGS == ("max_in_progress", "max_in_progress_per_profile"), (
            f"int closed set must match the contract verbatim, got {doctor_config._KANBAN_INT_SETTINGS!r}")
        assert doctor_config._KANBAN_BOOL_SETTINGS == ("auto_decompose", "auto_subscribe_on_create"), (
            f"bool closed set must match the contract verbatim, got {doctor_config._KANBAN_BOOL_SETTINGS!r}")

    @pytest.mark.parametrize("raw_config", [None, [], "kanban", 7, 3.5, True],
                             ids=["none", "list", "str", "int", "float", "bool"])
    def test_non_dict_raw_config_yields_no_findings(self, raw_config):
        assert _collect_raw(raw_config) == [], (
            f"non-dict raw config {raw_config!r} must short-circuit to [] (got findings)")

    def test_missing_kanban_key_yields_no_findings(self):
        assert _collect_raw({}) == []
        assert _collect_raw({"model": {"name": "m"}, "display": {}}) == []

    @pytest.mark.parametrize("yaml_text", ["kanban:\n", "kanban: null\n"], ids=["bare", "explicit-null"])
    def test_null_kanban_section_yields_no_findings(self, tmp_path, yaml_text):
        assert _collect_from_yaml(tmp_path, yaml_text) == [], (
            "a null kanban section is 'unset by design' and must produce zero findings")

    def test_finding_is_a_three_string_tuple(self, tmp_path):
        findings = _collect_from_yaml(tmp_path, _INCIDENT_YAML)
        assert findings, "incident YAML must produce findings for this arity check to mean anything"
        for finding in findings:
            assert isinstance(finding, tuple) and len(finding) == 3, (
                f"each finding must be a (key_path, warn_text, detail) tuple, got {finding!r}")
            assert all(isinstance(part, str) for part in finding), (
                f"every finding part must be str, got {finding!r}")

    def test_collector_is_pure_and_does_not_mutate_input(self):
        raw = {"kanban": {"max_in_progress": {"coder": 1}, "auto_decompose": "false", "max_spawn": 2}}
        snapshot = copy.deepcopy(raw)
        first = _collect_raw(raw)
        second = _collect_raw(raw)
        assert raw == snapshot, f"collector must not mutate its input, got {raw!r}"
        assert first == second, "repeated collection of the same input must be deterministic"
        assert first, "the bad forms in this input must actually produce findings"

    def test_warn_text_always_carries_its_key_path(self, tmp_path):
        yaml_text = (
            "kanban:\n"
            "  max_in_progress: \"abc\"\n"
            "  auto_subscribe_on_create: 0\n"
        )
        findings = _collect_from_yaml(tmp_path, yaml_text)
        assert len(findings) == 2, f"expected one finding per bad key, got {findings}"
        for key_path, warn_text, _detail in findings:
            assert key_path in warn_text, (
                f"warn_text {warn_text!r} must contain its key_path {key_path!r} verbatim")


class TestKanbanSectionLevelShape:
    """A kanban section that is not a mapping yields exactly one section-level finding and skips
    every per-key check."""

    @pytest.mark.parametrize("yaml_text", [
        "kanban: \"max_in_progress\"\n",
        "kanban: 7\n",
        "kanban: 1.5\n",
        "kanban: true\n",
        "kanban: [1, 2]\n",
    ], ids=["str", "int", "float", "bool", "list"])
    def test_non_mapping_kanban_section_yields_exactly_one_section_finding(self, tmp_path, yaml_text):
        findings = _collect_from_yaml(tmp_path, yaml_text)
        assert len(findings) == 1, (
            f"non-mapping kanban section must yield exactly 1 finding (per-key checks skipped), got {findings}")
        key_path, warn_text, _detail = findings[0]
        assert key_path == "kanban", (
            f"section-level finding key_path must be exactly 'kanban', got {key_path!r} in {findings}")
        assert "mapping" in warn_text, (
            f"section-level warn_text must contain the word 'mapping', got {warn_text!r}")
        assert "kanban" in warn_text, (
            f"section-level warn_text must name the kanban section, got {warn_text!r}")


class TestKanbanIntSettingForms:
    """int-slot form buckets for kanban.max_in_progress / kanban.max_in_progress_per_profile."""

    @pytest.mark.parametrize("yaml_body,key_path", [
        ("  max_in_progress_per_profile:\n    coder: 1\n", "kanban.max_in_progress_per_profile"),
        ("  max_in_progress: [1, 2]\n", "kanban.max_in_progress"),
        ("  max_in_progress: \"\"\n", "kanban.max_in_progress"),
        ("  max_in_progress_per_profile: not-a-number\n", "kanban.max_in_progress_per_profile"),
    ], ids=["dict", "list", "empty-string", "non-numeric-string"])
    def test_uncoercible_int_form_detail_says_silently_ignored(self, tmp_path, yaml_body, key_path):
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        matching = _findings_for(findings, key_path)
        assert len(matching) == 1, f"expected exactly 1 finding for {key_path}, got {findings}"
        assert len(findings) == 1, f"no other key may be flagged, got {findings}"
        detail = matching[0][2]
        assert "silently ignored" in detail, (
            f"{key_path}: uncoercible form means int() raises and the value is silently ignored — "
            f"detail must contain 'silently ignored', got {detail!r}")

    @pytest.mark.parametrize("yaml_body,key_path", [
        ("  max_in_progress: true\n", "kanban.max_in_progress"),
        ("  max_in_progress_per_profile: \"3\"\n", "kanban.max_in_progress_per_profile"),
        ("  max_in_progress: 1.5\n", "kanban.max_in_progress"),
    ], ids=["bool-true", "numeric-string", "float"])
    def test_coercible_int_form_detail_mentions_int_coercion_and_not_ignored(self, tmp_path, yaml_body, key_path):
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        matching = _findings_for(findings, key_path)
        assert len(matching) == 1, (
            f"{key_path}: coercible-but-suspicious form must still produce exactly 1 finding, got {findings}")
        detail = matching[0][2]
        assert "int(" in detail, (
            f"{key_path}: detail must name the int() coercion, got {detail!r}")
        assert "ignored" not in detail, (
            f"{key_path}: the value coerces fine today (nothing is ignored) — detail must NOT contain "
            f"'ignored', got {detail!r}")

    def test_bool_false_in_int_slot_reports_fallback_not_ignored(self, tmp_path):
        # CONTRACT_AMBIGUOUS: the contract row for `false` pins only "detail explains the fallback"
        # (its sibling out-of-range row pins "falls back to the default"); the assertions below keep
        # the non-controversial parts (finding exists, NOT the ignored bucket, fallback mentioned).
        findings = _collect_from_yaml(tmp_path, "kanban:\n  max_in_progress: false\n")
        matching = _findings_for(findings, "kanban.max_in_progress")
        assert len(matching) == 1, (
            f"kanban.max_in_progress: int(False) == 0 violates the >=1 floor and must be flagged, got {findings}")
        detail = matching[0][2]
        assert "silently ignored" not in detail, (
            f"kanban.max_in_progress: false falls back to the default, it is not silently ignored — "
            f"got {detail!r}")
        assert "default" in detail.lower(), (
            f"kanban.max_in_progress: detail must explain the fallback to the default, got {detail!r}")

    @pytest.mark.parametrize("yaml_body", ["  max_in_progress: 0\n", "  max_in_progress_per_profile: -4\n"],
                             ids=["zero", "negative"])
    def test_out_of_range_int_detail_says_falls_back_to_the_default(self, tmp_path, yaml_body):
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        assert len(findings) == 1, f"out-of-range int must produce exactly 1 finding, got {findings}"
        key_path, _warn_text, detail = findings[0]
        assert key_path in INT_KEY_PATHS, f"unexpected key flagged: {key_path!r}"
        assert "falls back to the default" in detail, (
            f"{key_path}: contract pins the wording 'falls back to the default' for the >=1 guard, "
            f"got {detail!r}")
        assert "silently ignored" not in detail, (
            f"{key_path}: out-of-range values are a fallback, not silently ignored — got {detail!r}")

    @pytest.mark.parametrize("yaml_body", ["  max_in_progress: 1\n", "  max_in_progress_per_profile: 3\n"],
                             ids=["one", "three"])
    def test_valid_positive_int_is_silent(self, tmp_path, yaml_body):
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        assert findings == [], f"legal positive int must produce zero findings, got {findings}"

    def test_explicit_null_int_value_is_silent(self, tmp_path):
        findings = _collect_from_yaml(
            tmp_path, "kanban:\n  max_in_progress:\n  max_in_progress_per_profile:\n")
        assert findings == [], (
            f"explicit null int keys are 'unset by design' and must produce zero findings, got {findings}")

    def test_int_keys_absent_from_kanban_mapping_is_silent(self, tmp_path):
        findings = _collect_from_yaml(tmp_path, "kanban:\n  auto_decompose: true\n")
        assert findings == [], (
            f"int keys that do not exist must not be checked (key-existence rule), got {findings}")


class TestKanbanBoolSettingForms:
    """bool-slot form buckets for kanban.auto_decompose / kanban.auto_subscribe_on_create."""

    @pytest.mark.parametrize("yaml_body,key_path", [
        ("  auto_decompose: \"false\"\n", "kanban.auto_decompose"),
        ("  auto_subscribe_on_create: \"true\"\n", "kanban.auto_subscribe_on_create"),
        ("  auto_decompose: \"\"\n", "kanban.auto_decompose"),
    ], ids=["string-false", "string-true", "empty-string"])
    def test_string_bool_detail_carries_intent_reversal_literal(self, tmp_path, yaml_body, key_path):
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        matching = _findings_for(findings, key_path)
        assert len(matching) == 1, f"expected exactly 1 finding for {key_path}, got {findings}"
        detail = matching[0][2]
        assert BOOL_INTENT_REVERSAL in detail, (
            f"{key_path}: any string in a bool slot must be explained with the pinned literal "
            f"{BOOL_INTENT_REVERSAL!r} (non-empty strings are truthy = intent reversal), got {detail!r}")

    @pytest.mark.parametrize("yaml_body", [
        "  auto_decompose: 0\n",
        "  auto_decompose: 1\n",
        "  auto_subscribe_on_create: 2\n",
    ], ids=["zero", "one", "two"])
    def test_numeric_bool_value_gets_a_finding(self, tmp_path, yaml_body):
        # CONTRACT_AMBIGUOUS: the contract pins "detail explains bool() swallows numbers and a real
        # bool should be used" but no verbatim substring — asserted as finding-presence + key_path
        # contract only.
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        assert len(findings) == 1, f"numeric value in a bool slot must produce exactly 1 finding, got {findings}"
        key_path, warn_text, detail = findings[0]
        assert key_path in BOOL_KEY_PATHS, f"unexpected key flagged: {key_path!r}"
        assert key_path in warn_text, f"warn_text {warn_text!r} must contain {key_path!r}"
        assert detail != "", f"{key_path}: numeric-bool finding must carry an explanatory detail"

    @pytest.mark.parametrize("yaml_body,key_path", [
        ("  auto_decompose:\n", "kanban.auto_decompose"),
        ("  auto_subscribe_on_create:\n", "kanban.auto_subscribe_on_create"),
    ], ids=["auto-decompose", "auto-subscribe"])
    def test_explicit_null_bool_value_gets_a_finding(self, tmp_path, yaml_body, key_path):
        # CONTRACT_AMBIGUOUS: detail wording (explicit null reads falsy=off while a missing key
        # defaults to on) is unpinned — asserted as finding-presence only; the discriminator vs the
        # int slot (explicit null there = zero findings) is covered in TestKanbanIntSettingForms.
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        matching = _findings_for(findings, key_path)
        assert len(matching) == 1, (
            f"{key_path}: explicit null is NOT the int-slot unset-by-design case — bool slot must flag "
            f"the missing-vs-null semantics, got {findings}")
        assert matching[0][2] != "", f"{key_path}: null-bool finding must carry an explanatory detail"

    @pytest.mark.parametrize("yaml_body", [
        "  auto_decompose: true\n",
        "  auto_subscribe_on_create: false\n",
    ], ids=["true", "false"])
    def test_real_bool_value_is_silent(self, tmp_path, yaml_body):
        findings = _collect_from_yaml(tmp_path, f"kanban:\n{yaml_body}")
        assert findings == [], (
            f"a real bool value is the supported form and must produce zero findings, got {findings}")

    def test_bool_keys_absent_from_kanban_mapping_is_silent(self, tmp_path):
        findings = _collect_from_yaml(tmp_path, "kanban:\n  max_in_progress: 2\n")
        assert findings == [], (
            f"bool keys that do not exist must not be checked (key-existence rule), got {findings}")


class TestKanbanClosedSetBoundary:
    """Keys outside the two closed-set enums are never looked at, whatever their shape."""

    def test_out_of_set_keys_are_never_checked(self, tmp_path):
        yaml_text = (
            "kanban:\n"
            "  max_spawn:\n"
            "    nested: 1\n"
            "  default_assignee: [1, 2]\n"
            "  auto_retry: \"false\"\n"
            "  unknown_setting: 0\n"
        )
        findings = _collect_from_yaml(tmp_path, yaml_text)
        assert findings == [], (
            f"keys outside the closed sets must be ignored in any shape, got {findings}")

    def test_mixed_closed_and_out_of_set_flags_only_the_closed_set(self, tmp_path):
        yaml_text = (
            "kanban:\n"
            "  max_spawn: {a: 1}\n"
            "  max_in_progress: \"abc\"\n"
        )
        findings = _collect_from_yaml(tmp_path, yaml_text)
        assert _key_paths(findings) == ["kanban.max_in_progress"], (
            f"only the closed-set key may be flagged, got {findings}")

    def test_empty_kanban_mapping_is_silent(self, tmp_path):
        findings = _collect_from_yaml(tmp_path, "kanban: {}\n")
        assert findings == [], (
            f"an empty kanban mapping has no closed-set keys to check, got {findings}")


class TestKanbanIncidentShape:
    """The real incident form end to end: a mapping in an int slot next to a string bool."""

    def test_incident_config_collects_exactly_two_findings(self, tmp_path):
        findings = _collect_from_yaml(tmp_path, _INCIDENT_YAML)
        assert len(findings) == 2, f"the incident shape must produce exactly 2 findings, got {findings}"
        # CONTRACT_AMBIGUOUS: the contract does not pin the order of findings within the list —
        # compared as a set of key_paths.
        assert sorted(_key_paths(findings)) == ["kanban.auto_decompose", "kanban.max_in_progress_per_profile"], (
            f"the two flagged keys must be the int-slot mapping and the string bool, got {findings}")
        int_detail = _findings_for(findings, "kanban.max_in_progress_per_profile")[0][2]
        assert "silently ignored" in int_detail, (
            f"int-slot mapping detail must say 'silently ignored', got {int_detail!r}")
        bool_detail = _findings_for(findings, "kanban.auto_decompose")[0][2]
        assert BOOL_INTENT_REVERSAL in bool_detail, (
            f"string-bool detail must contain {BOOL_INTENT_REVERSAL!r}, got {bool_detail!r}")

    def test_incident_config_drift_step_emits_exactly_two_warns(self, tmp_path):
        cfg = _write_config(tmp_path, _INCIDENT_YAML)
        finding = Finding()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_config._drift_kanban_settings(finding, False, cfg)
        out = buf.getvalue()
        assert "Kanban Settings" in out, f"drift step must open a 'Kanban Settings' section, got {out!r}"
        assert out.count("⚠") == 2, f"the incident shape must emit exactly 2 warn lines, got {out!r}"
        assert "kanban.max_in_progress_per_profile" in out, f"int key_path missing from output: {out!r}"
        assert "kanban.auto_decompose" in out, f"bool key_path missing from output: {out!r}"


class TestDriftKanbanSettingsWarnOnly:
    """_drift_kanban_settings mirrors _drift_legacy_custom_providers: raw-file read, warn-only,
    silent when clean."""

    def _run(self, tmp_path, yaml_text, should_fix=False):
        cfg = _write_config(tmp_path, yaml_text)
        finding = Finding()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_config._drift_kanban_settings(finding, should_fix, cfg)
        return buf.getvalue(), finding

    def test_bad_form_opens_section_and_emits_warn_line(self, tmp_path):
        yaml_text = "kanban:\n  max_in_progress: \"abc\"\n"
        out, finding = self._run(tmp_path, yaml_text)
        assert "Kanban Settings" in out, f"expected the section banner, got {out!r}"
        assert out.count("⚠") == 1, f"expected exactly 1 warn line, got {out!r}"
        assert "kanban.max_in_progress" in out, f"warn line must carry the key_path, got {out!r}"
        assert finding.fixed == 0 and finding.issues == [] and finding.manual_issues == [], (
            f"warn-only step must not touch Finding: fixed={finding.fixed} issues={finding.issues} "
            f"manual_issues={finding.manual_issues}")

    def test_drift_output_renders_every_collected_finding(self, tmp_path):
        """Chain consistency: every (key_path, warn_text, detail) the collector returns for a config
        file is rendered by the drift step for that same file, with the detail in parentheses."""
        cfg = _write_config(tmp_path, _INCIDENT_YAML)
        findings = _collect_raw(read_user_config_raw(cfg))
        assert len(findings) == 2, f"incident YAML must collect exactly 2 findings, got {findings}"
        finding = Finding()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_config._drift_kanban_settings(finding, False, cfg)
        out = buf.getvalue()
        for key_path, warn_text, detail in findings:
            assert key_path in out, f"{key_path}: key_path missing from drift output: {out!r}"
            assert warn_text in out, f"{key_path}: warn_text {warn_text!r} missing from drift output"
            # Collector details arrive pre-parenthesized; the step passes them to check_warn verbatim
            # (single parens, matching _drift_legacy_custom_providers). Lock full verbatim rendering.
            assert detail in out, f"{key_path}: detail {detail!r} missing from drift output: {out!r}"

    def test_findings_never_become_blocking_issues_or_fixes(self, tmp_path):
        out, finding = self._run(tmp_path, _INCIDENT_YAML)
        assert out, "bad kanban forms must surface as doctor warn output"
        assert finding.fixed == 0 and finding.issues == [] and finding.manual_issues == [], (
            f"kanban config findings are warn-only — Finding must stay untouched even with findings: "
            f"fixed={finding.fixed} issues={finding.issues} manual_issues={finding.manual_issues}")

    def test_warn_only_holds_under_should_fix(self, tmp_path):
        out, finding = self._run(tmp_path, _INCIDENT_YAML, should_fix=True)
        assert "Kanban Settings" in out and out.count("⚠") == 2, (
            f"--fix must not silence the kanban warns, got {out!r}")
        assert finding.fixed == 0 and finding.issues == [] and finding.manual_issues == [], (
            f"warn-only must hold under should_fix=True: fixed={finding.fixed} issues={finding.issues} "
            f"manual_issues={finding.manual_issues}")

    @pytest.mark.parametrize("yaml_text", [
        "model:\n  name: m\n",
        "kanban:\n",
        "kanban:\n  max_in_progress: 2\n  auto_decompose: true\n",
        "kanban:\n  max_spawn: 3\n  default_assignee: ralph\n  max_in_progress: 2\n  auto_decompose: true\n",
    ], ids=["no-kanban-key", "null-kanban", "clean-kanban", "out-of-set-plus-clean"])
    def test_silent_on_clean_configs(self, tmp_path, yaml_text):
        out, finding = self._run(tmp_path, yaml_text)
        assert out == "", f"clean config must stay fully silent (no section banner either), got {out!r}"
        assert finding.fixed == 0 and finding.issues == [] and finding.manual_issues == [], (
            f"clean config must not touch Finding: fixed={finding.fixed} issues={finding.issues} "
            f"manual_issues={finding.manual_issues}")

    def test_silent_when_config_file_is_missing(self, tmp_path):
        cfg = tmp_path / "absent.yaml"
        finding = Finding()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_config._drift_kanban_settings(finding, False, cfg)
        assert buf.getvalue() == "", "a missing config file must stay silent (read_user_config_raw → {})"
        assert finding.fixed == 0 and finding.issues == [] and finding.manual_issues == []


class TestConfigDriftStepsWiring:
    """_drift_kanban_settings is registered in _CONFIG_DRIFT_STEPS right after _drift_structure."""

    def test_kanban_drift_step_is_registered(self):
        assert doctor_config._drift_kanban_settings in doctor_config._CONFIG_DRIFT_STEPS, (
            "_drift_kanban_settings must be a member of _CONFIG_DRIFT_STEPS")

    def test_kanban_drift_step_immediately_follows_structure_step(self):
        steps = doctor_config._CONFIG_DRIFT_STEPS
        assert steps.index(doctor_config._drift_kanban_settings) == steps.index(doctor_config._drift_structure) + 1, (
            f"_drift_kanban_settings must come immediately after _drift_structure, got {steps}")

    def test_kanban_drift_step_precedes_legacy_custom_providers_step(self):
        steps = doctor_config._CONFIG_DRIFT_STEPS
        assert steps.index(doctor_config._drift_kanban_settings) < steps.index(doctor_config._drift_legacy_custom_providers), (
            f"_drift_kanban_settings must be inserted before _drift_legacy_custom_providers, got {steps}")

    def test_config_drift_docstring_mentions_kanban_setting_forms(self):
        doc = doctor_config._check_config_drift.__doc__ or ""
        assert "kanban" in doc.lower(), (
            f"_check_config_drift docstring must gain a kanban setting forms sentence, got {doc!r}")
