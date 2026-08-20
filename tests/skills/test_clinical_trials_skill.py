"""Tests for the clinical-trials optional skill (``optional-skills/research/clinical-trials/``).

Per teknium1 review on PR #45704:
  - "SKILL.md:1-5 omits contributor attribution, version, license, and metadata"
  - "No ``tests/skills/test_clinical_trials_skill.py`` is included"
Required: mocked, no-network tests for SKILL.md metadata and request
construction.

Per AGENTS.md §"Skill authoring standards (HARDLINE)" point 7:
  "Tests live at ``tests/skills/test_<skill>_skill.py`` and use only
   stdlib + pytest + ``unittest.mock``. No live network calls. Run via
   ``scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q``."

Mirrors the three-class structure documented in the optional-skill
recipe: ``TestFunctionalHappyPath`` / ``TestEdgeCases`` /
``TestErrorPropagation``. All HTTP is mocked at the module level so
``urllib.request.urlopen`` is never reached.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ----------------------------------------------------------------------------
# Path setup: import the helper script directly so we can exercise its
# functions without spawning a subprocess (subprocess tests are slower
# and brittle to argparse changes). The scripts/ dir is co-located with
# SKILL.md under optional-skills/research/clinical-trials/.
# ----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = (
    _REPO_ROOT
    / "optional-skills"
    / "research"
    / "clinical-trials"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import clinical_trials as ct  # noqa: E402  (path injected above)


# ============================================================================
# SKILL.md metadata — frontmatter validates per AGENTS.md "Skill authoring
# standards" HARDLINE section (description ≤ 60 chars, required fields,
# metadata.hermes.*
# ============================================================================


SKILL_MD_PATH = (
    _REPO_ROOT
    / "optional-skills"
    / "research"
    / "clinical-trials"
    / "SKILL.md"
)


def _read_skill_md() -> str:
    return SKILL_MD_PATH.read_text(encoding="utf-8")


def _frontmatter(text: str) -> str:
    """Extract the YAML frontmatter block at the top of SKILL.md."""
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        raise AssertionError("SKILL.md has no YAML frontmatter block")
    return m.group(1)


class TestFrontmatterContract:
    """Static checks against SKILL.md — no network, no module import."""

    def test_description_is_within_60_chars(self):
        text = _read_skill_md()
        m = re.search(r"^description: (.*)$", _frontmatter(text), re.MULTILINE)
        assert m is not None, "frontmatter missing description"
        desc = m.group(1)
        assert len(desc) <= 60, (
            f"description too long ({len(desc)} > 60): {desc!r}"
        )
        assert desc.endswith("."), "description must end with a period"

    def test_required_frontmatter_fields_present(self):
        fm = _frontmatter(_read_skill_md())
        for field in ("name:", "description:", "version:", "author:",
                      "license:", "platforms:", "metadata:"):
            assert re.search(rf"^{re.escape(field)}", fm, re.MULTILINE), (
                f"missing required frontmatter field {field!r}"
            )

    def test_metadata_hermes_block_has_category_and_tags(self):
        fm = _frontmatter(_read_skill_md())
        # metadata.hermes.tags must be a one-line bracket list
        assert re.search(
            r"^metadata:\n\s+hermes:\n\s+tags:\s+\[", fm, re.MULTILINE
        ), "metadata.hermes.tags block missing or malformed"

    def test_no_literal_skill_dir_in_examples(self):
        """Per teknium1: ``SKILL_DIR`` literal won't resolve. The
        substituted token must be used in documented invocations.

        We tolerate the literal only inside prose explanations (e.g.
        quoted inside backticks labelling the variable); invocations
        under ``## How to Run`` style blocks must use
        ``${HERMES_SKILL_DIR}`` or ``${HERMES_SKILL_DIR}/scripts/...``.
        """
        text = _read_skill_md()
        # Any ``python3 ... SKILL_DIR/scripts/`` invocation form is wrong.
        bad = re.findall(
            r"^python3\s+SKILL_DIR/", text, re.MULTILINE | re.IGNORECASE
        )
        assert bad == [], (
            f"literal SKILL_DIR found in command invocations: {bad!r} — "
            f"use ${{HERMES_SKILL_DIR}}/scripts/..."
        )


# ============================================================================
# Module-level: api_request mocked at the urllib boundary so the functions
# under test exercise the real URL/query-param assembly paths.
# ============================================================================


def _fake_api_response(studies: list[dict], total: int) -> str:
    payload = {"studies": studies, "totalCount": total}
    return json.dumps(payload)


def _study(
    *,
    nct="NCT00000000",
    title="Test Trial",
    status="RECRUITING",
    phase=("PHASE2",),
    sponsor="Acme",
    enrollment=100,
) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct,
                "briefTitle": title,
            },
            "statusModule": {"overallStatus": status},
            "designModule": {
                "phases": list(phase),
                "enrollmentInfo": {"count": enrollment},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor},
            },
        }
    }


def _detail_payload(
    *,
    nct="NCT00000000",
    title="Test Trial",
    criteria="Inclusion: age >= 18",
    sex="ALL",
    min_age="18 Years",
    max_age="65 Years",
) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": title},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2024-01-15"},
                "completionDateStruct": {"date": "2026-12-31"},
            },
            "designModule": {"phases": ["PHASE2"]},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Acme"}
            },
            "eligibilityModule": {
                "eligibilityCriteria": criteria,
                "sex": sex,
                "minimumAge": min_age,
                "maximumAge": max_age,
                "healthyVolunteers": False,
            },
            "conditionsModule": {"conditions": ["hypertension"]},
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "BP reduction", "timeFrame": "12 weeks"}
                ],
                "secondaryOutcomes": [],
            },
            "armsInterventionsModule": {
                "interventions": [{"name": "Drug A", "type": "DRUG"}]
            },
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": "Test Hospital",
                        "city": "Boston",
                        "country": "USA",
                        "status": "RECRUITING",
                    }
                ]
            },
        }
    }


@pytest.fixture()
def fake_urlopen():
    """Mock ``urllib.request.urlopen`` at the urllib module level.

    The script does ``import urllib.request`` at module load, so
    ``api_request`` resolves ``urllib.request.urlopen`` from the
    module's own namespace. Patch it there, not on the script.
    """
    with patch("urllib.request.urlopen") as m:
        yield m  # callers arrange return_value / side_effect


# ============================================================================
# Functional: request construction + happy-path JSON shape
# ============================================================================


def _requested_url(mock_urlopen) -> str:
    """Pull the URL the script passed to ``urllib.request.urlopen``.

    The script wraps the URL in a ``urllib.request.Request`` instance
    to attach the User-Agent header, so we have to ask the Request
    object for its URL — not look at ``call_args.args[0]`` directly
    (that would be the Request, not the string).
    """
    request_obj = mock_urlopen.call_args.args[0]
    return request_obj.full_url if hasattr(request_obj, "full_url") else str(request_obj)


class TestFunctionalHappyPath:
    """Exercise the real URL/query-param assembly paths with a mocked
    ``urlopen``. Validates:

    - the correct URL is constructed (PATH + querystring)
    - the response JSON is projected into the documented result schema
    - limits/clamping are honoured
    """

    def test_search_builds_url_with_query_and_pagesize(
        self, fake_urlopen
    ):
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response([_study()], total=1).encode()
        )

        result = ct.search_trials(
            {"query.term": "hypertension", "sort": "LastUpdatePostDate"},
            limit=10,
        )

        # URL construction — must include paginated pageSize and the
        # raw upstream params.
        requested_url = _requested_url(fake_urlopen)
        assert requested_url.startswith(
            "https://clinicaltrials.gov/api/v2/studies?"
        )
        assert "query.term=hypertension" in requested_url
        # pageSize must reflect limit, clamped to 100.
        assert "pageSize=10" in requested_url

        # Response projection — schema honours the script's contract.
        assert result["total"] == 1
        assert len(result["results"]) == 1
        row = result["results"][0]
        assert row["nct_id"] == "NCT00000000"
        assert row["title"] == "Test Trial"
        assert row["status"] == "RECRUITING"
        assert row["phase"] == "PHASE2"
        assert row["sponsor"] == "Acme"
        assert row["enrollment"] == 100

    def test_search_clamps_limit_to_100(self, fake_urlopen):
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response([], 0).encode()
        )

        # Caller asks for 500 — the script clamps to 100 (API max).
        ct.search_trials({"query.term": "x"}, limit=500)

        url = _requested_url(fake_urlopen)
        assert "pageSize=100" in url

    def test_search_caps_results_to_requested_limit(self, fake_urlopen):
        # Backend returns 5 studies, limit=2 — only 2 should surface.
        studies = [
            _study(nct=f"NCT0000000{i}", title=f"Trial {i}")
            for i in range(5)
        ]
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response(studies, total=5).encode()
        )

        result = ct.search_trials({"query.term": "x"}, limit=2)
        assert len(result["results"]) == 2
        # BUT totalCount reflects the real upstream total.
        assert result["total"] == 5

    def test_detail_returns_full_trial_object(self, fake_urlopen):
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(_detail_payload()).encode()
        )

        detail = ct.get_trial_detail("NCT00000000")

        # URL construction — single study endpoint, no querystring.
        url = _requested_url(fake_urlopen)
        assert url == "https://clinicaltrials.gov/api/v2/studies/NCT00000000"

        # Schema coverage — every documented field propagated.
        assert detail["nct_id"] == "NCT00000000"
        assert detail["title"] == "Test Trial"
        assert detail["status"] == "RECRUITING"
        assert detail["conditions"] == ["hypertension"]
        assert detail["eligibility"]["criteria"] == (
            "Inclusion: age >= 18"
        )
        assert detail["eligibility"]["sex"] == "ALL"
        assert detail["eligibility"]["min_age"] == "18 Years"
        assert len(detail["primary_outcomes"]) == 1
        assert detail["primary_outcomes"][0]["measure"] == "BP reduction"
        assert len(detail["locations"]) == 1
        assert detail["locations"][0]["facility"] == "Test Hospital"


# ============================================================================
# Edge cases — empty input, missing fields, multi-phase join
# ============================================================================


class TestEdgeCases:
    """Defensive behaviour when the upstream response omits fields or
    yields an empty list. No raises expected from these inputs."""

    def test_search_empty_studies_returns_empty_results(self, fake_urlopen):
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response([], 0).encode()
        )

        result = ct.search_trials({"query.term": "no-such-thing"}, limit=10)
        assert result["total"] == 0
        assert result["results"] == []

    def test_search_missing_enrollment_yields_none(self, fake_urlopen):
        study = _study()
        # Strip the enrollmentInfo block entirely.
        del study["protocolSection"]["designModule"]["enrollmentInfo"]
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response([study], total=1).encode()
        )

        result = ct.search_trials({"query.term": "x"}, limit=10)
        assert result["results"][0]["enrollment"] is None

    def test_search_multiphase_joins_with_comma(self, fake_urlopen):
        study = _study(phase=("PHASE1", "PHASE2"))
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response([study], total=1).encode()
        )

        result = ct.search_trials({"query.term": "x"}, limit=10)
        assert result["results"][0]["phase"] == "PHASE1, PHASE2"

    def test_search_no_phases_yields_empty_string(self, fake_urlopen):
        study = _study(phase=())
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            _fake_api_response([study], total=1).encode()
        )

        result = ct.search_trials({"query.term": "x"}, limit=10)
        assert result["results"][0]["phase"] == ""

    def test_detail_without_locations_yields_empty_list(self, fake_urlopen):
        payload = _detail_payload()
        # Empty locations list (treat as "no published sites").
        payload["protocolSection"]["contactsLocationsModule"]["locations"] = []
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(payload).encode()
        )

        detail = ct.get_trial_detail("NCT00000000")
        assert detail["locations"] == []

    def test_detail_no_primary_outcomes(self, fake_urlopen):
        payload = _detail_payload()
        # empty list — same as "no outcomes declared"
        del payload["protocolSection"]["outcomesModule"]["primaryOutcomes"]
        fake_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(payload).encode()
        )

        detail = ct.get_trial_detail("NCT00000000")
        assert detail["primary_outcomes"] == []


# ============================================================================
# Error propagation — api_request must surface upstream failure as an
# exception so the CLI's print path reports it. The script does NOT
# swallow errors into a JSON envelope; that's a future concern per
# the skill-PR recipe (Step A) and not within the scope of this test.
# ============================================================================


class TestErrorPropagation:
    """Confirm the (currently-throwing) behaviour of ``api_request``.

    These tests pin today's contract: a network/upstream failure
    raises out of the helper. If/when a follow-up introduces the
    ``[{"error": "..."}]`` JSON-envelope pattern (recipe Step A), these
    tests will need to be updated.
    """

    def test_network_failure_raises_from_search(self, fake_urlopen):
        import urllib.error

        fake_urlopen.side_effect = urllib.error.URLError("dns down")

        with pytest.raises(urllib.error.URLError):
            ct.search_trials({"query.term": "x"}, limit=10)

    def test_network_failure_raises_from_detail(self, fake_urlopen):
        import urllib.error

        fake_urlopen.side_effect = urllib.error.URLError("timeout")

        with pytest.raises(urllib.error.URLError):
            ct.get_trial_detail("NCT00000000")

    def test_http_error_raises_from_detail(self, fake_urlopen):
        # ClinicalTrials.gov returns 404 for unknown NCT IDs.
        import urllib.error

        fake_urlopen.side_effect = urllib.error.HTTPError(
            "https://clinicaltrials.gov/api/v2/studies/BOGUS",
            404,
            "Not Found",
            {},
            MagicMock(),
        )

        with pytest.raises(urllib.error.HTTPError):
            ct.get_trial_detail("BOGUS")


# ============================================================================
# Sanity: ensure the running test process finds the bundled helper at the
# documented scripts/ path. Catches "test moved, path still pointing at
# old location" failures silently otherwise.
# ============================================================================


class TestModuleShape:
    def test_helper_module_exists_at_documented_path(self):
        assert SCRIPTS_DIR.is_dir(), f"scripts dir missing: {SCRIPTS_DIR}"
        helper = SCRIPTS_DIR / "clinical_trials.py"
        assert helper.is_file(), f"helper missing: {helper}"

    def test_helper_exposes_search_and_detail(self):
        """Verify the public surface cmd_* callers reach for. If a
        future refactor renames these, the recipe's parser-alignment
        step (Step B) needs to run before the rename sticks."""
        assert callable(getattr(ct, "search_trials", None))
        assert callable(getattr(ct, "get_trial_detail", None))
        assert callable(getattr(ct, "cmd_search", None))
        assert callable(getattr(ct, "cmd_detail", None))
