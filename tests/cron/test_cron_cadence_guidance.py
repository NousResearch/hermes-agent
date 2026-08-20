"""Regression coverage for cron cadence-out-of-prompt guidance.

Ported from RooCodeInc/Roomote#1218 ("Automation prompts repeat their
configured cadence"): freeform scheduling requests mix timing with the work
to perform, and models were storing the cadence in the job prompt too. The
stored prompt must describe only the work; timing lives exclusively in the
'schedule' field. These tests pin the tool-level guidance and the
prompt-field schema description that instruct the model accordingly.
"""

from tools.cronjob_tools import CRONJOB_SCHEMA


class TestCadenceGuidance:
    def test_tool_description_tells_model_to_keep_cadence_out_of_prompt(self):
        desc = CRONJOB_SCHEMA["description"]
        # The guidance must name both halves of the contract: cadence stays
        # out of the prompt, and timing lives only in the schedule field.
        assert "cadence OUT of the prompt" in desc
        assert "'schedule' field" in desc

    def test_tool_description_explains_the_mixed_request_case(self):
        desc = CRONJOB_SCHEMA["description"]
        # A concrete mixed timing+work example so the model knows how to
        # split a freeform request rather than just being told "don't".
        assert "mixes timing with work" in desc

    def test_prompt_field_description_forbids_cadence(self):
        prompt_desc = CRONJOB_SCHEMA["parameters"]["properties"]["prompt"][
            "description"
        ]
        assert "Never repeat scheduling cadence" in prompt_desc
        assert "'schedule'" in prompt_desc

    def test_schedule_field_still_required_on_create(self):
        # The split only works if schedule remains the mandatory home for
        # timing — pin that the schema still requires it on create.
        schedule_desc = CRONJOB_SCHEMA["parameters"]["properties"]["schedule"][
            "description"
        ]
        assert "REQUIRED for action=create" in schedule_desc
