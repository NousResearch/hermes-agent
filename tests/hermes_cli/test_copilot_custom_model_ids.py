"""Regression for #110597: Copilot custom/enterprise (BYOK) model ids must survive normalization.

GitHub's enterprise custom-model (BYOK) catalog exposes Copilot ids shaped ``owner/sub/model``
— that is, TWO slashes. With no candidate matching the (unreachable) catalog,
``normalize_copilot_model_id`` stripped the FIRST path segment and returned the result:
``acme-github-copilot/GLM/glm-5.2`` became ``GLM/glm-5.2``, the CLI's second normalization pass
stripped again to ``glm-5.2``, and the Copilot API answered HTTP 400 ``model_not_supported``.
Every such call failed, for both custom models a tenant had configured.
"""

import pytest

from hermes_cli.models import normalize_copilot_model_id


# ``owner/sub/model`` — the shape an enterprise BYOK custom model has (tenant name sanitized).
CUSTOM_MODEL_IDS = [
    "acme-github-copilot/QDeepseekV4/deepseek-flash",
    "acme-github-copilot/GLM/glm-5.2",
    "acme-github-copilot/HUAWEI/glm-5.2",
]


@pytest.mark.parametrize("model_id", CUSTOM_MODEL_IDS)
def test_two_slash_custom_model_id_is_passed_through(model_id):
    """A two-slash id IS a Copilot id — normalize nothing, catalog or no catalog."""
    # ``hermes_cli/model_normalize.py`` calls this without an api_key, so the catalog is always
    # empty on the runtime path that produced the 400.
    assert normalize_copilot_model_id(model_id, catalog=[], api_key=None) == model_id
    assert normalize_copilot_model_id(
        model_id, catalog=[{"id": model_id}], api_key=None) == model_id


def test_single_vendor_prefix_still_folds_to_the_bare_id():
    """Contract preserved for the one-slash case this stripping exists for (#6879)."""
    assert normalize_copilot_model_id(
        "anthropic/claude-sonnet-4.6", catalog=[], api_key=None) == "claude-sonnet-4.6"
