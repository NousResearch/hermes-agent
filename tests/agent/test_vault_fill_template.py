"""Fill values containing template markers must remain literal data."""
import json
from agent.vault_login_classifier import build_fill_js


def test_template_markers_inside_value_are_not_substituted():
    value = 'left__NONCE__right__FILLS____EXPECTED_ORIGIN__'
    source = build_fill_js([{'index': 0, 'value': value}],
                           expected_origin='https://example.com', nonce='inspection')
    assert json.dumps(value) in source
