"""No-provider responses equal frozen raw bytes from pure main, not regenerated here."""
import json
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from hermes_cli.skill_catalog.registry import snapshot
from tests.hermes_cli._skill_catalog_stock_capture import BASE, STATES, capture, seed
from tests.hermes_cli.test_skill_catalog_contract import make_plugin
from tests.hermes_cli.test_skill_catalog_installed import _provider


@pytest.mark.parametrize('state', STATES)
def test_no_provider_http_matches_pure_main(_isolate_hermes_home, state):
    home = get_hermes_home()
    frozen = json.loads((Path(__file__).parents[1] / 'fixtures/skill_catalog/stock_profiles.json').read_text())
    assert frozen['base_sha'] == BASE
    seed(home, 'never-used' if state == 'previously-disabled' else state)
    if state == 'previously-disabled':
        manager, _record = _provider(home, target='local')
        assert snapshot(scope=str(home))
        manager.unload()
    elif state == 'flag-off':
        make_plugin(home)
        # Disabled config is written before the HTTP request, not a fake registry.
        seed(home, state)
    assert snapshot(scope=str(home)) == ()
    actual = capture(home, state)
    assert actual == frozen['states'][state]
    assert snapshot(scope=str(home)) == ()
    assert (home / 'skills/local/SKILL.md').read_text().endswith('Local\n')
