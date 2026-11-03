"""Tools-only publication jobs must provision the assembler's YAML dependency."""
from pathlib import Path
import shlex

import hermes_yaml


def test_channel_publisher_provisions_pinned_yaml_before_running_the_entrypoint(monkeypatch):
    from scripts.ci import python_packages, setup_toolchain
    import pm

    workflow = hermes_yaml.safe_load((Path(__file__).resolve().parents[2]
                                     / '.github/workflows/stable-release.yml').read_text(encoding='utf-8'))
    steps = workflow['jobs']['publish-channel']['steps']
    command, = [shlex.split(step['run']) for step in steps
                if 'scripts.releases.channel_claims' in step.get('run', '')]
    assert command[:3] == ['python', '-m', 'scripts.ci.python_packages']
    selected = Path('fixture-python')
    provisioned, invoked = [], []

    def ensure(name, requirements, *, explicit):
        provisioned.append((name, requirements, explicit))
        return selected

    monkeypatch.setattr(pm, 'ensure_environment', ensure)
    monkeypatch.setattr(setup_toolchain, 'python3_alias', lambda python: None)
    monkeypatch.setattr(python_packages.subprocess, 'call', lambda argv, **kwargs: invoked.append(argv) or 0)
    assert python_packages.main(command[3:]) == 0
    assert provisioned == [('ci-tools', ['ruamel.yaml==0.18.17'], True)]
    assert invoked == [[str(selected), '-m', 'scripts.releases.channel_claims', 'publish']]
