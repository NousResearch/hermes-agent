"""`hermes pm update --pnpm` drops the release-age excludes a refreshed lock no longer holds."""
import pytest

from hermes_agent import hermes_yaml
from pm.pnpm_release_age import exclude_hint, prune_excludes

LOCK = """lockfileVersion: '9.0'
packages:
  '@scope/kept@2.0.0':
    resolution: {integrity: sha512-x}
  narrowed@1.1.0:
    resolution: {integrity: sha512-x}
  peered@3.0.0(react@19.0.0):
    resolution: {integrity: sha512-x}
"""


def test_prune_keeps_locked_entries_and_their_inline_reasons(tmp_path):
    workspace, lock = tmp_path / 'pnpm-workspace.yaml', tmp_path / 'pnpm-lock.yaml'
    lock.write_text(LOCK, encoding='utf-8')
    workspace.write_text(
        "minimumReleaseAge: 20160\n"
        "# Exemptions.\n"
        "minimumReleaseAgeExclude:\n"
        "  - gone@1.0.0  # reason gone\n"
        "  - '@scope/kept@2.0.0'  # reason kept\n"
        "  - narrowed@1.0.0 || 1.1.0  # reason narrowed\n"
        "  - peered@3.0.0\n"
        "  - '@glob/*'\n"
        "overrides:\n"
        "  gone@1.0.0: 1.0.1\n", encoding='utf-8')

    assert prune_excludes(workspace, lock) == ['gone@1.0.0', 'narrowed@1.0.0 || 1.1.0']
    assert workspace.read_text(encoding='utf-8') == (
        "minimumReleaseAge: 20160\n"
        "# Exemptions.\n"
        "minimumReleaseAgeExclude:\n"
        "  - '@scope/kept@2.0.0'  # reason kept\n"
        "  - narrowed@1.1.0  # reason narrowed\n"
        "  - peered@3.0.0\n"
        "  - '@glob/*'\n"
        "overrides:\n"
        "  gone@1.0.0: 1.0.1\n")


def test_prune_leaves_a_fully_locked_list_byte_for_byte(tmp_path):
    workspace, lock = tmp_path / 'pnpm-workspace.yaml', tmp_path / 'pnpm-lock.yaml'
    lock.write_text(LOCK, encoding='utf-8')
    text = "minimumReleaseAgeExclude:\r\n  - narrowed@1.1.0  # why\r\n"
    workspace.write_bytes(text.encode())
    assert prune_excludes(workspace, lock) == []
    assert workspace.read_bytes() == text.encode()


@pytest.mark.parametrize('quote', ["'", '"'])
def test_prune_keeps_a_narrowed_scoped_entry_in_its_quotes(tmp_path, quote):
    # A plain YAML scalar can't start with `@`: unquoted, pnpm can't load the workspace at all.
    workspace, lock = tmp_path / 'pnpm-workspace.yaml', tmp_path / 'pnpm-lock.yaml'
    lock.write_text(LOCK, encoding='utf-8')
    workspace.write_text(f"minimumReleaseAgeExclude:\n  - {quote}@scope/kept@1.0.0 || 2.0.0{quote}  # why\n", encoding='utf-8')
    assert prune_excludes(workspace, lock) == ['@scope/kept@1.0.0 || 2.0.0']
    text = workspace.read_text(encoding='utf-8')
    assert text == f"minimumReleaseAgeExclude:\n  - {quote}@scope/kept@2.0.0{quote}  # why\n"
    assert hermes_yaml.safe_load(text) == {'minimumReleaseAgeExclude': ['@scope/kept@2.0.0']}


def test_hint_lines_paste_as_the_exclude_list_they_name():
    picks = ['@scope/young@1.0.0', 'young@1.0.0', '!bang@1.0.0', "it's@1.0.0", '*star@1.0.0', 'yes', '1.0']
    hint = exclude_hint(picks)
    assert '    - young@1.0.0  # ' in hint  # a safe name stays plain
    pasted = [line for line in hint.splitlines() if line.lstrip().startswith('- ')]
    assert hermes_yaml.safe_load('minimumReleaseAgeExclude:\n' + '\n'.join(pasted)) == {'minimumReleaseAgeExclude': picks}
