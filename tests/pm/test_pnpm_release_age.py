"""`hermes pm update --pnpm` drops the release-age excludes a refreshed lock no longer holds."""
from pm.pnpm_release_age import prune_excludes

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
